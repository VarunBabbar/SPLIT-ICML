import builtins
import sys
import resplit
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from split import GOSDTClassifier
from split._tree import Node, Leaf
import split
from resplit import TREEFARMS
from resplit.helper_functions_resplit import are_trees_same
from resplit.helper_functions_resplit import get_num_leaves_gosdt, _tree_to_dict
import os

def get_num_leaves_greedy(model):
    if isinstance(model, Leaf):
        return 1
    return get_num_leaves_greedy(model.left_child) + get_num_leaves_greedy(model.right_child)

class RESPLIT(resplit.model.treefarms.TREEFARMS):
    def __init__(self, config, load_path=False, fill_tree = 'optimal',save_trie_tmp = True):
        """
        Initialize the RESPLIT algorithm, a hybrid method that first enumerates near-optimal prefix trees 
        using TREEFARMS and then fills their leaves using one of three strategies: greedy, optimal, or 
        TREEFARMS itself. This enables exploration of the Rashomon set through fine-grained subtree expansion.

        Parameters
        ----------
        config : dict
            A dictionary of hyperparameters and settings. Must include keys such as:
                - 'regularization': float
                - 'depth_budget': int
                - 'rashomon_bound_multiplier': float
                - (optional) 'fill_tree': {'greedy', 'optimal', 'treefarms'}

        load_path : str or bool, default=False
            If provided (i.e., not False), loads a precomputed Rashomon prefix set from the given path instead of recomputing it.
            The path should point to a pickled object containing a dict with key 'rset'.

        fill_tree : str, default='optimal'
            Strategy for filling leaves of the prefix trees. Options:
                - 'greedy'     : fill leaves using a greedy splitting heuristic
                - 'optimal'    : fill leaves using optimal trees (e.g., via GOSDT)
                - 'treefarms'  : fill leaves using TREEFARMS subtrees

        save_trie_tmp : bool, default=True
            If True, sets a temporary file path in `config['rashomon_trie']` to store the Rashomon prefix trie
            based on the current config values. Used internally for caching and reproducibility.

        Attributes
        ----------
        models : list
            A list of decision trees (or sets of trees, if `fill_tree='treefarms'`) generated from the Rashomon prefix set.

        num_models : int
            Total number of trees stored when `fill_tree='treefarms'`.

        num_models_per_prefix : list
            Number of trees generated for each prefix when using TREEFARMS subtree expansion.

        classes : list
            Class labels used for prediction and remapping.

        rashomon_set_prefix : TREEFARMS
            The initial Rashomon prefix tree generator (either computed or loaded from disk).
        """
        self.config = config
        if 'fill_tree' not in self.config:
            self.fill_tree = fill_tree
        else:
            self.fill_tree = self.config['fill_tree'] # greedy, optimal, treefarms
        if save_trie_tmp:
            self.config['rashomon_trie'] = os.path.join(
            'tmp/rashomon_trie_{}_{}_{}.json'.format(config['depth_budget'], config['rashomon_bound_multiplier'], config['regularization']))

        self.load_path = load_path
        if not self.load_path:
            self.rashomon_set_prefix = TREEFARMS(self.config)
        else:
            self.rashomon_set_prefix = pd.read_pickle(
                open(load_path, 'rb'))['rset']
        # self.fill_tree = fill_tree 
        self.models = []
        self.num_models = 0
        self.num_models_per_prefix = []
        self.classes = [0,1]

    def fit(self, X, y):
        self.n = X.shape[0]

        if not self.load_path:
            self.rashomon_set_prefix.fit(X, y) 

        self.remaining_depth = self.config['depth_budget'] - \
            self.config['cart_lookahead_depth']+1
        print("Found set of near optimal prefixes. Filling in their leaves now.")
        for i in tqdm(range(self.rashomon_set_prefix.get_tree_count())):
            tree_dict = vars(self.rashomon_set_prefix[i])['source']
            tree = self.dict_to_tree(tree_dict, X, y)
            if self.fill_tree == 'greedy':
                tree = self.fill_leaves_with_greedy(tree, X, y)
                tree_pred = np.array([self._predict_sample(
                    X.values[i, :], tree) for i in range(X.shape[0])])
                tree_loss = (y != tree_pred).mean()
                tree_leaves = get_num_leaves_greedy(tree)
                obj = tree_loss + self.config['regularization']*tree_leaves
                self.models.append(tree)
            elif self.fill_tree == 'optimal':
                tree = self.fill_leaves_with_optimal(tree, X, y)
                tree_pred = np.array([self._predict_sample(
                    X.values[i, :], tree) for i in range(X.shape[0])])
                tree_loss = (y != tree_pred).mean()
                tree_leaves = get_num_leaves_greedy(tree)
                obj = tree_loss + self.config['regularization']*tree_leaves
                self.models.append(tree)
            elif self.fill_tree == 'treefarms':
                start_time = time.time()
                trees = self.fill_leaves_with_treefarms(tree, X, y)
                end_time = time.time()
                print("Time taken to fill leaves with treefarms: ",
                      end_time-start_time)
                num_models, trees = self.enumerate_treefarms_subtrees(trees)
                self.num_models += num_models
                self.num_models_per_prefix.append(num_models)
                self.models.append(trees)
        if self.fill_tree == 'treefarms':
            self.hash = self.hash_for_indexing(self.models)
    
    def save_trie(self, path):
        # TODO: Implement this
        
        return

    def hash_for_indexing(self,prefixes):
        num_models = self.__len__()
        hashes = {}
        for i in range(num_models):
            tree = self.get_item_for_hashing(i)
            hashes[i] = tree
        return hashes

    def __len__(self):
        if self.fill_tree == 'treefarms':
            return self.num_models
        else:
            return len(self.models)

    def __getitem__(self,idx):
        if self.fill_tree == 'treefarms':
            return self.get_item_for_hashing(idx)
        else:
            return self.models[idx]

    def get_item_for_hashing(self, idx):
        # find the appropriate prefix
        counter = 0
        num_trees = 0
        # TODO: Make this binary search.
        while num_trees <= idx and counter < len(self.models):
            num_trees += self.num_models_per_prefix[counter]
            counter += 1
        num_trees -= self.num_models_per_prefix[counter-1]
        self.appropriate_idx = (counter-1, idx-num_trees)
        return self.get_leaf_subtree_at_idx(self.models[counter-1], idx-num_trees)

    def get_leaf_subtree_at_idx(self, prefix_trie, idx):
        """
        Given an index, retrive the corresponding tree whose lookahead prefix is the prefix_trie
        """
        if isinstance(prefix_trie, Leaf):
            return prefix_trie

        if isinstance(prefix_trie, list):
            return prefix_trie[idx]

        tree = Node(feature=int(prefix_trie.feature),
                    left_child=None, right_child=None)
        left_count = prefix_trie.left_child[1]
        right_count = prefix_trie.right_child[1]
        right_idx = idx % right_count
        left_idx = idx//right_count
        tree.left_child = self.get_leaf_subtree_at_idx(
            prefix_trie.left_child[0], left_idx)
        tree.right_child = self.get_leaf_subtree_at_idx(
            prefix_trie.right_child[0], right_idx)

        return tree

    def enumerate_treefarms_subtrees(self, tree):
        """
        Returns the total number of tree structures induced by the cross
        product of subtree candidates in node.left_child and node.right_child.
        """

        # 1. Base cases
        if tree is None:
            return 1

        if isinstance(tree, list):  # list of subtrees rooted at the leaf of the prefix tree
            return len(tree), tree

        left_expansions, left_subtree = self.enumerate_treefarms_subtrees(
            tree.left_child)
        tree.left_child = (left_subtree, left_expansions)
        right_expansions, right_subtree = self.enumerate_treefarms_subtrees(
            tree.right_child)
        tree.right_child = (right_subtree, right_expansions)
        return left_expansions * right_expansions, tree

    def train_treefarms(self, X, y, feature_set='', **kwargs):
        """
        Train a TREEFARMS model on the given dataset. Output a list of trees as GOSDT Node objects
        """
        config = kwargs
        config['rashomon_trie'] = os.path.join('tmp/rashomon_subtrie_{}.json'.format(feature_set))
        model = TREEFARMS(config)
        model.fit(X, y)

        num_trees = model.get_tree_count()
        trees = []
        for i in range(num_trees):
            tree_dict = vars(model[i])['source']
            tree = self.dict_to_tree(tree_dict, X, y)
            trees.append(tree)
        return trees

    def fill_leaves_with_treefarms(self, tree, X, y,feature_set=''):
        """
        Fill the leaves of a prefix-tree with TREEFARMS models. 
        I.e. for a current leaf, the parent node in the prefix tree now points to the list of trees output by the train_treefarms function above
        """
        if isinstance(tree, Leaf):
            # compute rashomon bound multiplier
            leaf_config = {
                "regularization": min(0.1, self.config['regularization']*self.n/len(y)),
                "depth_budget": self.remaining_depth,
                "time_limit": 10,
                "similar_support": False,
                "verbose": False,
                'allow_small_reg': False,
            }
            greedy_tree, _ = self.train_greedy(
                X, y, leaf_config["depth_budget"], leaf_config['regularization'])
            greedy_preds = np.array([self._predict_sample(X.values[i, :], greedy_tree)
                                     for i in range(X.shape[0])])
            greedy_leaves = get_num_leaves_greedy(greedy_tree)
            greedy_loss = (y != greedy_preds).mean() + \
                leaf_config['regularization']*greedy_leaves
            # optimal_tree = GOSDTClassifier(**leaf_config)
            # optimal_tree.fit(X, y)
            # optimal_loss = (y != optimal_tree.predict(X)).mean(
            # ) + leaf_config['regularization']*get_num_leaves_gosdt(optimal_tree)
            rashomon_bound = greedy_loss
            # rashomon_bound = min(
            #     0.01, max(0, (greedy_loss - optimal_loss)/optimal_loss))
            # print("remaining depth: ", self.remaining_depth, min(
            #     0.1, self.config['regularization']*self.n/len(y)), rashomon_bound)
            treefarms_leaf_config = {'depth_budget': self.remaining_depth, 'regularization': min(0.1, self.config['regularization']*self.n/len(y)),
                  'rashomon_bound': rashomon_bound}
            if rashomon_bound <= self.config['regularization']*self.n/len(y) + 0.0001:
                node = [tree]
                return node
            else:
                model = self.train_treefarms(
                    X, y, feature_set, **treefarms_leaf_config)
                return model
        else:
            X_left = X[X.iloc[:, tree.feature] == True]
            y_left = y[X.iloc[:, tree.feature] == True]
            X_right = X[X.iloc[:, tree.feature] == False]
            y_right = y[X.iloc[:, tree.feature] == False]
            tree.left_child = self.fill_leaves_with_treefarms(
                tree.left_child, X_left, y_left, feature_set + str(tree.feature) + '_' + '1_')
            tree.right_child = self.fill_leaves_with_treefarms(
                tree.right_child, X_right, y_right, feature_set + str(tree.feature) + '_' + '0_')
        return tree

    def fill_leaves_with_greedy(self, tree, X, y):
        """
        Complete a given prefix tree output by treefarms greedily
        """
        if isinstance(tree, Leaf):
            node, loss = self.train_greedy(
                X, y, self.remaining_depth, self.config['regularization'])
            return node
        else:
            X_left = X[X.iloc[:, tree.feature] == True]
            y_left = y[X.iloc[:, tree.feature] == True]
            X_right = X[X.iloc[:, tree.feature] == False]
            y_right = y[X.iloc[:, tree.feature] == False]
            tree.left_child = self.fill_leaves_with_greedy(
                tree.left_child, X_left, y_left)
            tree.right_child = self.fill_leaves_with_greedy(
                tree.right_child, X_right, y_right)
        return tree

    def fill_leaves_with_optimal(self, tree, X, y):
        if isinstance(tree, Leaf):
            leaf_config = {
                "regularization": min(0.1, self.config['regularization']*self.n/len(y)),
                "depth_budget": self.remaining_depth,
                "time_limit": 10,
                "similar_support": False,
                "verbose": False,
                'allow_small_reg': False,
            }

            leaf_clf = GOSDTClassifier(**leaf_config)
            leaf_clf.fit(X, y)

            #extract and return the tree from the leaf classifier
            leaf_clf_as_tree = self.extract_tree(leaf_clf)
            return leaf_clf_as_tree
        else:
            X_left = X[X.iloc[:, tree.feature] == True]
            y_left = y[X.iloc[:, tree.feature] == True]
            X_right = X[X.iloc[:, tree.feature] == False]
            y_right = y[X.iloc[:, tree.feature] == False]
            tree.left_child = self.fill_leaves_with_optimal(
                tree.left_child, X_left, y_left)
            tree.right_child = self.fill_leaves_with_optimal(
                tree.right_child, X_right, y_right)
            return tree

    def remap_tree(self, tree, tree_classes):
        '''
        Helper to remap a tree to use the same class indices as the main tree
        '''
        if isinstance(tree, Leaf):
            return Leaf(prediction=self.classes.index(tree_classes[tree.prediction]), 
                        loss=tree.loss)
        else:
            return Node(tree.feature, 
                        self.remap_tree(tree.left_child, tree_classes), 
                        self.remap_tree(tree.right_child, tree_classes))

    def extract_tree(self, clf):
        '''
        Helper to take an expanded leaf classifier and extract a tree, 
        remapping to use the same class indices as the main tree 
        '''
        expanded_leaf = clf.trees_[0].tree
        expanded_leaf_classes = clf.classes_
        return self.remap_tree(expanded_leaf, expanded_leaf_classes)
    
    def train_gosdt(self, X, y, depth_budget, reg):

        model = GOSDTClassifier(depth_budget=depth_budget,
                                regularization=reg)
        model.fit(X, y)

        return model

    def train_greedy(self, X, y, depth_budget, reg):

        node = Node(feature=None, left_child=None, right_child=None)

        # take majority label
        flag = True
        if len(y) > 0:
            y_pred = int(y.mean() > 0.5)
            loss = (y_pred != y).sum()/self.n + reg
        else:
            loss = 0
            y_pred = 0
            flag = False

        if depth_budget > 1 and flag:
            best_feature = self.find_best_feature_to_split_on(X, y)
            X_left = X[X.iloc[:, best_feature] == True]
            y_left = y[X.iloc[:, best_feature] == True]

            X_right = X[X.iloc[:, best_feature] == False]
            y_right = y[X.iloc[:, best_feature] == False]

            if len(X_left) != 0 and len(X_right) != 0:
                reg_left = reg*len(y)/(len(y_left))  # option to add this
                reg_right = reg*len(y)/(len(y_right))

                left_node, left_loss = self.train_greedy(
                    X_left, y_left, depth_budget-1, reg)
                right_node, right_loss = self.train_greedy(
                    X_right, y_right, depth_budget-1, reg)
                if left_loss + right_loss < loss:  # only split if it improves the loss
                    loss = left_loss + right_loss
                    node.left_child = left_node
                    node.right_child = right_node
                    node.feature = best_feature
                else:
                    node = Leaf(prediction=y_pred, loss=loss-reg)
            else:
                node = Leaf(prediction=y_pred, loss=loss-reg)
        else:
            node = Leaf(prediction=y_pred, loss=loss-reg)

        return node, loss

    def _predict_sample(self, x_i, node):
        # do this for the entire dataset
        if isinstance(node, Leaf):
            return node.prediction
        elif x_i[node.feature]:
            return self._predict_sample(x_i, node.left_child)
        else:
            return self._predict_sample(x_i, node.right_child)

    def predict(self, X_test, idx):
        if type(X_test) == pd.DataFrame:
            X_values = X_test.values
        else:
            X_values = X_test
        return np.array([self._predict_sample(X_values[i, :], self.__getitem__(idx))
                         for i in range(X_values.shape[0])])

    def dict_to_tree(self, tree_dict, X, y):
        # Base case: if the dictionary represents a prediction, return a Leaf object
        if "prediction" in tree_dict:
            return Leaf(prediction=tree_dict["prediction"], loss=(y != int(tree_dict["prediction"])).sum()/self.n)

        # Recursive case: construct a Node object
        # Recursively build the true and false branches
        left_child = self.dict_to_tree(
            tree_dict["true"], X[X.iloc[:, tree_dict["feature"]] == 1], y[X.iloc[:, tree_dict["feature"]] == 1])
        right_child = self.dict_to_tree(
            tree_dict["false"], X[X.iloc[:, tree_dict["feature"]] == 0], y[X.iloc[:, tree_dict["feature"]] == 0])
        node = Node(feature=tree_dict["feature"],
                    left_child=left_child, right_child=right_child)

        return node

    def entropy(self, ps):
        """
        Calculate the entropy of a given list of binary labels.
        """
        p_positive = ps[0]
        if p_positive == 0 or p_positive == 1:
            return 0  # Entropy is 0 if all labels are the same
        entropy_val = - (p_positive * np.log2(p_positive) +
                         (1 - p_positive) * np.log2(1 - p_positive))
        return entropy_val

    def find_best_feature_to_split_on(self, X, y):
        num_features = X.shape[1]
        max_gain = -10
        gain_of_feature_to_split = 0
        p_original = np.mean(y)
        entropy_original = self.entropy([p_original, 1-p_original])
        best_feature = -1
        for feature in range(num_features):
            # Left child labels
            p_left = np.mean(y[X.iloc[:, feature] == 1])

            # Right child labels
            p_right = np.mean(y[X.iloc[:, feature] == 0])

            p_left = 0 if np.isnan(p_left) else p_left
            p_right = 0 if np.isnan(p_right) else p_right

            entropy_left = self.entropy(np.array([p_left, 1 - p_left]))

            entropy_right = self.entropy(np.array([p_right, 1 - p_right]))

            proportion_of_examples_in_left_leaf = (
                np.sum(X.iloc[:, feature] == 1) / len(X))
            proportion_of_examples_in_right_leaf = (
                np.sum(X.iloc[:, feature] == 0) / len(X))
            gain = entropy_original - (proportion_of_examples_in_left_leaf * entropy_left +
                                       proportion_of_examples_in_right_leaf * entropy_right)
            if gain >= max_gain:
                max_gain = gain
                best_feature = feature

        return best_feature

# Save the original print function if you might want to restore it later
# original_print = builtins.print

# # Replace the built-in print with a no-op (does nothing)
# builtins.print = lambda *args, **kwargs: None
