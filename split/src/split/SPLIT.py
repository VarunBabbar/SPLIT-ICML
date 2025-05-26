import pandas as pd
import numpy as np
from split import ThresholdGuessBinarizer, GOSDTClassifier
from split._tree import Node, Leaf
import argparse
from sklearn.model_selection import train_test_split
import time
from .utils import num_leaves, tree_to_dict

class SPLIT: 
    def __init__(self, time_limit=60, verbose=False, reg=0.001, 
                 lookahead_depth_budget=3, full_depth_budget = 6,
                 similar_support=False, allow_small_reg=True, greedy_postprocess = False, 
                 binarize = False, gbdt_n_est=50, gbdt_max_depth=1, 
                 ):

        # check that lookahead is a valid argument: 
        if lookahead_depth_budget == 1:
            print(f"Warning: lookahead {lookahead_depth_budget} equals 1. " +
                  "This means the lookahead will be ignored (depth 1 is just the root).")
            filtered_lookahead_depth_budget = 0
        else:
            filtered_lookahead_depth_budget = lookahead_depth_budget
        self.clf = GOSDTClassifier(depth_budget=full_depth_budget, 
                                   time_limit=time_limit, 
                                   verbose=verbose, 
                                   regularization=reg, 
                                   similar_support=similar_support, 
                                   allow_small_reg=allow_small_reg, 
                                   cart_lookahead_depth=filtered_lookahead_depth_budget, 
                                   )

        # Standard way of handling expansion parameters for depth: 
        self.remaining_depth = full_depth_budget - filtered_lookahead_depth_budget+1 # depth budget for expanding leaves
        self.has_no_depth_limit = False # flag for if the full tree has no depth limit

        self.greedy_postprocess = greedy_postprocess

        # Edge cases for depth:
        if filtered_lookahead_depth_budget == 0: # 0 is the encoding for no lookahead
            self.remaining_depth = 0 
        elif full_depth_budget == 0: # no depth limit at all to the full tree, just to lookahead 
            self.has_no_depth_limit = True
        elif filtered_lookahead_depth_budget > full_depth_budget: 
            print(f"Warning: lookahead {filtered_lookahead_depth_budget} exceeds full_depth_budget {full_depth_budget}." +
                  f"This means the call is equivalent to optimizing a tree without lookahead, using full depth {filtered_lookahead_depth_budget}")

        self.leaf_config = {
            "regularization": reg,
            "depth_budget": 0 if self.has_no_depth_limit else self.remaining_depth,
            "time_limit": time_limit,
            "similar_support": similar_support,
            "verbose": verbose, 
            'allow_small_reg': allow_small_reg
        }
        self.binarize = binarize
        if self.binarize: # binarize using threshold guessing
            self.enc = ThresholdGuessBinarizer(n_estimators=gbdt_n_est,
                                                  max_depth=gbdt_max_depth, 
                                                    random_state=42)
            self.enc.set_output(transform="pandas")
            # fit 
            X_train = self.enc.fit_transform(X_train, y_train)

        self.verbose = verbose
        self.time_limit = time_limit
                                            

            

    def fit(self, X_train: pd.DataFrame, y_train):
        '''
        Requires X_train to be binary
        '''
        self.clf.fit(X_train, y_train)
        self.classes = self.clf.classes_.tolist()
        # fill each leaf with a split classifier
        self.n = X_train.shape[0]
        if self.remaining_depth > 0 or self.has_no_depth_limit:
            if self.greedy_postprocess:
                self.tree = self.fill_leaves_with_greedy(self.clf.trees_[0].tree, X_train, y_train)
            else:
                self.tree = self.fill_leaves(self.clf.trees_[0].tree, X_train, y_train)
        else:
            self.tree = None

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

    def entropy(self,ps):
        """
        Calculate the entropy of a given list of binary labels.
        """
        p_positive = ps[0]
        if p_positive == 0 or p_positive == 1:
            return 0  # Entropy is 0 if all labels are the same
        entropy_val = - (p_positive * np.log2(p_positive) +
                         (1 - p_positive) * np.log2(1 - p_positive))
        return entropy_val

    def find_best_feature_to_split_on(self, X_train,y_train):
        num_features = X_train.shape[1]
        max_gain = -10
        gain_of_feature_to_split = 0
        p_original = np.mean(y_train)
        entropy_original = self.entropy([p_original, 1-p_original])
        best_feature = -1
        for feature in range(num_features):
            # Left child labels
            p_left = np.mean(y_train[X_train.iloc[:, feature] == 1])
            
            # Right child labels
            p_right = np.mean(y_train[X_train.iloc[:, feature] == 0])

            p_left = 0 if np.isnan(p_left) else p_left
            p_right = 0 if np.isnan(p_right) else p_right
        
            entropy_left = self.entropy(np.array([p_left, 1 - p_left]))
            
            entropy_right = self.entropy(np.array([p_right, 1 - p_right]))
            
            proportion_of_examples_in_left_leaf = (np.sum(X_train.iloc[:, feature] == 1) / len(X_train))
            proportion_of_examples_in_right_leaf = (np.sum(X_train.iloc[:, feature] == 0) / len(X_train))
            gain = entropy_original - ( proportion_of_examples_in_left_leaf* entropy_left +
                                        proportion_of_examples_in_right_leaf* entropy_right)
            if gain >= max_gain:
                max_gain = gain
                best_feature = feature

        return best_feature

    def train_greedy(self, X_train,y_train,depth_budget,reg):
        node = Node(feature = None, left_child = None, right_child = None)

        # take majority label
        flag = True
        if len(y_train) > 0:
            y_pred = int(y_train.mean()>0.5)
            loss = (y_pred != y_train).sum()/self.n + reg
        else:
            loss = 0
            y_pred = 0
            flag = False

        if depth_budget > 1 and flag: 
            best_feature = self.find_best_feature_to_split_on(X_train,y_train)
            X_train_left = X_train[X_train.iloc[:, best_feature] == True]
            y_train_left = y_train[X_train.iloc[:, best_feature] == True]

            X_train_right = X_train[X_train.iloc[:, best_feature] == False]
            y_train_right = y_train[X_train.iloc[:, best_feature] == False]
            
            if len(X_train_left) != 0 and len(X_train_right) != 0:
                reg_left = reg*len(y_train)/(len(y_train_left)) # option to add this
                reg_right = reg*len(y_train)/(len(y_train_right))
                
                left_node, left_loss = self.train_greedy(X_train_left, y_train_left, depth_budget-1,reg)
                right_node, right_loss = self.train_greedy(X_train_right, y_train_right, depth_budget-1,reg)
                
                if left_loss + right_loss < loss: # only split if it improves the loss
                    node.left_child = left_node
                    node.right_child = right_node
                    node.feature = best_feature
                    loss = left_loss + right_loss
                else:
                    node = Leaf(prediction = y_pred, loss = loss-reg)
            else:
                node = Leaf(prediction = y_pred, loss = loss-reg)
        else:
            node = Leaf(prediction = y_pred, loss = loss-reg)
        return node, loss

    def fill_leaves_with_greedy(self,tree,X_train,y_train):
        if isinstance(tree, Leaf):
            node, loss = self.train_greedy(X_train, y_train, self.remaining_depth, self.leaf_config['regularization'] * self.n/len(y_train))
            return node
        else:
            X_left = X_train[X_train.iloc[:, tree.feature] == True]
            y_left = y_train[X_train.iloc[:, tree.feature] == True]
            X_right = X_train[X_train.iloc[:, tree.feature] == False]
            y_right = y_train[X_train.iloc[:, tree.feature] == False]
            tree.left_child = self.fill_leaves_with_greedy(tree.left_child, X_left, y_left)
            tree.right_child = self.fill_leaves_with_greedy(tree.right_child, X_right, y_right)
        return tree

    def fill_leaves(self, tree, X_train, y_train):
        if isinstance(tree, Leaf):
            #rescale regularization to be the same as the original model
            # despite training on a subset of the data
            config = self.leaf_config.copy()
            config['regularization'] = config['regularization'] * self.n/len(y_train)

            # fit a gosdt classifier to the data in this leaf
            leaf_clf = GOSDTClassifier(**config)
            leaf_clf.fit(X_train, y_train)

            #extract and return the tree from the leaf classifier
            leaf_clf_as_tree = self.extract_tree(leaf_clf)
            return leaf_clf_as_tree
        else:
            X_left = X_train[X_train.iloc[:, tree.feature] == True]
            y_left = y_train[X_train.iloc[:, tree.feature] == True]
            X_right = X_train[X_train.iloc[:, tree.feature] == False]
            y_right = y_train[X_train.iloc[:, tree.feature] == False]
            tree.left_child = self.fill_leaves(tree.left_child, X_left, y_left)
            tree.right_child = self.fill_leaves(tree.right_child, X_right, y_right)
        return tree

    def _predict_sample(self, x_i, node):
        if isinstance(node, Leaf):
            return self.clf.classes_[node.prediction]
        elif x_i[node.feature]:
            return self._predict_sample(x_i, node.left_child)
        else:
            return self._predict_sample(x_i, node.right_child)

    def predict(self, X_test: pd.DataFrame):
        if self.tree is None:
            return self.clf.predict(X_test)
        else:
            X_values = X_test.values
            return np.array([self._predict_sample(X_values[i, :], self.tree)
                             for i in range(X_values.shape[0])])

    def tree_to_dict(self):
        if self.tree is None:
            return tree_to_dict(self.clf.trees_[0].tree, self.clf.classes_)
        else: 
            return self._tree_to_dict(self.tree)

    def _tree_to_dict(self, node): 
        if isinstance(node, Leaf):
            return {'prediction': self.clf.classes_[node.prediction]}
        else:
            return {"feature": node.feature,
                   "True": self._tree_to_dict(node.left_child),
                   "False": self._tree_to_dict(node.right_child)
            }

# class LookaheadWrapper(SPLIT):
#     def __init__(self, gbdt_n_est=40, gbdt_max_depth=1, reg = 0.001, 
#                  lookahead_depth_budget=3, time_limit=60, 
#                  full_depth_budget = 6, similar_support=False, verbose=True, 
#                  allow_small_reg=True):
#         self.enc = ThresholdGuessBinarizer(n_estimators=gbdt_n_est, 
#                                            max_depth=gbdt_max_depth, 
#                                            random_state=2021)
#         self.enc.set_output(transform="pandas")
#         super().__init__(reg=reg, similar_support=similar_support, 
#               time_limit=time_limit,
#               lookahead_depth_budget=lookahead_depth_budget, 
#               full_depth_budget=full_depth_budget,
#               verbose=verbose,
#               allow_small_reg=allow_small_reg)
        

#     def fit(self, X_train: pd.DataFrame, y_train):
#         # Guess Thresholds
#         X_train_guessed = self.enc.fit_transform(X_train, y_train)
#         # No LB guess for now - want it to be the same model as the self.enc transform fitter

#         # Train the split classifier
#         super().fit(X_train_guessed, y_train)
    

#     def predict(self, X_test: pd.DataFrame):
#         X_test_guessed = self.enc.transform(X_test)
#         return super().predict(X_test_guessed)
       

# def test_lookahead_exact(): 
#     data = pd.DataFrame({'a': [1, 1, 0, 0, 1], 
#                       'b': [1, 0, 1, 0, 1], 
#                       'y': [1, 0, 0, 1, 1]})
#     y = data['y']
#     X = data.drop(columns='y')
#     model = SPLIT(lookahead_depth_budget=2, time_limit=60, verbose=True, reg=0.001)
#     model.fit(X, y)
#     preds = model.predict(X)
#     assert np.all(preds == y)

# def test_lookahead_exact_depth_3(): 
#     data = pd.DataFrame({
#                       'a': [1, 1, 0, 0, 1, 0, 1, 0, 1], 
#                       'b': [1, 0, 1, 0, 1, 0, 1, 1, 0], 
#                       'c': [1, 0, 1, 0, 1, 1, 0, 0, 1], 
#                       'y': [1, 1, 0, 0, 1, 1, 0, 1, 0]})
#     y = data['y']
#     X = data.drop(columns='y')
#     model = SPLIT(lookahead_depth_budget=3, full_depth_budget=6,
#                                    time_limit=60, verbose=True, reg=0.001)
#     model.fit(X, y) # core question - is the prefix, before being filled in, actually optimal given cart being filled in? or is the cart heuristic just being evaluated as a leaf? 
#     preds = model.predict(X)
#     assert np.all(preds == y)

# def test_lookahead(): 
#     data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
#                       'b': [1, 4, 3, 0, 1], 
#                       'y': [1, 0, 0, 1, 1]})
#     y = data['y']
#     X = data.drop(columns='y')
#     model = LookaheadWrapper(gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
#                              lookahead_depth_budget=3, time_limit=60, verbose=True)
#     model.fit(X, y)
#     preds = model.predict(X)
#     assert np.all(preds == y)

# def test_lookahead_and_wrapper_db_1(): 
#     data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
#                       'b': [1, 4, 3, 0, 1], 
#                       'y': [1, 0, 0, 1, 1]})
#     y = data['y']
#     X = data.drop(columns='y')
#     model = LookaheadWrapper(gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
#                              lookahead_depth_budget=1, time_limit=60, verbose=True)
#     model.fit(X, y)
#     preds = model.predict(X)
#     assert np.all(preds == y)


if __name__ == "__main__":
    # read in parameters with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbdt_n_est", type=int, default=40)
    parser.add_argument("--gbdt_max_depth", type=int, default=1)
    parser.add_argument("-l", "--reg", type=float, default=0.001)
    parser.add_argument("-d", "--depth_budget", type=int, default=6)
    parser.add_argument("-t", "--time_limit", type=int, default=60)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dataset", type=str, default="compas")
    parser.add_argument("--no_guess", action="store_true")
    parser.add_argument("--lookahead", type=int, default=3)
    args = parser.parse_args()

    # Read the dataset
    df = pd.read_csv(f'datasets/{args.dataset}.csv')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)
    # Train the model
    start = time.time()
    if args.no_guess:
        model = SPLIT(lookahead_depth_budget=args.lookahead, 
                                  time_limit=args.time_limit, 
                                  verbose=args.verbose, 
                                  reg=args.reg, 
                                  full_depth_budget=args.depth_budget)
    # else:
    #     model = LookaheadWrapper(lookahead_depth_budget=args.lookahead, 
    #                              gbdt_n_est=args.gbdt_n_est, gbdt_max_depth=args.gbdt_max_depth, 
    #                              reg=args.reg, time_limit=args.time_limit, verbose=args.verbose, 
    #                              full_depth_budget=args.depth_budget)
    model.fit(X_train, y_train)
    print(f"Initialization/Training time: {time.time()-start}")

    # Evaluate the model
    pred_start_time = time.time()
    y_pred = model.predict(X_test)
    print(f"Test Prediction time: {time.time()-pred_start_time}")
    print("Train_acc: " + str(sum(model.predict(X_train) == y_train)/len(y_train)))
    print(f"Test accuracy: {sum(y_pred == y_test)/len(y_test)}")

    print("Number of leaves in the tree: ", num_leaves(model.tree_to_dict()))
