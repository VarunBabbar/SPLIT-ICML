import pandas as pd
import numpy as np
from split import ThresholdGuessBinarizer, GOSDTClassifier
from split._tree import Node, Leaf
import argparse
from sklearn.model_selection import train_test_split
import time
from .utils import num_leaves, tree_to_dict

'''
Working notes: 
- start with including a depth budget, though eventually we can remove this
- removing this may require tracking whether leaves have converged (i.e. not expanded on the last call)
    - that also makes the current implementation faster

'''

LOOKAHEAD_RANGE = 2

# version with no automatic guessing
class LicketySPLIT: 
    def __init__(self, time_limit=60, verbose=True, reg=0.001, 
                 full_depth_budget = 6, lookahead_range=LOOKAHEAD_RANGE,
                 similar_support=False, allow_small_reg=True, binarize = False, 
                 gbdt_n_est=50, gbdt_max_depth=1):

         #2 corresponds to one-step lookahead
        if lookahead_range < 2:
            raise ValueError("lookahead_range must be at least 2")
        self.has_no_depth_limit = False
        if full_depth_budget == 0: # no depth limit at all to the full tree
            self.has_no_depth_limit = True
        elif full_depth_budget < lookahead_range:
            raise ValueError("full_depth_budget must be at least 2 (or 0 for no limit)")

        self.config = {
            "regularization": reg,
            "depth_budget": full_depth_budget,
            "time_limit": time_limit,
            "similar_support": similar_support,
            "verbose": verbose, 
            'allow_small_reg': allow_small_reg,
            'cart_lookahead_depth': lookahead_range,
        }
        self.lookahead_range = lookahead_range
        self.classes = None
        self.tree = None
        self.binarize = binarize
        if self.binarize:
            self.enc = ThresholdGuessBinarizer(n_estimators=gbdt_n_est, 
                                           max_depth=gbdt_max_depth, 
                                           random_state=2021)
            self.enc.set_output(transform="pandas")
            X_train = self.enc.fit_transform(X_train, y_train)
        self.verbose = verbose
        self.similar_support = similar_support
        self.allow_small_reg = allow_small_reg
        self.time_limit = time_limit
        self.full_depth_budget = full_depth_budget
        self.reg = reg
        

    def fit(self, X_train: pd.DataFrame, y_train): #does initial fit, then calls helper
        clf = GOSDTClassifier(**self.config)
        clf.fit(X_train, y_train)
        self.classes = clf.classes_.tolist()
        tree = clf.trees_[0].tree

        if not self.config['depth_budget'] < self.lookahead_range or self.has_no_depth_limit: # otherwise, one convergence condition
            n = X_train.shape[0]
            child_config = self.config.copy()
            child_config['depth_budget'] = self.config['depth_budget'] - LOOKAHEAD_RANGE + 1 # we know this doesn't decrease config to 0 because of the condition we're in
            tree = self._fill_leaves(tree, X_train, y_train, n, child_config)
        
        self.tree = tree

    def _recursive_fit(self, X_train, y_train, config): 
        clf = GOSDTClassifier(**config)
        clf.fit(X_train, y_train)
        tree = self.extract_tree(clf)
        if not config['depth_budget'] < self.lookahead_range or self.has_no_depth_limit: # otherwise, one convergence condition 
            n = X_train.shape[0]
            child_config = config.copy()
            child_config['depth_budget'] = config['depth_budget'] - LOOKAHEAD_RANGE + 1 # we know this doesn't decrease config to 0 because of the condition we're in
            tree = self._fill_leaves(tree, X_train, y_train, n, child_config)
        

        return tree
    
    def _fill_leaves(self, tree, X_train, y_train, n, child_config):
        if isinstance(tree, Leaf):
            #rescale regularization to be the same as the original model
            # despite training on a subset of the data
            config = child_config.copy()
            config['regularization'] = config['regularization'] * n/len(y_train)

            # fit a GOSDT classifier to the data in this leaf
            return self._recursive_fit(X_train, y_train, config)
        else:
            X_left = X_train[X_train.iloc[:, tree.feature] == True]
            y_left = y_train[X_train.iloc[:, tree.feature] == True]
            X_right = X_train[X_train.iloc[:, tree.feature] == False]
            y_right = y_train[X_train.iloc[:, tree.feature] == False]
            tree.left_child = self._fill_leaves(tree.left_child, X_left, y_left, n, child_config)
            tree.right_child = self._fill_leaves(tree.right_child, X_right, y_right, n, child_config)
        return tree

    def remap_tree(self, tree, tree_classes): # WORKING NOTES: change this to return another object that has an is_converged flag
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
    
    def _predict_sample(self, x_i, node):
        if isinstance(node, Leaf):
            return self.classes[node.prediction]
        elif x_i[node.feature]:
            return self._predict_sample(x_i, node.left_child)
        else:
            return self._predict_sample(x_i, node.right_child)

    def predict(self, X_test: pd.DataFrame):
        if self.tree is None:
            raise ValueError("Model has not been trained to have a valid tree yet")
        X_values = X_test.values
        return np.array([self._predict_sample(X_values[i, :], self.tree)
                         for i in range(X_values.shape[0])])
        
    def tree_to_dict(self):
        if self.tree is None:
            raise ValueError("Model has not been trained to have a valid tree yet")
        return self._tree_to_dict(self.tree)
    
    def _tree_to_dict(self, node): 
        if isinstance(node, Leaf):
            return {'prediction': self.classes[node.prediction]}
        else:
            return {"feature": node.feature,
                   "True": self._tree_to_dict(node.left_child),
                   "False": self._tree_to_dict(node.right_child)
            }


class RecursionWrapper(LicketySPLIT):
    def __init__(self, gbdt_n_est=40, gbdt_max_depth=1, reg = 0.001, 
                 lookahead_range=LOOKAHEAD_RANGE, time_limit=60, 
                 full_depth_budget = 6, similar_support=False, verbose=True, 
                 allow_small_reg=True):
        self.enc = ThresholdGuessBinarizer(n_estimators=gbdt_n_est, 
                                           max_depth=gbdt_max_depth, 
                                           random_state=2021)
        self.enc.set_output(transform="pandas")
        super().__init__(reg=reg, similar_support=similar_support, 
              time_limit=time_limit,
              lookahead_range=lookahead_range, 
              full_depth_budget=full_depth_budget,
              verbose=verbose,
              allow_small_reg=allow_small_reg)
        

    def fit(self, X_train: pd.DataFrame, y_train):
        # Guess Thresholds
        X_train_guessed = self.enc.fit_transform(X_train, y_train)
        # No LB guess for now - want it to be the same model as the self.enc transform fitter

        # Train the GOSDT classifier
        super().fit(X_train_guessed, y_train)
    

    def predict(self, X_test: pd.DataFrame):
        X_test_guessed = self.enc.transform(X_test)
        return super().predict(X_test_guessed)
       

def test_lookahead_exact(): 
    data = pd.DataFrame({'a': [1, 1, 0, 0, 1], 
                      'b': [1, 0, 1, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = LicketySPLIT(time_limit=60, verbose=True, reg=0.001)
    # import pdb; pdb.set_trace()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)

def test_lookahead(): 
    data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
                      'b': [1, 4, 3, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = RecursionWrapper(gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
                             time_limit=60, verbose=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)

def test_lookahead_and_wrapper_db_1(): 
    data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
                      'b': [1, 4, 3, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = RecursionWrapper(gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
                             time_limit=60, verbose=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)


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
    args = parser.parse_args()

    # Read the dataset
    df = pd.read_csv(f'datasets/{args.dataset}.csv')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)
    # Train the model
    start = time.time()
    if args.no_guess:
        model = LicketySPLIT(time_limit=args.time_limit, 
                                  verbose=args.verbose, 
                                  reg=args.reg, 
                                  full_depth_budget=args.depth_budget)
    else:
        model = RecursionWrapper(gbdt_n_est=args.gbdt_n_est, gbdt_max_depth=args.gbdt_max_depth, 
                                 reg=args.reg, time_limit=args.time_limit, verbose=args.verbose, 
                                 full_depth_budget=args.depth_budget)
    model.fit(X_train, y_train)
    print(f"Initialization/Training time: {time.time()-start}")

    # Evaluate the model
    pred_start_time = time.time()
    y_pred = model.predict(X_test)
    print(f"Test Prediction time: {time.time()-pred_start_time}")
    print("Train_acc: " + str(sum(model.predict(X_train) == y_train)/len(y_train)))
    print(f"Test accuracy: {sum(y_pred == y_test)/len(y_test)}")

    print("Number of leaves in the tree: ", num_leaves(model.tree_to_dict()))
