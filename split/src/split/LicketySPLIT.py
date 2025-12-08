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

class LicketySPLIT: 
    def __init__(self, time_limit=60, verbose=False, reg=0.001, 
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
        self.verbose = verbose
        self.similar_support = similar_support
        self.allow_small_reg = allow_small_reg
        self.time_limit = time_limit
        self.full_depth_budget = full_depth_budget
        self.reg = reg
        

    def fit(self, X_train: pd.DataFrame, y_train): #does initial fit, then calls helper
        '''
        Requires X_train to be binary, or for self.binarize to be true
        (in the latter case X_train is binarized according to the threshold
        guessing transform, as fit on X_train)
        '''
        if self.binarize:
            X_train_bin = self.enc.fit_transform(X_train, y_train)
        else:
            X_train_bin = X_train
        clf = GOSDTClassifier(**self.config)
        clf.fit(X_train_bin, y_train)
        self.classes = clf.classes_.tolist()
        tree = clf.trees_[0].tree

        if not self.config['depth_budget'] < self.lookahead_range or self.has_no_depth_limit: # otherwise, one convergence condition
            n = X_train_bin.shape[0]
            child_config = self.config.copy()
            child_config['depth_budget'] = self.config['depth_budget'] - LOOKAHEAD_RANGE + 1 # we know this doesn't decrease config to 0 because of the condition we're in
            tree = self._fill_leaves(tree, X_train_bin, y_train, n, child_config)
        
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
        '''
        Requires X_train to be binary
        '''
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
        '''
        Requires X_test to be binary, or for self.binarize to be true
        '''
        if self.binarize: # apply binarizer
            X_test_bin = self.enc.transform(X_test)
        else:
            X_test_bin = X_test
        if self.tree is None:
            raise ValueError("Model has not been trained to have a valid tree yet")
        X_values = X_test_bin.values
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
                   "true": self._tree_to_dict(node.left_child),
                   "false": self._tree_to_dict(node.right_child)

            }
