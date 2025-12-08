import pandas as pd
import numpy as np
from split._tree import Node, Leaf
import argparse
from sklearn.model_selection import train_test_split
import time

# version with no automatic guessing
class GreedyWrapper: 
    def __init__(self,reg=0.001, depth_budget = 5):
        
        self.depth_budget = depth_budget
        self.reg = reg
        self.classes_ = [0,1]
        self.has_no_depth_limit = False # flag for if the full tree has no depth limit
        
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
    
    def fit(self, X_train: pd.DataFrame, y_train):
        '''
        Requires X_train to be binary
        '''
        self.n = X_train.shape[0]
        tree, loss = self.train_greedy(X_train,y_train,self.depth_budget,self.reg)
        self.tree = tree
        # fill each leaf with a split classifier
        
    
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
                    loss = left_loss + right_loss
                    node.left_child = left_node
                    node.right_child = right_node
                    node.feature = best_feature
                else:
                    node = Leaf(prediction = y_pred, loss = loss-reg)
            else:
                node = Leaf(prediction = y_pred, loss = loss-reg)
        else:
            node = Leaf(prediction = y_pred, loss = loss-reg)
        return node, loss

    def _predict_sample(self, x_i, node):
        if isinstance(node, Leaf):
            return self.classes_[node.prediction]
        elif x_i[node.feature]:
            return self._predict_sample(x_i, node.left_child)
        else:
            return self._predict_sample(x_i, node.right_child)

    def predict(self, X_test: pd.DataFrame):
        X_values = X_test.values
        return np.array([self._predict_sample(X_values[i, :], self.tree)
                         for i in range(X_values.shape[0])])

    def tree_to_dict(self):
        return self._tree_to_dict(self.tree)

    def _tree_to_dict(self, node): 
        if isinstance(node, Leaf):
            return {'prediction': self.classes_[node.prediction]}
        else:
            return {"feature": node.feature,
                   "true": self._tree_to_dict(node.left_child),
                   "false": self._tree_to_dict(node.right_child)
            }
    def num_leaves(self,tree_as_dict=None):
        if tree_as_dict is None:
            tree_as_dict = self.tree_to_dict()
        if 'prediction' in tree_as_dict:
            return 1
        else:
            return self.num_leaves(tree_as_dict['True']) + self.num_leaves(tree_as_dict['False'])
            
lamda = 0.001
depth_budget = 5
greedy_tree = GreedyWrapper(depth_budget = depth_budget,reg = lamda)
greedy_tree.fit(X_train,y_train)
y_pred = greedy_tree.predict(X_train)
print((y_pred != y_train).mean())
print(greedy_tree.tree_to_dict())
print(greedy_tree.num_leaves())

