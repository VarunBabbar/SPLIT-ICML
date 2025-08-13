from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
import pandas as pd
import numpy as np
from split import ThresholdGuessBinarizer, GOSDTClassifier
from split._tree import Node, Leaf


class LicketySplitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=None,
                 class_weight=None,
                 ccp_alpha=0.0,
                 lookahead=2,
                 depth_budget=6,
                 regularization_scale=1.0,
                 binarize=False,
                 binarization_strategy='threshold_guess'):
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.lookahead = lookahead
        self.depth_budget = depth_budget
        self.regularization_scale = regularization_scale
        self.binarize = binarize
        self.binarization_strategy = binarization_strategy
        
    def _train_core(self, X, y, sample_weight, rng):
        if self.binarize:
            if not hasattr(self, 'enc_'):
                self.enc_ = ThresholdGuessBinarizer(
                    n_estimators=50, 
                    max_depth=1, 
                    random_state=rng.randint(0, 2**31-1)
                )
                self.enc_.set_output(transform="pandas")
            X_bin = self.enc_.fit_transform(X, y)
        else:
            X_bin = X
            
        config = {
            "regularization": 0.001 * self.regularization_scale,
            "depth_budget": self.depth_budget,
            "time_limit": 60,
            "similar_support": False,
            "verbose": False,
            'allow_small_reg': True,
            'cart_lookahead_depth': self.lookahead,
        }
        
        clf = GOSDTClassifier(**config)
        clf.fit(X_bin, y, sample_weight=sample_weight)
        self.classes_ = clf.classes_.tolist()
        tree = clf.trees_[0].tree
        
        if not config['depth_budget'] < self.lookahead or self.depth_budget == 0:
            n = X_bin.shape[0]
            child_config = config.copy()
            child_config['depth_budget'] = config['depth_budget'] - self.lookahead + 1
            tree = self._fill_leaves(tree, X_bin, y, n, child_config, sample_weight)
            
        return tree
        
    def _fill_leaves(self, tree, X_train, y_train, n, child_config, sample_weight):
        if isinstance(tree, Leaf):
            config = child_config.copy()
            config['regularization'] = config['regularization'] * n/len(y_train)
            
            clf = GOSDTClassifier(**config)
            clf.fit(X_train, y_train, sample_weight=sample_weight)
            expanded_tree = clf.trees_[0].tree
            expanded_classes = clf.classes_
            return self._remap_tree(expanded_tree, expanded_classes)
        else:
            mask_left = X_train.iloc[:, tree.feature] == True
            mask_right = X_train.iloc[:, tree.feature] == False
            
            X_left = X_train[mask_left]
            y_left = y_train[mask_left]
            sw_left = sample_weight[mask_left] if sample_weight is not None else None
            
            X_right = X_train[mask_right]
            y_right = y_train[mask_right]
            sw_right = sample_weight[mask_right] if sample_weight is not None else None
            
            tree.left_child = self._fill_leaves(tree.left_child, X_left, y_left, n, child_config, sw_left)
            tree.right_child = self._fill_leaves(tree.right_child, X_right, y_right, n, child_config, sw_right)
        return tree
        
    def _remap_tree(self, tree, tree_classes):
        if isinstance(tree, Leaf):
            return Leaf(
                prediction=self.classes_.index(tree_classes[tree.prediction]), 
                loss=tree.loss
            )
        else:
            return Node(
                tree.feature, 
                self._remap_tree(tree.left_child, tree_classes), 
                self._remap_tree(tree.right_child, tree_classes)
            )
    
    def fit(self, X, y, sample_weight=None):
        rng = check_random_state(self.random_state)
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = np.array(y)
        
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            
        self.tree_ = self._train_core(X, y, sample_weight, rng)
        return self
        
    def predict(self, X):
        # TODO: Implement predict method
        pass
        
    def predict_proba(self, X):
        # TODO: Implement predict_proba method
        pass