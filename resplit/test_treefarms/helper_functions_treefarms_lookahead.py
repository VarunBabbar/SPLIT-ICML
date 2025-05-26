# This file contains helper functions for the treefarms_lookahead.py file

from gosdt import GOSDTClassifier
from gosdt._tree import Node, Leaf
import gosdt

def _num_leaves(tree_as_dict) -> int:
    if tree_as_dict is None:
        print("Tree is None")
        return -1
    if 'prediction' in tree_as_dict:
        return 1
    else:
        return _num_leaves(tree_as_dict['True']) + _num_leaves(tree_as_dict['False'])


def get_num_leaves_greedy(model):
    if isinstance(model, Leaf):
        return 1
    return get_num_leaves_greedy(model.left_child) + get_num_leaves_greedy(model.right_child)


def _tree_to_dict(node, classes):
    if isinstance(node, gosdt._tree.Leaf):
        return {'prediction': classes[node.prediction]}
    else:
        return {"feature": node.feature,
                "True": _tree_to_dict(node.left_child, classes),
                "False": _tree_to_dict(node.right_child, classes)
                }

def get_num_leaves_gosdt(model):
    if isinstance(model, GOSDTClassifier):
        clf = model
    else:
        clf = model.clf
    model_obj = clf.trees_[0].tree
    model_dict = _tree_to_dict(model_obj, [0, 1])
    return _num_leaves(model_dict)


def are_trees_same(tree1, tree2):
    # check if both trees are the same
    if isinstance(tree1, Leaf) and isinstance(tree2, Leaf):
        return tree1.prediction == tree2.prediction

    if isinstance(tree1, Node) and isinstance(tree2, Node):
        return tree1.feature == tree2.feature and are_trees_same(tree1.left_child, tree2.left_child) and are_trees_same(tree1.right_child, tree2.right_child)
