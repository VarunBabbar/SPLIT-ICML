from split._tree import Node, Leaf
from typing import Union

def num_leaves(tree_as_dict) -> int:
    if tree_as_dict is None:
        print("Tree is None")
        return -1
    if 'prediction' in tree_as_dict:
        return 1
    else:
        return num_leaves(tree_as_dict['True']) + num_leaves(tree_as_dict['False'])
    

def tree_to_dict(tree: Union[Node, Leaf], classes):
    return _tree_to_dict(tree, classes)

def _tree_to_dict(node, classes): 
    if isinstance(node, Leaf):
        return {'prediction': classes[node.prediction]}
    else:
        return {"feature": node.feature,
                "True": _tree_to_dict(node.left_child, classes),
                "False": _tree_to_dict(node.right_child, classes)
        }