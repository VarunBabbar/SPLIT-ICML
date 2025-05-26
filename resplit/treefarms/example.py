import pandas as pd
import numpy as np
import time
import pathlib
from treefarms import TREEFARMS

# read the dataset
df = pd.read_csv("experiments/datasets/compas/binned.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]
h = df.columns[:-1]

config = {
    "regularization": 0.01,  # regularization penalizes the tree with more leaves. We recommend to set it to relative high value to find a sparse tree.
    "rashomon_bound_multiplier": 0.05,  # rashomon bound multiplier indicates how large of a Rashomon set would you like to get
}

model = TREEFARMS(config)

model.fit(X, y)

first_tree = model[0]

print("evaluating the first model in the Rashomon set", flush=True)

# get the results
train_acc = first_tree.score(X, y)
n_leaves = first_tree.leaves()
n_nodes = first_tree.nodes()

print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(first_tree)
