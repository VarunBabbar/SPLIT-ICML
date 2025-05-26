# SPLIT-ICML
Official code for the ICML 2025 Spotlight paper "Near Optimal Decision Trees in a SPLIT Second" 

# Installation Instructions
This repository contains implementations of both packages used and described in the paper. First `cd` into this repository. Then install both SPLIT and RESPLIT via the following. 
```bash
pip install resplit/ split/
```

To use SPLIT:

```python
from split import SPLIT
import pandas as pd
lookahead_depth = 2
depth_buget = 5
dataset = pd.read_csv('path/to/compas.csv') 
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
regularization = 0.01
model = SPLIT(lookahead_depth_budget=lookahead_depth, reg=regularization, full_depth_budget=depth_buget, verbose=False, binarize=False,time_limit=100)
# set binarize = True if dataset is not binarized.
model.fit(X,y)
y_pred = model.predict(X)
tree = model.tree
print(tree)
```
To run LicketySPLIT
```python
from split import LicketySPLIT
model = LicketySPLIT(full_depth_budget=full_depth_budget,reg=regularization)
.... # same as above
...
```

To run RESPLIT:


```python
from resplit import RESPLIT
import pandas as pd
dataset = pd.read_csv('path/to/compas.csv') 
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
config = {
    "regularization": 0.005,
    "rashomon_bound_multiplier": 0.01, # Sets the Rashomon set threshold as the set of all models which are within `(1+ε)L*` of the best loss `L*`.
    "depth_budget": 5,
    'cart_lookahead_depth': 3,
    "verbose": False
}
model = RESPLIT(config, fill_tree = "treefarms")
# Options for fill_tree: "treefarms", "optimal", "greedy". "treefarms" will fill each leaf of each prefix with another TreeFARMS Rashomon set. "optimal" will complete prefixes using GOSDT. "greedy" will do so using greedy completions. 
model.fit(X,y)
tree = model[i] # get the ith tree
print(tree)
y_pred = model.predict(X,i) # predictions for the ith tree
```
For now, we recommend running RESPLIT via a command line script (e.g. python3 run_resplit_on_compas.py) or a slurm script rather than in a Jupyter notebook. We have observed some timeout issues in Jupyter and are investigating these actively.  

We also note the other options in the config which are most commonly used:

1. `rashomon_bound_adder`: A alternative to `rashomon_bound_multiplier`. It sets the Rashomon set threshold as the set of all models which are within `L* + ε` of the best loss `L*`.
2. `rashomon_bound`: An alternative to `rashomon_bound_multiplier`. It sets the Rashomon set threshold as the set of all models which are within the rashomon bound. This is a hard loss instead of a relative `ε` threshold.


For more config options, check out the README in the `resplit` directory.
