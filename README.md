# SPLIT-ICML
Official code for the ICML 2025 Spotlight paper "Near Optimal Decision Trees in a SPLIT Second" 

# Installation Instructions
This repository contains implementations of both packages used and described in the paper. Both packages and their respective dependencies can be installed via the following. 
```bash
pip install resplit/ split/
```

To use SPLIT / LicketySPLIT:

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

To run RESPLIT


```python
from resplit import RESPLIT
import pandas as pd
dataset = pd.read_csv('path/to/compas.csv') 
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
config = {
    "regularization": 0.001,
    "rashomon_bound_multiplier": 0.02,
    "depth_budget": 5,
    'cart_lookahead_depth': 3,
    "verbose": False
}
model = RESPLIT(config, fill_tree = "treefarms")
# find the set of near optimal lookahead prefixes, and fill each leaf of each prefix with another TreeFARMS Rashomon set. See function for more options.
model.fit(X,y)
tree = model[0]
y_pred = tree.predict(X)
```
We also note the other options in the config which are most commonly used:

1. `rashomon_bound_adder`: A replacement for `rashomon_bound_multiplier`. It sets the Rashomon set threshold as the set of all models which are within `L* + ε` of the best loss `L*`.
