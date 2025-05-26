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
full_depth_budget = 5
regularization = 0.01
model = SPLIT(lookahead_depth_budget=lookahead_depth, time_limit=100, reg=regularization, full_depth_budget=full_depth_budget, verbose=False, binarize=False)
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
