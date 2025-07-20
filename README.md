# SPLIT-ICML
Official code for the ICML 2025 Oral paper: **"Near Optimal Decision Trees in a SPLIT Second"**.


[Check out the paper here](https://arxiv.org/pdf/2502.15988)

Note that this code is in beta. Based on positive reception at ICML we plan to create more accessible versions of the code soon - please stay tuned! In the meantime, this code is here for data science practitioners, researchers, and students interested in trying our method and/or replicating our paper results. If you encounter problems, please feel free to leave a github issue.

This is an open source project; contributions are welcome!

# Installation Instructions and Quickstart

This repository contains implementations of both packages used and described in the paper. First `cd` into this repository. Then install both SPLIT and RESPLIT via the following. 
```bash
pip install resplit/ split/
```

To run SPLIT:

```python
from split import SPLIT
import pandas as pd
lookahead_depth = 2
depth_buget = 5
dataset = pd.read_csv('path/to/compas.csv') 
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
# y should correspond to a binary class label. 
regularization = 0.01
model = SPLIT(lookahead_depth_budget=lookahead_depth, reg=regularization, full_depth_budget=depth_buget, verbose=False, binarize=False,time_limit=100)
# set binarize = True if dataset is not binarized.
model.fit(X,y)
y_pred = model.predict(X)
tree = model.tree
print(tree)
```
To run LicketySPLIT:
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
# Options for fill_tree: "treefarms", "optimal", "greedy".
# "treefarms" will fill each leaf of each prefix with another TreeFARMS Rashomon set.
# "optimal" will complete prefixes using GOSDT.
# "greedy" will do so using greedy completions. 
model.fit(X,y)
i = 0
tree = model[i] # get the ith tree
print(tree)
y_pred = model.predict(X,i) # predictions for the ith tree
```

### Notes on Using RESPLIT

1. **Command Line Preferred**  
   For now, we recommend running RESPLIT via a command-line script (e.g., `python3 run_resplit_on_compas.py`) or a SLURM script rather than in a Jupyter notebook.  
   We have observed some timeout issues in Jupyter and are actively investigating these.

2. **Quick Runtime Comparison**  
   There is an example in `resplit/example/` where you can first run:

   ```bash
   pip install treefarms
   python3 resplit_example.py
   ```
   This will demonstrate the difference in runtime between TreeFARMS and RESPLIT.

### Common Config Options for RESPLIT

- **`rashomon_bound_adder`**  
  An alternative to `rashomon_bound_multiplier`. It sets the Rashomon set threshold as the set of all models within `L* + ε` of the best loss `L*`.

- **`rashomon_bound`**  
  Another alternative to `rashomon_bound_multiplier`. It sets the Rashomon set threshold as the set of all models within a fixed loss value. This is a hard threshold rather than a relative `ε`.

For more config options, check out the README in the `resplit` directory.
