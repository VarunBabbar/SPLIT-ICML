# SPLIT-ICML
Official code for the ICML 2025 Spotlight paper "Near Optimal Decision Trees in a SPLIT Second" 

# Instructions

For now, run 
```bash
pip install split
```

To run SPLIT
```python
from split import SPLIT
lookahead_depth = 2
depth_buget = 5
dataset = pd.read_csv('compas.csv')
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
full_depth_budget = 5
regularization = 0.01
model = SPLIT(lookahead_depth_budget=lookahead_depth, time_limit=args.time_limit, reg=regularization, full_depth_budget=full_depth_budget, verbose=False, binarize=False) # set binarize = True if dataset is not binarized.
model.fit(X,y)
y_pred = model.predict(X)
```
To run LicketySPLIT
```python
from split import LicketySPLIT
model = LicketySPLIT(full_depth_budget=full_depth_budget,reg=regularization)
```
