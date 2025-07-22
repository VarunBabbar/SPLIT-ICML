import pandas as pd
import numpy as np
from split import SPLIT, LicketySPLIT

'''
Runs several basic tests of SPLIT.
Instructions: run `pip install pytest`in the SPLIT virtual environment
then `pytest test.py` from this repository's root directory
'''

def test_lookahead_exact(): 
    data = pd.DataFrame({'a': [1, 1, 0, 0, 1], 
                      'b': [1, 0, 1, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = SPLIT(lookahead_depth_budget=2, time_limit=60, verbose=True, reg=0.001)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)

def test_lookahead_exact_depth_3(): 
    data = pd.DataFrame({
                      'a': [1, 1, 0, 0, 1, 0, 1, 0, 1], 
                      'b': [1, 0, 1, 0, 1, 0, 1, 1, 0], 
                      'c': [1, 0, 1, 0, 1, 1, 0, 0, 1], 
                      'y': [1, 1, 0, 0, 1, 1, 0, 1, 0]})
    y = data['y']
    X = data.drop(columns='y')
    model = SPLIT(lookahead_depth_budget=3, full_depth_budget=6,
                                   time_limit=60, verbose=True, reg=0.001)
    model.fit(X, y) # core question - is the prefix, before being filled in, actually optimal given cart being filled in? or is the cart heuristic just being evaluated as a leaf? 
    preds = model.predict(X)
    assert np.all(preds == y)

def test_encoder(): 
    data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
                      'b': [1, 4, 3, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = SPLIT(binarize=True, gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
                    lookahead_depth_budget=3, time_limit=60, verbose=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)

def test_lookahead_and_encode_db_1(): 
    data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
                      'b': [1, 4, 3, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = SPLIT(binarize=True, gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
                             lookahead_depth_budget=1, time_limit=60, verbose=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)

def test_lookahead_exact(): 
    data = pd.DataFrame({'a': [1, 1, 0, 0, 1], 
                      'b': [1, 0, 1, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = LicketySPLIT(time_limit=60, verbose=True, reg=0.001)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)

def test_lookahead(): 
    data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
                      'b': [1, 4, 3, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = LicketySPLIT(gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
                             time_limit=60, verbose=True, binarize=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)

def test_lookahead_and_wrapper_db_1(): 
    data = pd.DataFrame({'a': [1, 1, 4, 0, 1], 
                      'b': [1, 4, 3, 0, 1], 
                      'y': [1, 0, 0, 1, 1]})
    y = data['y']
    X = data.drop(columns='y')
    model = LicketySPLIT(gbdt_n_est=2, gbdt_max_depth=1, reg=0.001, 
                             time_limit=60, verbose=True, binarize=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == y)