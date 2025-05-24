import os
import pandas as pd
import numpy as np

from split import NumericBinarizer 

tests_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(tests_dir)
datasets_dir = os.path.join(root_dir, "datasets/")   


def test_spiral():
    spiral = pd.read_csv(datasets_dir + "spiral.csv")
    X = spiral.iloc[:, :-1]
    enc = NumericBinarizer()
    Xt = enc.fit_transform(X)
    assert Xt.shape[0] == 100
    assert Xt.shape[1] == 180
    assert np.array_equal(X, enc.inverse_transform(Xt))


def test_compas():
    compas = pd.read_csv(datasets_dir + "compas.csv")
    X = compas.iloc[:, :-1]
    enc = NumericBinarizer()
    Xt = enc.fit_transform(X)
    assert Xt.shape[0] == 6907
    assert Xt.shape[1] == 134
    assert np.array_equal(X, enc.inverse_transform(Xt))


def test_broward():
    broward = pd.read_csv(datasets_dir + "broward_general_2y.csv")
    X = broward.iloc[:, :-1]
    enc = NumericBinarizer()
    Xt = enc.fit_transform(X)
    assert Xt.shape[0] == 1954
    assert Xt.shape[1] == 588
    assert np.array_equal(X, enc.inverse_transform(Xt))


def test_iris():
    from sklearn.datasets import load_iris
    iris = load_iris().data
    enc = NumericBinarizer()
    Xt = enc.fit_transform(iris)
    assert Xt.shape[0] == 150
    assert Xt.shape[1] == 119
    assert np.array_equal(iris, enc.inverse_transform(Xt))
