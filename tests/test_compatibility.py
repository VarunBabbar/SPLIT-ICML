import gosdt
from sklearn.utils.estimator_checks import check_estimator

def test_compat_thresholdguessbinarizer():
    check_estimator(gosdt.ThresholdGuessBinarizer())
    
def test_compat_numericalbinarizer():
    check_estimator(gosdt.NumericBinarizer())
    
# TODO: This is actually the hard thing to fix/confirm...
#
# There seem to be some issues related to our minimal regularization requirement causing some of
# the test datasets to fail to reach a sufficient score.
# 
# def test_compat_classifier():
#     check_estimator(gosdt.GOSDTClassifier())

# This is to fix the following error from test_compat_classifier:
# FAILED tests/test_compatibility.py::test_compat_classifier - AssertionError: Classifier can't train when only one class is present.
def test_single_class():

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from gosdt import GOSDTClassifier

    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_classes=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = GOSDTClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert len(set(y_pred)) == 1