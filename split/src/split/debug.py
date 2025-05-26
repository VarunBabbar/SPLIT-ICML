import split
from sklearn.datasets import load_iris

# Load the dataset
X, y = load_iris(return_X_y=True, as_frame=True)
enc = split.NumericBinarizer()
X_bin = enc.set_output(transform='pandas').fit_transform(X)

# Train the GOSDT classifier
clf = split.GOSDTClassifier(regularization=0.1, verbose=True, debug=True)
clf.fit(X_bin, y)

# The fit method above will produce a `debug_(timestamp)` folder in the current directory.
# To use the `cli` target, run the following command:
# /Path/To/cli debug_(timestamp)/
