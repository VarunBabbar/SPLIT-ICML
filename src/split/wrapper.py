import pandas as pd
from split import ThresholdGuessBinarizer, splitClassifier
import argparse
from sklearn.model_selection import train_test_split
import time
from .utils import num_leaves, tree_to_dict


class splitWrapper:
    def __init__(self, gbdt_n_est=40, gbdt_max_depth=1, reg = 0.001, 
                 depth_budget=6, time_limit=60, similar_support=False, verbose=True):
        self.enc = ThresholdGuessBinarizer(n_estimators=gbdt_n_est, 
                                           max_depth=gbdt_max_depth, 
                                           random_state=2021)
        self.enc.set_output(transform="pandas")
        self.clf = splitClassifier(depth_budget=depth_budget, 
                                   time_limit=time_limit, 
                                   verbose=verbose, 
                                   regularization=reg, 
                                   similar_support=similar_support
                                   )

    def fit(self, X_train: pd.DataFrame, y_train):
        # Guess Thresholds
        X_train_guessed = self.enc.fit_transform(X_train, y_train)
        # No LB guess for now - want it to be the same model as the self.enc transform fitter

        # Train the split classifier
        self.clf.fit(X_train_guessed, y_train)

    def predict(self, X_test: pd.DataFrame):
        X_test_guessed = self.enc.transform(X_test)
        return self.clf.predict(X_test_guessed)
    
class ExactsplitWrapper: 
    def __init__(self, depth_budget=6, time_limit=60, verbose=True, reg=0.001):
        self.clf = splitClassifier(depth_budget=depth_budget, 
                                   time_limit=time_limit, 
                                   verbose=verbose, 
                                   regularization=reg, 
                                   similar_support=False, 
                                   allow_small_reg=True
                                   )

    def fit(self, X_train: pd.DataFrame, y_train):
        '''
        Requires X_train to be binary
        '''
        self.clf.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        return self.clf.predict(X_test)
    


if __name__ == "__main__":
    # read in parameters with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbdt_n_est", type=int, default=40)
    parser.add_argument("--gbdt_max_depth", type=int, default=1)
    parser.add_argument("-l", "--reg", type=float, default=0.001)
    parser.add_argument("-d", "--depth_budget", type=int, default=6)
    parser.add_argument("-t", "--time_limit", type=int, default=60)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dataset", type=str, default="compas")
    parser.add_argument("--no_guess", action="store_true")
    args = parser.parse_args()

    # Read the dataset
    df = pd.read_csv(f'datasets/{args.dataset}.csv')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)
    # Train the model
    start = time.time()
    if args.no_guess:
        model = ExactsplitWrapper(depth_budget=args.depth_budget, 
                                  time_limit=args.time_limit, 
                                  verbose=args.verbose, 
                                  reg=args.reg)
    else:
        model = splitWrapper(gbdt_n_est=args.gbdt_n_est, gbdt_max_depth=args.gbdt_max_depth, 
                         reg=args.reg, depth_budget=args.depth_budget, 
                         time_limit=args.time_limit, verbose=args.verbose)
    model.fit(X_train, y_train)
    print(f"Initialization/Training time: {time.time()-start}")

    # Evaluate the model
    pred_start_time = time.time()
    y_pred = model.predict(X_test)
    print(f"Test Prediction time: {time.time()-pred_start_time}")
    print("Train_acc: " + str(sum(model.predict(X_train) == y_train)/len(y_train)))
    print(f"Test accuracy: {sum(y_pred == y_test)/len(y_test)}")

    tree_as_dict = tree_to_dict(model.clf.trees_[0].tree, classes=model.clf.classes_)
    print(f'# leaves: {num_leaves(tree_as_dict)}')
    
    # # fit without guessing
    # def fit_exact(self, X_train, y_train)
