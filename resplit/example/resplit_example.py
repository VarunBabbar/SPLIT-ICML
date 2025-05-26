import argparse
import time
import pandas as pd
from treefarms import TREEFARMS
from resplit import RESPLIT

def load_dataset(path):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    print(f" Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
    return X, y

def run_resplit(X, y):
    print("\n Running RESPLIT...")
    config = {
        "regularization": 0.005,
        "rashomon_bound_multiplier": 0.01,
        "depth_budget": 5,
        "cart_lookahead_depth": 3,
        "verbose": False
    }
    model = RESPLIT(config, fill_tree='treefarms')
    start = time.perf_counter()
    model.fit(X, y)
    duration = time.perf_counter() - start
    print(f" RESPLIT completed in {duration:.2f} seconds with {len(model)} trees")
    return duration, len(model)

def run_treefarms(X, y):
    print("\n Running TREEFARMS...")
    config = {
        "regularization": 0.005,
        "rashomon_bound_multiplier": 0.01,
        "depth_budget": 5,
        "verbose": False
    }
    model = TREEFARMS(config)
    start = time.perf_counter()
    model.fit(X, y)
    duration = time.perf_counter() - start
    print(f" TREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")
    return duration, model.get_tree_count()

def main():
    parser = argparse.ArgumentParser(description="Compare RESPLIT and TREEFARMS on a dataset")
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default="compas.csv",
        help="Path to CSV dataset with binary label in last column"
    )
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    time_resplit, num_trees_resplit = run_resplit(X, y)
    time_treefarms, num_trees_treefarms = run_treefarms(X, y)

    print("\nSummary:")
    print(f"   RESPLIT    : {time_resplit:.2f} seconds, {num_trees_resplit} trees")
    print(f"   TREEFARMS  : {time_treefarms:.2f} seconds, {num_trees_treefarms} trees")


if __name__ == "__main__":
    main()
