import numpy as np
from tqdm import tqdm
from treefarms.model.tree_classifier import (
    TreeClassifier,
)  # Import the tree classification model


def make_internal_node(feature):
    node = {}
    node["feature"] = int(feature)
    node["relation"] = "=="
    node["reference"] = "true"
    return node


def make_leaf_node(val):
    node = {}
    node["prediction"] = val
    node["name"] = "Prediction"
    return node


def subset(X, capture_set, feature, prediction):
    feature = int(feature)
    capturing_points = X[:, feature]
    if not prediction:
        capturing_points = np.logical_not(capturing_points)
    return np.logical_and(capture_set, capturing_points)


def tree_to_hist(tree):
    out = []
    working = [tree]
    while working:
        layer = []
        new_working = []
        for i in working:
            if "prediction" in i:
                layer.append(-1 - i["prediction"])
            else:
                layer.append(i["feature"])
                new_working.append(i["true"])
                new_working.append(i["false"])
        out.append(" ".join(map(str, layer)))
        working = new_working

    return out


class ModelSetContainer:
    def __init__(self, model_set):
        self.regularization = model_set["metadata"]["regularization"]
        self.dataset_size = model_set["metadata"]["dataset_size"]
        self.available_metrics = model_set["available_metric_values"]
        self.storage = model_set["storage"]

        model_count = 0
        for i in self.available_metrics["metric_pointers"]:
            model_count += self.storage[i]["count"]
        self.model_count = model_count

    def __getitem__(self, key):
        return self.get_tree_at_idx(key)

    def get_tree_at_idx(self, idx):
        return TreeClassifier(self.get_tree_at_idx_raw(idx))

    def get_tree_count(self):
        return self.model_count

    def get_tree_metric_at_idx(self, idx):
        return self.transform_metric(self.get_tree_metric_at_idx_raw(idx))

    def to_trie(self):
        trie = {}
        for i in tqdm(
            range(self.get_tree_count()),
            desc=f"Generating trie from {self.get_tree_count()} trees",
        ):
            hist = tree_to_hist(self.get_tree_at_idx_raw(i))
            head = trie
            for node in hist[:-1]:
                if node not in head:
                    head[node] = {}
                head = head[node]
            leaf = hist[-1]
            assert leaf not in head
            head[leaf] = self.get_tree_metric_at_idx(i)
        return trie

    def get_model_set(self, idx):
        return self.storage[idx]

    def get_tree_at_idx_raw(self, idx):
        for entry in self.available_metrics["metric_pointers"]:
            count = self.get_model_set(entry)["count"]
            if idx < count:
                return self.get_tree_at_model_set_idx(entry, idx)
            idx -= count
        raise "Index exceeds total stored trees"

    def get_tree_at_model_set_idx(self, model_set_idx, idx):
        model_set = self.get_model_set(model_set_idx)

        assert idx < model_set["count"]
        if model_set["terminal"]:
            if idx == 0:
                return make_leaf_node(model_set["prediction"])
            idx -= 1
        for split_on in model_set["mapping"]:
            for pair in model_set["mapping"][split_on]:
                left_count = self.get_model_set(pair[0])["count"]
                right_count = self.get_model_set(pair[1])["count"]

                pair_count = left_count * right_count
                if idx < pair_count:
                    node = make_internal_node(split_on)
                    node["true"] = self.get_tree_at_model_set_idx(
                        pair[0], idx // right_count
                    )
                    node["false"] = self.get_tree_at_model_set_idx(
                        pair[1], idx % right_count
                    )
                    return node
                idx -= pair_count

    def get_tree_metric_at_idx_raw(self, idx):
        for entry, metric_value in zip(
            self.available_metrics["metric_pointers"],
            self.available_metrics["metric_values"],
        ):
            count = self.get_model_set(entry)["count"]
            if idx < count:
                return metric_value
            idx -= count
        raise "Index exceeds total stored trees"

    def transform_metric(self, metric):
        return {
            "objective": metric[0],
            "loss": metric[1] / self.dataset_size,
            "complexity": metric[2] * self.regularization,
        }

    # Unpolished Methods
    def evaluate(self, X, y):
        for i in self.available_metrics["metric_pointers"]:
            self.evaluate_at_idx(X, y, np.ones(y.shape), i)

    def evaluate_at_idx(self, X, y, capture_set, idx):
        model_set = self.storage[idx]
        if "prediction_metric" in model_set:
            return model_set["prediction_metric"]
        prediction_metric = {}
        model_set["prediction_metric"] = prediction_metric
        if model_set["terminal"]:
            prediction = model_set["prediction"]
            if prediction:
                y = np.logical_not(y)

            falses = np.logical_and(capture_set, y).sum()

            prediction_metric[(falses, 1)] = 1

        mapping = model_set["mapping"]
        for feature in mapping:
            left_capture_set = subset(X, capture_set, feature, True)
            right_capture_set = subset(X, capture_set, feature, False)
            for pair in mapping[feature]:
                left_metrics = self.evaluate_at_idx(X, y, left_capture_set, pair[0])
                right_metrics = self.evaluate_at_idx(X, y, right_capture_set, pair[1])
                for left_metric in left_metrics:
                    for right_metric in right_metrics:
                        new_metric = (
                            left_metric[0] + right_metric[0],
                            left_metric[1] + right_metric[1],
                        )
                        if new_metric not in prediction_metric:
                            prediction_metric[new_metric] = 0
                        prediction_metric[new_metric] += (
                            left_metrics[left_metric] * right_metrics[right_metric]
                        )

        return model_set["prediction_metric"]

    def has_hist(self, hist):
        candidate_pairs = [[]]
        for i in hist:
            splits = list(map(int, i.split(" ")))
            new_candidate_pairs = []
            for idx in range(0, len(splits), 2):

                if len(splits) > 1:
                    left_candidate_pairs = []
                    right_candidate_pairs = []
                    pair = (splits[idx], splits[idx + 1])
                    candidates = candidate_pairs[idx // 2]
                    for i in candidates:
                        items_left = self.hist_item_in_model_set(pair[0], i[0])
                        items_right = self.hist_item_in_model_set(pair[1], i[1])
                        if items_left and items_right:
                            left_candidate_pairs += items_left
                            right_candidate_pairs += items_right
                    if not left_candidate_pairs and not right_candidate_pairs:
                        return False
                    if pair[0] >= 0:
                        new_candidate_pairs.append(left_candidate_pairs)
                    if pair[1] >= 0:
                        new_candidate_pairs.append(right_candidate_pairs)

                else:
                    val = splits[0]
                    new_candidate_pairs = candidate_pairs
                    for i in self.available_metrics["metric_pointers"]:
                        items = self.hist_item_in_model_set(val, i)
                        if val < 0:
                            return items[0]
                        if items:
                            candidate_pairs[0] += items

            candidate_pairs = new_candidate_pairs

        return True

    def hist_item_in_model_set(self, hist_item, idx):
        if hist_item < 0:
            if self.storage[idx]["terminal"]:
                assert self.storage[idx]["prediction"] == -(hist_item + 1)
                return [True]
        else:
            hist_item = str(hist_item)
            if hist_item in self.storage[idx]["mapping"]:
                return self.storage[idx]["mapping"][hist_item]
        return False
