### Preparing LicketySplitClassifier for scikit-learn integration

#### Scope
Goal: ship a binary decision tree classifier with global/near-optimal search and lookahead that integrates with scikit‑learn’s tree API and tooling.

### Implementation steps

- [x] Create external prototype `LicketySplitClassifier(BaseEstimator, ClassifierMixin)` with parameters: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state`, `class_weight`, `ccp_alpha`, `lookahead`, `depth_budget`, `regularization_scale`, `binarize`, `binarization_strategy`.
- [ ] Implement `fit(X, y, sample_weight=None)` using current `split/src/split/LicketySPLIT.py` logic behind a pure function `_train_core(X, y, sample_weight, rng)`; return `self`.
- [ ] Implement `predict(X)` and `predict_proba(X)`; set `classes_`, `n_features_in_`, and ensure deterministic output with `random_state`.
- [ ] Add parameter validation via `sklearn.utils._param_validation`-style checks or equivalent in prototype.
- [ ] Add `get_params`/`set_params` support implicitly by inheriting from `BaseEstimator`; avoid logic in `__init__` beyond assignments.

- [ ] Refactor `split/src/split/LicketySPLIT.py` into a reusable core module (no prints, no argparse, no globals); expose helpers to build a binary tree from trained submodels.
- [ ] Ensure optional binarization path using `ThresholdGuessBinarizer` with `set_output(transform="pandas")`; accept numpy and pandas inputs; store `feature_names_in_`.
- [ ] Add `sample_weight` plumbed to solver where applicable (no-ops allowed initially, but accept argument and test shape).
- [ ] Map internal `Node/Leaf` to a neutral structure returned by `_train_core` for easy conversion later.

- [ ] Implement `check_estimator` smoke test locally on the prototype and fix failures (shape checks, dtypes, error messages).
- [ ] Add unit tests for: binary/multiclass fit-predict, `random_state` reproducibility, `sample_weight`, `lookahead>=2`, `max_depth` edge cases, `binarize=True/False`.

- [ ] Design mapping from internal tree to `sklearn.tree._tree.Tree` arrays (children, feature, threshold, value); document array shapes needed.
- [ ] Implement builder that writes `Tree` arrays from the learned structure; verify `predict_proba` via `self.tree_.predict` matches prototype outputs.
- [ ] Refactor class to `ClassifierMixin, BaseDecisionTree` once `Tree` population works; set `_parameter_constraints` and `__sklearn_tags__`.

- [ ] Add experimental gate module `sklearn/experimental/enable_lickety_split.py` and import target class from `sklearn.tree` namespace for tests.
- [ ] Write minimal docs: NumPy-style docstring on class; small example usage; reference to experimental enable import.
- [ ] Add example script `examples/tree/plot_lickety_split.py` demonstrating lookahead vs baseline tree on a small dataset.

- [ ] Set up CI task to run unit tests and estimator checks for the prototype package.
- [ ] Add a basic benchmark script comparing `DecisionTreeClassifier` vs `LicketySplitClassifier` on 2 small datasets.

- [ ] Prepare upstream-ready PR patchset outline: file locations, tests path, docs stubs, and experimental exposure steps.

#### Acceptance constraints
- No new runtime dependencies outside scikit‑learn’s stack. If you currently rely on GOSDT, reimplement the solver logic or vendor a minimal dependency-free core.
- Public API must follow estimator guidelines and pass common checks.
- Start as experimental; promote once stable.

#### Target module and base class
- Target package: `sklearn/tree/`
- Base class target for upstream PR: subclass `ClassifierMixin` and `BaseDecisionTree`, populate `self.tree_` with a `sklearn.tree._tree.Tree` instance.
- Reference locations:
```13:101:sklearn/tree/_classes.py
class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
```

```707:716:sklearn/tree/_classes.py
class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
```

#### External-repo prototype (development) API
- Implement as a normal estimator first to iterate faster:
  - Class signature: `class LicketySplitClassifier(BaseEstimator, ClassifierMixin)`
  - Methods: `__init__`, `fit`, `predict`, `predict_proba`, `get_params`, `set_params`
  - Parameters: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state`, `class_weight`, `ccp_alpha`, plus:
    - `lookahead` (int ≥ 2)
    - `depth_budget` (int or None)
    - `regularization_scale` (float > 0)
    - `binarize` (bool)
    - `binarization_strategy` (str)
- Ensure determinism via `random_state`. Accept `sample_weight`.

Skeleton (no comments):
```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state
import numpy as np

class LicketySplitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None,
                 class_weight=None, ccp_alpha=0.0, lookahead=2, depth_budget=None,
                 regularization_scale=1.0, binarize=False, binarization_strategy="gb"):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.lookahead = lookahead
        self.depth_budget = depth_budget
        self.regularization_scale = regularization_scale
        self.binarize = binarize
        self.binarization_strategy = binarization_strategy

    def fit(self, X, y, sample_weight=None):
        X = check_array(X, accept_sparse=False, dtype=np.float64, force_all_finite=True)
        y = np.asarray(y)
        check_classification_targets(y)
        rng = check_random_state(self.random_state)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]
        self._feature_names_in = None
        self._fitted_thresholds_ = None
        self._model_ = self._train_core(X, y_encoded, sample_weight, rng)
        return self

    def _train_core(self, X, y, sample_weight, rng):
        return {"tree": None}

    def predict_proba(self, X):
        check_is_fitted(self, "_model_")
        X = check_array(X, accept_sparse=False, dtype=np.float64, force_all_finite=True)
        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)
```

#### Upstream-ready integration (scikit-learn tree API)
- Refactor to subclass `BaseDecisionTree` so exporters, pruning, and tags work and behavior matches built-in trees.

Target class file:
- Either extend `sklearn/tree/_classes.py` or create `sklearn/tree/_lickety_split.py` and import in `sklearn/tree/__init__.py` later.

Skeleton (no comments):
```python
import copy
import numbers
import numpy as np
from sklearn.base import ClassifierMixin, _fit_context
from sklearn.tree._classes import BaseDecisionTree
from sklearn.tree._tree import Tree
from sklearn.utils._param_validation import Hidden, StrOptions, Interval, Real, RealNotInt
from sklearn.tree._criterion import Criterion
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import metadata_routing

class LicketySplitClassifier(ClassifierMixin, BaseDecisionTree):
    __metadata_request__predict_proba = {"check_input": metadata_routing.UNUSED}
    __metadata_request__fit = {"check_input": metadata_routing.UNUSED}

    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "criterion": [StrOptions({"gini", "entropy", "log_loss"}), Hidden(Criterion)],
        "class_weight": [dict, list, StrOptions({"balanced"}), None],
        "lookahead": [Interval(numbers.Integral, 2, None, closed="left")],
        "depth_budget": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "regularization_scale": [Interval(Real, 0.0, None, closed="left")],
        "binarize": [bool],
        "binarization_strategy": [StrOptions({"gb", "none"})],
    }

    def __init__(self, *, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0,
                 monotonic_cst=None, lookahead=2, depth_budget=None,
                 regularization_scale=1.0, binarize=False, binarization_strategy="gb"):
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes, random_state=random_state,
                         min_impurity_decrease=min_impurity_decrease, class_weight=class_weight,
                         ccp_alpha=ccp_alpha, monotonic_cst=monotonic_cst)
        self.lookahead = lookahead
        self.depth_budget = depth_budget
        self.regularization_scale = regularization_scale
        self.binarize = binarize
        self.binarization_strategy = binarization_strategy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, check_input=True):
        X, y = validate_data(self, X, y, validate_separately=(dict(dtype=np.float32, accept_sparse="csc"),
                                                              dict(ensure_2d=False, dtype=None)))
        self._fit_core_and_build_tree_(X, y, sample_weight)
        return self

    def _fit_core_and_build_tree_(self, X, y, sample_weight):
        n_features = X.shape[1]
        classes = np.unique(y)
        n_classes = np.array([classes.shape[0]], dtype=np.intp)
        self.tree_ = Tree(n_features, n_classes, 1)
        self.n_features_in_ = n_features
        self.classes_ = classes
        self.n_classes_ = classes.shape[0]
        self._populate_tree_structure_(X, y, sample_weight)

    def _populate_tree_structure_(self, X, y, sample_weight):
        pass

    def predict_proba(self, X, check_input=True):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float32, accept_sparse="csr", reset=False)
        proba = self.tree_.predict(X)
        return proba[:, : self.n_classes_]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        return tags
```

Key implementation notes for the upstream class:
- Implement your lookahead and global search in `_populate_tree_structure_` (and any helpers), then write node arrays into `self.tree_`. Align with how `DecisionTreeClassifier` shapes `self.tree_` so exporters and pruning work.
- If you cannot map to `Tree`, do not inherit `BaseDecisionTree`. Instead, ensure you still pass common checks as a standard estimator; exporters and tree utils won’t apply.

#### Experimental gating
- Add an enable module to gate public import while experimental:
```python
# sklearn/experimental/enable_lickety_split.py
from sklearn import tree
from sklearn.tree._lickety_split import LicketySplitClassifier
setattr(tree, "LicketySplitClassifier", LicketySplitClassifier)
tree.__all__ += ["LicketySplitClassifier"]
```

- Import usage:
  - `from sklearn.experimental import enable_lickety_split  # noqa`
  - `from sklearn.tree import LicketySplitClassifier`

#### Public exposure (post-experimental)
- In `sklearn/tree/__init__.py`, export the estimator by adding to imports and `__all__`.

#### Parameter validation and tags
- Use `_parameter_constraints` to validate all params.
- Provide `__sklearn_tags__` to declare capabilities: sparse support, missing value allowance, multi_label, etc. Mirror patterns in `DecisionTreeClassifier`.

#### Handling binarization and regularization
- For `binarize=True`, implement threshold guessing internally using only scikit‑learn components available in‑tree.
- Ensure `regularization_scale` adjusts penalties deterministically based on subset sizes.

#### Tests
- Unit tests under `sklearn/tree/tests/test_lickety_split.py`:
  - Fit/predict binary and multilabel where applicable.
  - `sample_weight` support.
  - Reproducibility with `random_state`.
  - Edge cases: single-class training error, min_samples constraints, depth budgets, lookahead≥2 enforced.

- Estimator checks locally in your external repo to prepare:
```python
import pytest
import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks
from your_pkg import LicketySplitClassifier

@parametrize_with_checks([LicketySplitClassifier()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
```

- Experimental import tests mirror:
```python
def test_enable_imports():
    code = """
from sklearn.experimental import enable_lickety_split  # noqa
from sklearn.tree import LicketySplitClassifier
"""
    assert code is not None
```

#### Documentation and examples
- API docstring following scikit‑learn style: parameters, attributes, notes, examples.
- User Guide section in tree docs describing lookahead and optimality.
- Example in `examples/tree/plot_lickety_split.py`.

#### Build/CI and perf
- No meson changes if only Python changes and reuse existing Cython `Tree` class. If you add Cython, update `sklearn/tree/meson.build`.
- Add a small benchmark in `asv_benchmarks` if performance claims are central.

#### Submission checklist
- Estimator passes `parametrize_with_checks`.
- Full `pytest` suite green.
- API docs render and example runs.
- Experimental gate in place.
- No new runtime dependencies.

- Summary:
  - Prototype as a standard estimator; then upstream as `ClassifierMixin, BaseDecisionTree` populating `_tree.Tree`.
  - Gate under `sklearn/experimental` initially, export later in `sklearn/tree/__init__.py`.
  - Implement params: `lookahead`, `depth_budget`, `regularization_scale`, `binarize`, `binarization_strategy`, with constraints and tags.
  - Provide tests, docs, and examples aligned with scikit‑learn conventions.