"""Library-backed tree fitting with inspectable numerical prediction snapshots."""

import json
import warnings
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from .features import FEATURE_NAMES
from .objectives import calendar_representation


@dataclass(frozen=True)
class TreeSpec:
    family: str
    max_depth: int | None = 8
    min_samples_leaf: int = 10
    n_estimators: int = 1
    learning_rate: float = .1

    def __post_init__(self):
        if self.family not in ("tree", "forest", "boosting"):
            raise ValueError("Unknown tree family.")
        if self.max_depth is not None and (type(self.max_depth) is not int or not 1 <= self.max_depth <= 32):
            raise ValueError("Depth must be 1..32 or None.")
        if type(self.min_samples_leaf) is not int or self.min_samples_leaf < 1:
            raise ValueError("Use a positive integer leaf support.")
        cap = {"tree": 1, "forest": 64, "boosting": 300}[self.family]
        if type(self.n_estimators) is not int or not 1 <= self.n_estimators <= cap:
            raise ValueError("Tree count exceeds the declared family cap.")
        if isinstance(self.learning_rate, bool) or not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("Use a finite learning rate in (0, 1].")
        if self.family == "boosting" and (self.max_depth is None or self.max_depth > 3):
            raise ValueError("Boosting trees are bounded to depth three.")


def default_tree_candidates():
    """Three configurations per family, declared before looking at validation."""
    return {
        "tree": [TreeSpec("tree", 3, 20), TreeSpec("tree", 8, 10), TreeSpec("tree", None, 1)],
        "forest": [TreeSpec("forest", 8, 10, 64), TreeSpec("forest", 16, 5, 64),
                   TreeSpec("forest", None, 1, 64)],
        "boosting": [TreeSpec("boosting", 3, 10, count) for count in (100, 200, 300)],
    }


def _calendar(features):
    values, _ = calendar_representation(features)
    # sklearn's exact-split tree implementations convert inputs to float32.
    return np.asarray(values, dtype=np.float32)


def _tree_snapshot(estimator):
    tree = estimator.tree_
    return {"children_left": tree.children_left.tolist(), "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist(), "samples": tree.n_node_samples.tolist(),
            "weighted_samples": tree.weighted_n_node_samples.tolist(),
            "impurity": tree.impurity.tolist(), "max_depth": int(tree.max_depth),
            "node_count": int(tree.node_count), "leaf_count": int(estimator.get_n_leaves())}


def _validate_tree(tree):
    n = len(tree["value"])
    fields = ("children_left", "children_right", "feature", "threshold", "samples", "weighted_samples", "impurity")
    if n < 1 or any(len(tree[key]) != n for key in fields) or tree["node_count"] != n:
        raise ValueError("Inconsistent tree arrays.")
    for key in ("value", "threshold", "weighted_samples", "impurity"):
        if not np.isfinite(tree[key]).all():
            raise ValueError("Nonfinite tree state.")
    for node, (left, right, feature) in enumerate(zip(tree["children_left"], tree["children_right"], tree["feature"], strict=True)):
        if left == right == -1:
            continue
        if any(type(child) is not int or not node < child < n for child in (left, right)) or type(feature) is not int or feature not in range(len(FEATURE_NAMES)):
            raise ValueError("Invalid or cyclic tree structure.")


def _predict_tree(tree, X):
    _validate_tree(tree)
    left = np.asarray(tree["children_left"]); right = np.asarray(tree["children_right"])
    feature = np.asarray(tree["feature"]); threshold = np.asarray(tree["threshold"])
    nodes = np.zeros(len(X), dtype=np.int64)
    for _ in range(len(left)):
        active = np.flatnonzero(left[nodes] != -1)
        if not len(active):
            return np.asarray(tree["value"])[nodes]
        current = nodes[active]
        nodes[active] = np.where(X[active, feature[current]] <= threshold[current], left[current], right[current])
    raise ValueError("Tree traversal did not reach leaves.")


def predict_snapshot(snapshot, features):
    """Replay fitted numerical state; fitting and outcomes are absent from this path."""
    if snapshot["version"] != "calendar-tree-state-v1" or tuple(snapshot["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Unsupported tree state or feature order.")
    X = _calendar(features)
    trees = snapshot["trees"]
    if not trees:
        raise ValueError("A fitted snapshot needs at least one tree.")
    if snapshot["aggregation"] == "mean":
        return np.mean([_predict_tree(tree, X) for tree in trees], axis=0)
    if snapshot["aggregation"] == "sequential_sum":
        result = np.full(len(X), snapshot["initial_prediction"], dtype=np.float64)
        for tree in trees:
            result += snapshot["learning_rate"] * _predict_tree(tree, X)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite ensemble output.")
        return result
    raise ValueError("Unknown aggregation policy.")


def trace_snapshot(snapshot, feature_row, *, tree_index=0):
    """Return the actual tests and leaf reached in one retained component tree."""
    X = _calendar([feature_row]); tree = snapshot["trees"][tree_index]; _validate_tree(tree)
    node = 0; steps = []
    while tree["children_left"][node] != -1:
        index = tree["feature"][node]; value = float(X[0, index]); threshold = tree["threshold"][node]
        go_left = value <= threshold
        steps.append({"node": node, "feature": FEATURE_NAMES[index], "value": value,
                      "threshold": threshold, "direction": "left" if go_left else "right"})
        node = tree["children_left" if go_left else "children_right"][node]
    return {"steps": steps, "leaf": node, "leaf_value": tree["value"][node],
            "distinct_training_samples": tree["samples"][node],
            "weighted_training_samples": tree["weighted_samples"][node]}


def fit_tree_model(features, targets, spec: TreeSpec, *, seed=17):
    """Use sklearn to construct the trees; export all state needed for prediction."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Use an integer random seed in the uint32 range.")
    X = _calendar(features); y = np.asarray(targets, dtype=np.float64)
    if y.shape != (len(X),) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Expected aligned finite nonnegative rental targets.")
    common = dict(max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf,
                  max_leaf_nodes=1024, random_state=seed)
    if spec.family == "tree":
        estimator = DecisionTreeRegressor(criterion="squared_error", **common)
    elif spec.family == "forest":
        estimator = RandomForestRegressor(n_estimators=spec.n_estimators, criterion="squared_error",
                                         bootstrap=True, max_features=.7, n_jobs=1, **common)
    else:
        estimator = GradientBoostingRegressor(n_estimators=spec.n_estimators,
                    loss="squared_error", learning_rate=spec.learning_rate,
                    subsample=1.0, n_iter_no_change=None, **common)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    fit_seconds = perf_counter() - started
    trees = ([estimator] if spec.family == "tree" else list(estimator.estimators_) if spec.family == "forest"
             else list(estimator.estimators_[:, 0]))
    snapshot = {"version": "calendar-tree-state-v1", "library": "scikit-learn", "library_version": sklearn.__version__,
                "feature_names": list(FEATURE_NAMES), "input_dtype": "float32; exact binary calendar values",
                "target_transform": "identity", "family": spec.family, "specification": asdict(spec),
                "seed": seed, "estimator_parameters": estimator.get_params(deep=False),
                "aggregation": "sequential_sum" if spec.family == "boosting" else "mean",
                "initial_prediction": float(estimator.init_.constant_[0, 0]) if spec.family == "boosting" else 0.0,
                "learning_rate": spec.learning_rate if spec.family == "boosting" else None,
                "trees": [_tree_snapshot(tree) for tree in trees]}
    started = perf_counter(); reference = estimator.predict(X); library_predict_seconds = perf_counter() - started
    difference = float(np.max(np.abs(reference - predict_snapshot(snapshot, X))))
    if difference > 1e-9:
        raise ValueError("Numerical snapshot differs from its fitted library estimator.")
    receipt = {"status": "completed", "fit_seconds": fit_seconds,
               "library_training_prediction_seconds": library_predict_seconds,
               "fit_objective": "squared-error split/leaf means" if spec.family != "boosting" else "sequential squared-error residual trees",
               "stopping_rule": "structural caps and requested estimator count; no validation early stopping",
               "tree_count": len(trees), "total_nodes": sum(t["node_count"] for t in snapshot["trees"]),
               "total_leaves": sum(t["leaf_count"] for t in snapshot["trees"]),
               "maximum_depth": max(t["max_depth"] for t in snapshot["trees"]),
               "snapshot_json_bytes": len((json.dumps(snapshot, indent=2, allow_nan=False) + "\n").encode()),
               "training_replay_maximum_difference": difference,
               "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]}
    return snapshot, receipt
