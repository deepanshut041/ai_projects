"""Trace one tiny real-data tree, then compare bounded calendar estimators."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from bike_demand_planner.evaluation import _write_json, load_labelled_partition
from bike_demand_planner.tree_evaluation import run_tree_study


def leaf_probe(processed_directory: Path):
    rows, _ = load_labelled_partition(processed_directory, "train")
    wanted = {72, 77, 95, 100}; chosen = [r for r in rows if r.identity.record_id in wanted]
    if len(chosen) != 4:
        raise ValueError("The real four-row teaching probe is unavailable.")
    X = np.asarray([[r.features[8]] for r in chosen]); y = np.asarray([r.target for r in chosen])
    stump = DecisionTreeRegressor(max_depth=1, random_state=17).fit(X, y)
    boosting = GradientBoostingRegressor(n_estimators=1, max_depth=1, learning_rate=.1,
                  loss="squared_error", random_state=17).fit(X, y)
    return {"scope": "four real training rows; one declared hr_08 input; teaching probe, not a forecast benchmark",
            "records": [{"record_id": r.identity.record_id, "time": r.identity.target_time, "target": r.target} for r in chosen],
            "initial_mean": float(y.mean()), "root_threshold": float(stump.tree_.threshold[0]),
            "requested_hr_08_values": [0, 1], "leaf_predictions": stump.predict([[0], [1]]).tolist(),
            "constant_sse": float(np.sum((y.mean()-y)**2)), "stump_sse": float(np.sum((stump.predict(X)-y)**2)),
            "boosting_first_tree_leaf_corrections": boosting.estimators_[0,0].predict([[0], [1]]).tolist(),
            "boosting_after_one_stage": boosting.predict([[0], [1]]).tolist()}


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root/"data"/"processed"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = project_root/"runs"/("trees-"+stamp)
    probe = leaf_probe(processed_directory)
    print(json.dumps({"leaf_probe": probe}, indent=2))
    report = run_tree_study(processed_directory, run_directory, project_root)
    _write_json(run_directory/"leaf-probe.json", probe)
    print(json.dumps({"run_directory": str(run_directory), "status": report["status"],
                      "selected": {k: report["selection"][k] for k in ("configuration", "policy", "mean_validation_mae", "representative_trial")}}, indent=2))
