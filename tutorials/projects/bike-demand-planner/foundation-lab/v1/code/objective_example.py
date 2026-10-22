"""Trace a recorded hourly residual, then compare controlled linear-model choices."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_planner.objective_evaluation import run_objective_study
from bike_demand_planner.objectives import LossConfig, loss_values_and_slopes


def loss_probe(prediction=143.79444765760556, target=154.0):
    rows = {}
    for label, config in (("half_mse", LossConfig()), ("mae", LossConfig("mae")), ("huber10", LossConfig("huber", 10))):
        values, slopes = loss_values_and_slopes([prediction-target], config)
        rows[label] = {"loss": float(values[0]), "slope": float(slopes[0])}
    return {"training_record": 54, "target_time": "2011-01-03T08:00:00", "target": target,
            "observed_training_target": 154.0,
            "kind": "recorded training target" if target == 154 else "illustrative changed-target counterfactual",
            "prediction": prediction, "residual": prediction-target, "losses": rows}


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root/"data/processed"
    run_directory = project_root/"runs"/("objectives-"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    print(json.dumps(loss_probe(), indent=2), flush=True)
    report = run_objective_study(processed_directory, run_directory, project_root)
    (run_directory/"loss-probe.json").write_text(json.dumps(loss_probe(), indent=2)+"\n")
    print(json.dumps({"run_directory": str(run_directory), "selected_rates": report["selected_rates"],
                      "selection": report["selection"]}, indent=2))
