"""Entry point for fitting a demand estimator."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_planner.linear_evaluation import run_linear_regression


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root / "data/processed"
    run_directory = project_root / "runs" / ("linear-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    learning_rate, max_steps, gradient_tolerance = .1, 10_000, 1e-7
    try:
        record = run_linear_regression(processed_directory, run_directory, project_root,
                                       learning_rate=learning_rate, max_steps=max_steps,
                                       gradient_tolerance=gradient_tolerance)
    except (OSError, ValueError, FloatingPointError) as error:
        print(json.dumps({"status": "error", "message": str(error), "run_directory": str(run_directory)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps({"status": record["status"], "run_directory": str(run_directory),
                      "optimization": record["optimization"], "metrics": record["metrics"],
                      "test_evaluated": False}, indent=2))
