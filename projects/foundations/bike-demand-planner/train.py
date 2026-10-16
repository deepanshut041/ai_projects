"""Entry point for fitting a demand estimator."""

from pathlib import Path

from bike_demand_planner.project import unavailable


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_directory = project_root / "data"
    run_directory = project_root / "runs" / "local"
    raise SystemExit(unavailable(
        "train", data_directory=data_directory, run_directory=run_directory,
    ))
