"""Entry point for predictions from a saved estimator."""

from pathlib import Path

from bike_demand_planner.project import unavailable


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    model_directory = project_root / "models" / "selected"
    request_path = project_root / "data" / "request.json"
    raise SystemExit(unavailable(
        "inference", model_directory=model_directory, request_path=request_path,
    ))
