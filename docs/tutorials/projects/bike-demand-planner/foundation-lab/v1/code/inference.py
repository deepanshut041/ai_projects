"""Entry point for predictions from a saved estimator."""

from pathlib import Path

import json

from bike_demand_planner.artifacts import read_json
from bike_demand_planner.saved_predictor import SavedPredictor


def main(model_directory: Path, request_path: Path) -> int:
    try:
        predictor = SavedPredictor(model_directory)
        print(json.dumps(predictor.predict(read_json(request_path)), indent=2, allow_nan=False))
        return 0
    except FileNotFoundError:
        print(json.dumps({"status": "needs_run", "message": "Run application_example.py to prepare the saved predictor and example request."}))
        return 2
    except (ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "invalid_input", "message": str(error)}))
        return 2


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    model_directory = project_root / "models" / "selected"
    request_path = project_root / "data" / "request.json"
    raise SystemExit(main(model_directory, request_path))
