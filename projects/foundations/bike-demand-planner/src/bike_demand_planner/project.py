"""Project metadata and availability of unfinished operations."""

import json
from pathlib import Path

from . import __version__


def project_info() -> dict[str, str]:
    return {
        "project": "Bike Demand Planner",
        "version": __version__,
        "status": "calendar baselines, linear regression and bounded tree/forest/boosting comparisons",
        "target": "hourly rental count",
        "numerics": "linear prediction, half squared error, gradients and one SGD step",
        "data_preparation": "official audit, fixed calendar encoding and frozen chronological membership",
        "training": "training-only fits; explicit linear studies and library-backed tree estimators",
        "evaluation": "aligned raw/nonnegative errors, seed-aggregated comparisons and validation-led selection",
        "inference": "not implemented",
    }


def unavailable(operation: str, **inputs: str | Path) -> int:
    """Report an unavailable operation without creating output files."""
    print(json.dumps({
        "operation": operation,
        "status": "not_implemented",
        "inputs": {key: str(value) for key, value in inputs.items()},
        "message": "This operation is not implemented. No output was created.",
    }, indent=2))
    return 2
