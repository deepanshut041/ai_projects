"""Project metadata for the local modelling and replay application."""

from . import __version__


def project_info() -> dict[str, str]:
    return {
        "project": "Bike Demand Planner",
        "version": __version__,
        "status": "saved calendar predictor, local recovery, historical replay and frozen final assessment",
        "target": "hourly rental count",
        "numerics": "linear prediction, half squared error, gradients and one SGD step",
        "data_preparation": "official audit, fixed calendar encoding and frozen chronological membership",
        "training": "training-only fits; explicit linear studies and library-backed tree estimators",
        "evaluation": "development comparisons and protocol-gated held-out assessment with persisted pre-label forecasts",
        "inference": "verified JSON export and fresh-process parity; local replay and benchmark views",
    }
