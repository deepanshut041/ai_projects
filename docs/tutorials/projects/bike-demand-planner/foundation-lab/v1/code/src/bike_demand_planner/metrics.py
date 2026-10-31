"""Error summaries for aligned hourly rental predictions."""

import math
from collections.abc import Sequence


def regression_metrics(targets: Sequence[float], predictions: Sequence[float]) -> dict:
    """Use prediction minus target; underprediction averages over all rows."""
    if not targets or len(targets) != len(predictions):
        raise ValueError("Expected nonempty, equally sized targets and predictions.")
    if any(not math.isfinite(value) for value in (*targets, *predictions)):
        raise ValueError("Targets and predictions must be finite.")
    errors = [prediction - target for target, prediction in zip(targets, predictions, strict=True)]
    deficits = [max(-error, 0.0) for error in errors]
    count = len(errors)
    under_count = sum(error < 0 for error in errors)
    deficit_sum = math.fsum(deficits)
    result = {
        "count": count,
        "mae": math.fsum(abs(error) for error in errors) / count,
        "rmse": math.sqrt(math.fsum(error * error for error in errors) / count),
        "mean_signed_error": math.fsum(errors) / count,
        "mean_underprediction": deficit_sum / count,
        "underprediction_count": under_count,
        "mean_underprediction_when_under": deficit_sum / under_count if under_count else None,
    }
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise ValueError("Error summaries exceed finite numeric range.")
    return result
