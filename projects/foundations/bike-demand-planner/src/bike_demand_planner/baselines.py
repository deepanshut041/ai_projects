"""Simple predictors fitted only from training outcomes."""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MeanBaseline:
    value: float
    training_count: int

    def predict(self, count: int) -> list[float]:
        """Repeat the fitted statistic without accepting outcomes."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Prediction count must be a nonnegative integer.")
        return [self.value] * count


def fit_mean_baseline(training_targets: Sequence[float]) -> MeanBaseline:
    """The training mean minimizes squared error among constant predictors."""
    if not training_targets or any(not math.isfinite(value) or value < 0 for value in training_targets):
        raise ValueError("Expected nonempty, finite, nonnegative training targets.")
    return MeanBaseline(math.fsum(training_targets) / len(training_targets), len(training_targets))


def _validate_group_key(key: tuple[int, int]) -> None:
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError("Expected an (hour, workingday) pair.")
    hour, workingday = key
    if type(hour) is not int or hour not in range(24) or type(workingday) is not int or workingday not in (0, 1):
        raise ValueError("Expected hour 0..23 and workingday 0 or 1.")


@dataclass(frozen=True)
class CalendarGroupMean:
    hour: int
    workingday: int
    value: float
    training_count: int


@dataclass(frozen=True)
class HourWorkingDayBaseline:
    groups: tuple[CalendarGroupMean, ...]
    fallback: MeanBaseline

    def predict(self, keys: Sequence[tuple[int, int]]) -> list[float]:
        """Look up fitted groups; valid absent groups use the global training mean."""
        values = {(group.hour, group.workingday): group.value for group in self.groups}
        predictions = []
        for key in keys:
            _validate_group_key(key)
            predictions.append(values.get(key, self.fallback.value))
        return predictions


def fit_hour_workingday_baseline(
    training_keys: Sequence[tuple[int, int]], training_targets: Sequence[float]
) -> HourWorkingDayBaseline:
    """Minimize training squared error separately in each observed calendar group."""
    if len(training_keys) != len(training_targets):
        raise ValueError("Each training target needs one calendar group.")
    fallback = fit_mean_baseline(training_targets)
    grouped_targets: dict[tuple[int, int], list[float]] = {}
    for key, target in zip(training_keys, training_targets, strict=True):
        _validate_group_key(key)
        grouped_targets.setdefault(key, []).append(target)
    groups = tuple(
        CalendarGroupMean(hour, workingday, math.fsum(targets) / len(targets), len(targets))
        for (hour, workingday), targets in sorted(grouped_targets.items())
    )
    return HourWorkingDayBaseline(groups, fallback)
