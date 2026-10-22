"""Linear prediction and a squared-error update for one observation."""

from collections.abc import Sequence
from math import isfinite


def _check_dimensions(first: Sequence[float], second: Sequence[float]) -> None:
    if len(first) == 0 or len(first) != len(second):
        raise ValueError("Expected matching, nonempty vectors.")


def _check_finite(*values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError("Values must be finite.")


def predict(features: Sequence[float], weights: Sequence[float], bias: float) -> float:
    """Return the weighted feature sum plus an intercept."""
    _check_dimensions(features, weights)
    _check_finite(*features, *weights, bias)
    return sum(x * w for x, w in zip(features, weights, strict=True)) + bias


def half_squared_error(prediction: float, target: float) -> float:
    """Return half the squared residual for one observation."""
    _check_finite(prediction, target)
    residual = prediction - target
    return 0.5 * residual**2


def squared_error_gradients(
    features: Sequence[float], prediction: float, target: float,
) -> tuple[list[float], float]:
    """Return the weight and bias gradients of half squared error."""
    if len(features) == 0:
        raise ValueError("Expected at least one feature.")
    _check_finite(*features, prediction, target)
    residual = prediction - target
    return [residual * x for x in features], residual


def sgd_step(
    weights: Sequence[float], bias: float, weight_gradients: Sequence[float],
    bias_gradient: float, learning_rate: float,
) -> tuple[list[float], float]:
    """Return new parameters without modifying the input weights."""
    _check_dimensions(weights, weight_gradients)
    _check_finite(*weights, bias, *weight_gradients, bias_gradient, learning_rate)
    if learning_rate < 0:
        raise ValueError("The learning rate must be nonnegative.")
    next_weights = [
        weight - learning_rate * gradient
        for weight, gradient in zip(weights, weight_gradients, strict=True)
    ]
    next_bias = bias - learning_rate * bias_gradient
    return next_weights, next_bias
