"""Float64 linear regression with training-only scaling and full-batch gradients."""

from dataclasses import dataclass

import numpy as np


def _matrix(values, width: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape) or (width is not None and matrix.shape[1] != width):
        raise ValueError("Expected a nonempty [batch, features] matrix with the declared feature count.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature values must be finite.")
    return matrix


def _vector(values, count: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (count,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected finite {name} with shape ({count},), without broadcasting.")
    return vector


def _scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"Expected a finite scalar {name}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Expected a finite scalar {name}.")
    return number


def forward(features, weights, bias: float) -> np.ndarray:
    """Return [batch] predictions; an intercept broadcasts only after shape checks."""
    X = _matrix(features)
    w = _vector(weights, X.shape[1], "weights")
    b = _scalar(bias, "bias")
    with np.errstate(over="raise", invalid="raise"):
        predictions = X @ w + b
    if not np.isfinite(predictions).all():
        raise ValueError("Linear predictions became nonfinite.")
    return predictions


def _objective_and_gradients(X, y, w, b) -> tuple[float, np.ndarray, float]:
    with np.errstate(over="raise", invalid="raise"):
        errors = X @ w + b - y
        loss = float(np.dot(errors, errors) / (2 * len(y)))
        weight_gradients = X.T @ errors / len(y)
        bias_gradient = float(errors.mean())
    if not np.isfinite(loss) or not np.isfinite(weight_gradients).all() or not np.isfinite(bias_gradient):
        raise ValueError("The objective or gradients became nonfinite.")
    return loss, weight_gradients, bias_gradient


def loss_and_gradients(features, targets, weights, bias: float) -> tuple[float, np.ndarray, float]:
    """Mean half squared error and its analytic weight/intercept gradients."""
    X = _matrix(features)
    y = _vector(targets, X.shape[0], "targets")
    w = _vector(weights, X.shape[1], "weights")
    return _objective_and_gradients(X, y, w, _scalar(bias, "bias"))


def batch_step(features, targets, weights, bias: float, learning_rate: float) -> tuple[np.ndarray, float]:
    """Apply one full-batch update without modifying the supplied parameters."""
    rate = _scalar(learning_rate, "learning rate")
    if rate < 0:
        raise ValueError("The learning rate must be nonnegative.")
    _, weight_gradients, bias_gradient = loss_and_gradients(features, targets, weights, bias)
    return np.asarray(weights, dtype=np.float64) - rate * weight_gradients, float(bias) - rate * bias_gradient


@dataclass(frozen=True)
class FeatureStandardizer:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    constant_columns: tuple[int, ...]
    training_count: int

    def transform(self, features) -> np.ndarray:
        X = _matrix(features, len(self.means))
        with np.errstate(over="raise", invalid="raise"):
            result = (X - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(result).all():
            raise ValueError("Standardized features became nonfinite.")
        return result


def fit_standardizer(training_features) -> FeatureStandardizer:
    """Fit column means/population deviations on training; constants use scale one."""
    X = _matrix(training_features)
    with np.errstate(over="raise", invalid="raise"):
        means = X.mean(axis=0)
        scales = X.std(axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("Training feature statistics became nonfinite.")
    constants = np.flatnonzero(scales == 0)
    scales[constants] = 1.0
    return FeatureStandardizer(tuple(means.tolist()), tuple(scales.tolist()), tuple(constants.tolist()), len(X))


@dataclass(frozen=True)
class LinearRegressionModel:
    weights: tuple[float, ...]
    bias: float
    standardizer: FeatureStandardizer

    def predict(self, features) -> np.ndarray:
        """Transform with saved training statistics, then predict without outcomes."""
        return forward(self.standardizer.transform(features), self.weights, self.bias)

    def raw_parameters(self) -> tuple[tuple[float, ...], float]:
        """Express the same fitted function in the original feature coordinates."""
        weights = np.asarray(self.weights) / np.asarray(self.standardizer.scales)
        bias = self.bias - float(weights @ np.asarray(self.standardizer.means))
        return tuple(weights.tolist()), bias


@dataclass(frozen=True)
class OptimizationPoint:
    step: int
    half_mse: float
    gradient_norm: float


@dataclass(frozen=True)
class LinearFit:
    model: LinearRegressionModel
    history: tuple[OptimizationPoint, ...]
    converged: bool
    stop_reason: str
    maximum_curvature: float


def fit_linear_regression(
    training_features, training_targets, *, learning_rate: float = 0.1,
    max_steps: int = 10_000, gradient_tolerance: float = 1e-7,
) -> LinearFit:
    """Fit using training-only gradients, a fixed rate and a declared stopping rule."""
    X = _matrix(training_features)
    y = _vector(training_targets, len(X), "targets")
    rate = _scalar(learning_rate, "learning rate")
    tolerance = _scalar(gradient_tolerance, "gradient tolerance")
    if rate <= 0 or tolerance <= 0 or type(max_steps) is not int or max_steps < 1:
        raise ValueError("Use a positive rate/tolerance and a positive integer update limit.")
    standardizer = fit_standardizer(X)
    Z = standardizer.transform(X)
    design = np.column_stack((Z, np.ones(len(Z))))
    maximum_curvature = float(np.linalg.eigvalsh(design.T @ design / len(Z))[-1])
    if rate * maximum_curvature >= 2:
        raise ValueError(f"Learning rate must be below {2 / maximum_curvature:.6g} for this training design.")
    weights = np.zeros(Z.shape[1], dtype=np.float64)
    bias = float(y.mean())
    history = []
    converged = False
    for step in range(max_steps + 1):
        loss, weight_gradients, bias_gradient = _objective_and_gradients(Z, y, weights, bias)
        gradient_norm = float(np.hypot(np.linalg.norm(weight_gradients), bias_gradient))
        history.append(OptimizationPoint(step, loss, gradient_norm))
        if gradient_norm <= tolerance:
            converged = True
            break
        if step == max_steps:
            break
        weights = weights - rate * weight_gradients
        bias = bias - rate * bias_gradient
    model = LinearRegressionModel(tuple(weights.tolist()), bias, standardizer)
    return LinearFit(model, tuple(history), converged,
                     "training_gradient_tolerance" if converged else "maximum_updates", maximum_curvature)


def least_squares_reference(training_features, training_targets, standardizer: FeatureStandardizer) -> tuple[LinearRegressionModel, int]:
    """Solve the same training problem stably, including rank-deficient designs."""
    Z = standardizer.transform(training_features)
    y = _vector(training_targets, len(Z), "targets")
    design = np.column_stack((Z, np.ones(len(Z))))
    parameters, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    return LinearRegressionModel(tuple(parameters[:-1].tolist()), float(parameters[-1]), standardizer), int(rank)
