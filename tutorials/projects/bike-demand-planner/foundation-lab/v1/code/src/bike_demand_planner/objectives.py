"""Explicit rental-unit losses, calendar representations and bounded linear fits."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .features import FEATURE_NAMES
from .optimizers import OptimizerConfig, _update, initial_state
from .regression import FeatureStandardizer, _matrix, _scalar, _vector, fit_standardizer


@dataclass(frozen=True)
class LossConfig:
    kind: str = "half_mse"
    delta: float = 10.0
    l2: float = 0.0

    def __post_init__(self):
        if self.kind not in ("half_mse", "mae", "huber"):
            raise ValueError("Unknown loss.")
        if _scalar(self.delta, "Huber threshold") <= 0 or _scalar(self.l2, "L2 strength") < 0:
            raise ValueError("Use positive delta and nonnegative L2 strength.")


def loss_values_and_slopes(errors, config: LossConfig):
    """Return per-row loss and its derivative/subgradient with respect to prediction."""
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim != 1 or not len(e) or not np.isfinite(e).all():
        raise ValueError("Expected finite nonempty residuals.")
    with np.errstate(over="raise", invalid="raise"):
        if config.kind == "half_mse":
            values, slopes = .5 * e ** 2, e.copy()
        elif config.kind == "mae":
            values, slopes = np.abs(e), np.sign(e)  # zero is the chosen subgradient at equality
        else:
            absolute = np.abs(e)
            quadratic = np.minimum(absolute, config.delta)
            values = .5 * quadratic ** 2 + config.delta * (absolute - quadratic)
            slopes = np.clip(e, -config.delta, config.delta)
    return values, slopes


def objective_and_gradients(features, targets, weights, bias, config: LossConfig):
    """Mean data loss plus lambda/2 ||weights||²; the intercept is unpenalized."""
    X = _matrix(features)
    y = _vector(targets, len(X), "targets")
    w = _vector(weights, X.shape[1], "weights")
    b = _scalar(bias, "bias")
    values, slopes = loss_values_and_slopes(X @ w + b - y, config)
    with np.errstate(over="raise", invalid="raise"):
        penalty = .5 * config.l2 * float(w @ w)
        total = float(values.mean()) + penalty
        dw = X.T @ slopes / len(X) + config.l2 * w
        db = float(slopes.mean())
    if not np.isfinite(total) or not np.isfinite(dw).all() or not np.isfinite(db):
        raise ValueError("Objective or gradients became nonfinite.")
    return total, dw, db


def calendar_representation(features, kind="onehot45"):
    """Transform an already frozen calendar vector without using outcomes."""
    X = _matrix(features, len(FEATURE_NAMES))
    if not np.isin(X, [0, 1]).all() or any(not np.all(X[:, a:b].sum(axis=1) == 1)
                                         for a, b in ((0, 24), (24, 31), (31, 43))):
        raise ValueError("Expected the fixed binary calendar encoding.")
    if kind == "onehot45":
        return X.copy(), FEATURE_NAMES
    if kind == "hour_workingday69":
        names = FEATURE_NAMES + tuple(f"hr_{hour:02d}_workingday" for hour in range(24))
        return np.column_stack((X, X[:, :24] * X[:, 44:45])), names
    if kind != "cyclic8":
        raise ValueError("Unknown calendar representation.")
    values, names = [], []
    for label, start, stop, period in (("hr", 0, 24, 24), ("weekday", 24, 31, 7), ("month", 31, 43, 12)):
        angle = 2 * np.pi * X[:, start:stop].argmax(axis=1) / period
        values.extend((np.sin(angle), np.cos(angle)))
        names.extend((label + "_sin", label + "_cos"))
    return np.column_stack((*values, X[:, 43], X[:, 44])), tuple(names + ["holiday", "workingday"])


@dataclass(frozen=True)
class ObjectiveModel:
    weights: tuple[float, ...]
    bias: float
    standardizer: FeatureStandardizer
    representation: str
    feature_names: tuple[str, ...]

    def predict(self, features, policy="raw"):
        X, names = calendar_representation(features, self.representation)
        if names != self.feature_names:
            raise ValueError("Feature order differs from the saved model.")
        values = self.standardizer.transform(X) @ np.asarray(self.weights) + self.bias
        return apply_prediction_policy(values, policy)


def apply_prediction_policy(values, policy="raw"):
    predictions = np.asarray(values, dtype=np.float64)
    if predictions.ndim != 1 or not len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("Expected finite nonempty predictions.")
    if policy == "raw":
        return predictions.copy()
    if policy == "nonnegative":
        return np.maximum(predictions, 0)
    raise ValueError("Unknown prediction policy.")


def fit_objective_model(raw_features, targets, *, loss=LossConfig(), representation="onehot45",
                        scaling="standardized", solver="adam", rate=1.0, max_steps=2000,
                        max_seconds=60.0):
    """Fit deterministic full-batch calculations with explicit budgets and receipts."""
    X, names = calendar_representation(raw_features, representation)
    y = _vector(targets, len(X), "targets")
    if scaling not in ("raw", "standardized") or solver not in ("adam", "sgd", "least_squares"):
        raise ValueError("Unknown scaling or solver.")
    if type(max_steps) is not int or max_steps < 1 or _scalar(rate, "rate") <= 0 or _scalar(max_seconds, "time budget") <= 0:
        raise ValueError("Use positive update/rate/time budgets.")
    if solver == "least_squares" and loss.kind != "half_mse":
        raise ValueError("The augmented solve supports squared error only.")
    fitted = fit_standardizer(X)
    scaler = fitted if scaling == "standardized" else FeatureStandardizer(
        (0.0,) * X.shape[1], (1.0,) * X.shape[1], fitted.constant_columns, len(X))
    Z = scaler.transform(X)
    theta = np.append(np.zeros(Z.shape[1]), y.mean())
    history = []
    started = perf_counter()
    rank = None
    stop = "maximum_updates"
    updates = 0

    def observe(step):
        total, dw, db = objective_and_gradients(Z, y, theta[:-1], theta[-1], loss)
        penalty = .5 * loss.l2 * float(theta[:-1] @ theta[:-1])
        point = {"step": step, "objective": total, "data_loss": total - penalty,
                 "penalty": penalty, "gradient_norm": float(np.hypot(np.linalg.norm(dw), db)),
                 "elapsed_seconds": perf_counter() - started}
        history.append(point)
        return total

    initial = observe(0)
    status = "completed"
    try:
        if solver == "least_squares":
            A = np.column_stack((Z, np.ones(len(Z))))
            # Multiplying by sqrt(N*lambda) matches a MEAN data term plus lambda/2 ||w||².
            penalty_rows = np.zeros((Z.shape[1], Z.shape[1] + 1))
            penalty_rows[:, :-1] = np.eye(Z.shape[1]) * np.sqrt(len(Z) * loss.l2)
            if loss.l2:
                A = np.vstack((A, penalty_rows))
                rhs = np.concatenate((y, np.zeros(Z.shape[1])))
            else:
                rhs = y
            theta, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
            rank = int(rank)
            stop = "direct_training_solve"
            observe(0)
        else:
            config = OptimizerConfig(kind=solver)
            state = initial_state(len(theta))
            for step in range(1, max_steps + 1):
                if perf_counter() - started >= max_seconds:
                    stop = "time_budget"
                    break
                _, dw, db = objective_and_gradients(Z, y, theta[:-1], theta[-1], loss)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    updated, next_state = _update(theta, np.append(dw, db), state, config, rate)
                if not np.isfinite(updated).all():
                    raise ValueError("Parameter update became nonfinite.")
                theta, state = updated, next_state
                updates = step
                if step <= 2 or step % 100 == 0 or step == max_steps:
                    if observe(step) > 100 * max(initial, 1):
                        status, stop = "failed", "diverged"
                        break
            if history[-1]["step"] != updates:
                observe(updates)
        if not np.isfinite(theta).all():
            raise ValueError("Fitted parameters are nonfinite.")
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        status, stop = "failed", "nonfinite"
        failure = {"type": type(error).__name__, "message": str(error)}
    else:
        failure = None
    model = ObjectiveModel(tuple(theta[:-1].tolist()), float(theta[-1]), scaler, representation, names)
    receipt = {"status": status, "stop_reason": stop, "updates": updates,
               "processed_examples": updates * len(X), "converged": None,
               "dtype": "float64", "batch": "all training rows",
               "max_steps": max_steps, "max_seconds": max_seconds,
               "optimizer": asdict(OptimizerConfig(kind=solver)) if solver != "least_squares" else None,
               "training_count": len(X), "elapsed_seconds": perf_counter() - started,
               "rank": rank, "loss": asdict(loss), "solver": solver, "rate": rate,
               "scaling": scaling, "representation": representation,
               "initialization": "zero weights and training target mean", "target_transform": "identity",
               "history": history, "failure": failure,
               "direct_solver_work": "one augmented factorization" if solver == "least_squares" else None}
    return model, receipt
