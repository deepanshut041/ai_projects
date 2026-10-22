"""Reproducible epoch batches, example-based schedules and bounded linear trials."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .optimizers import OptimizerConfig, OptimizerState, _update, initial_state
from .regression import _matrix, _objective_and_gradients, _vector


class BatchStream:
    """Shuffle without replacement each epoch and retain its exact next position."""

    def __init__(self, count: int, batch_size: int, seed: int):
        if any(type(value) is not int for value in (count, batch_size, seed)) or count < 1 or batch_size < 1 or seed < 0:
            raise ValueError("Use positive row/batch counts and a nonnegative integer seed.")
        self.count, self.batch_size = count, batch_size
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.order = self.rng.permutation(count)
        self.cursor, self.epoch, self.examples_seen = 0, 0, 0

    def next_indices(self, remaining: int | None = None) -> np.ndarray:
        if remaining is not None and (type(remaining) is not int or remaining < 1):
            raise ValueError("Remaining example budget must be positive.")
        if self.cursor == self.count:
            self.epoch += 1
            self.order = self.rng.permutation(self.count)
            self.cursor = 0
        size = min(self.batch_size, self.count - self.cursor, remaining or self.batch_size)
        indices = self.order[self.cursor:self.cursor + size].copy()
        self.cursor += size
        self.examples_seen += size
        return indices

    def to_dict(self) -> dict:
        return {"count": self.count, "batch_size": self.batch_size, "order": self.order.tolist(),
                "cursor": self.cursor, "epoch": self.epoch, "examples_seen": self.examples_seen,
                "rng": self.rng.bit_generator.state}

    @classmethod
    def from_dict(cls, values: dict):
        stream = cls(values["count"], values["batch_size"], 0)
        order = np.asarray(values["order"])
        if order.dtype.kind not in "iu" or order.shape != (stream.count,) or not np.array_equal(np.sort(order), np.arange(stream.count)):
            raise ValueError("Saved epoch order must be a permutation of the training row positions.")
        cursor, epoch, seen = (values[key] for key in ("cursor", "epoch", "examples_seen"))
        if any(type(v) is not int for v in (cursor, epoch, seen)) or not 0 <= cursor <= stream.count or epoch < 0 or seen != epoch * stream.count + cursor:
            raise ValueError("Saved epoch/cursor/example counts disagree.")
        stream.order, stream.cursor, stream.epoch, stream.examples_seen = order.copy(), cursor, epoch, seen
        stream.rng.bit_generator.state = values["rng"]
        return stream


@dataclass(frozen=True)
class TrialConfig:
    optimizer: OptimizerConfig = OptimizerConfig()
    learning_rate: float = .01
    batch_size: int = 32
    seed: int = 17
    max_examples: int = 86_450
    max_updates: int = 100_000
    max_seconds: float = 300
    decay_examples: int | None = None
    divergence_multiple: float = 100

    def __post_init__(self):
        for key in ("batch_size", "max_examples", "max_updates"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        for key in ("learning_rate", "max_seconds", "divergence_multiple"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        if self.divergence_multiple <= 1:
            raise ValueError("Divergence multiple must exceed one.")
        if self.decay_examples is not None and (type(self.decay_examples) is not int or self.decay_examples < 1):
            raise ValueError("Decay interval must be a positive integer example count.")


def scheduled_rate(config: TrialConfig, examples_seen: int) -> float:
    if type(examples_seen) is not int or examples_seen < 0:
        raise ValueError("Schedule progress must be a nonnegative integer example count.")
    return config.learning_rate / (1 + examples_seen / config.decay_examples) if config.decay_examples else config.learning_rate


@dataclass(frozen=True)
class TrialPoint:
    updates: int
    examples_processed: int
    epochs_completed: int
    half_mse: float
    gradient_norm: float
    next_learning_rate: float
    elapsed_seconds: float


@dataclass(frozen=True)
class OptimizationTrial:
    parameters: np.ndarray
    history: tuple[TrialPoint, ...]
    stop_reason: str
    elapsed_seconds: float
    state: dict
    first_batches: tuple[tuple[int, ...], ...]
    last_batch_size: int


def fit_trial(standardized_features, targets, config: TrialConfig, *, continuation: dict | None = None,
              pause_after_updates: int | None = None) -> OptimizationTrial:
    """Fit training data only; never use validation for updates or stopping."""
    Z = _matrix(standardized_features)
    y = _vector(targets, len(Z), "targets")
    if pause_after_updates is not None and (type(pause_after_updates) is not int or pause_after_updates < 1):
        raise ValueError("Pause position must be a positive integer update index.")
    stream = BatchStream(len(Z), config.batch_size, config.seed)
    theta = np.append(np.zeros(Z.shape[1], dtype=np.float64), y.mean())
    optimizer_state = initial_state(len(theta))
    prior_seconds = 0.0
    if continuation is not None:
        if continuation.get("resumable") is not True:
            raise ValueError("Failed trial state cannot be continued as a successful fit.")
        if continuation["version"] != "linear-optimization-state-v1" or continuation["configuration"] != asdict(config):
            raise ValueError("Continuation configuration differs from the saved state.")
        theta = _vector(continuation["parameters"], len(theta), "saved parameters").copy()
        stream = BatchStream.from_dict(continuation["batch_stream"])
        if stream.count != len(Z) or stream.batch_size != config.batch_size:
            raise ValueError("Continuation batch stream differs from the training design.")
        optimizer_state = OptimizerState.from_dict(continuation["optimizer"], len(theta))
        prior_seconds = continuation["elapsed_seconds"]
        if not np.isfinite(prior_seconds) or prior_seconds < 0:
            raise ValueError("Saved elapsed time must be finite and nonnegative.")
    started = perf_counter()
    history, first_batches = [], []
    last_batch_size = 0
    initial_loss = _objective_and_gradients(Z, y, np.zeros(Z.shape[1]), float(y.mean()))[0]
    threshold = config.divergence_multiple * max(initial_loss, 1.0)

    def observe():
        loss, dw, db = _objective_and_gradients(Z, y, theta[:-1], theta[-1])
        history.append(TrialPoint(optimizer_state.step, stream.examples_seen,
                                 stream.examples_seen // len(Z), loss, float(np.hypot(np.linalg.norm(dw), db)),
                                 scheduled_rate(config, stream.examples_seen), prior_seconds + perf_counter() - started))
        return loss

    observe()
    stop_reason = "examples_budget"
    while stream.examples_seen < config.max_examples:
        if pause_after_updates is not None and optimizer_state.step >= pause_after_updates:
            stop_reason = "paused"
            break
        if optimizer_state.step >= config.max_updates:
            stop_reason = "updates_budget"
            break
        if prior_seconds + perf_counter() - started >= config.max_seconds:
            stop_reason = "time_budget"
            break
        rate = scheduled_rate(config, stream.examples_seen)
        indices = stream.next_indices(config.max_examples - stream.examples_seen)
        last_batch_size = len(indices)
        if len(first_batches) < 2:
            first_batches.append(tuple(indices.tolist()))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                loss, dw, db = _objective_and_gradients(Z[indices], y[indices], theta[:-1], theta[-1])
                if loss > threshold:
                    stop_reason = "diverged"
                    break
                updated, next_state = _update(theta, np.append(dw, db), optimizer_state, config.optimizer, rate)
                if not np.isfinite(updated).all():
                    stop_reason = "nonfinite"
                    break
                theta, optimizer_state = updated, next_state
                if optimizer_state.step <= 2 or stream.cursor == stream.count:
                    if observe() > threshold:
                        stop_reason = "diverged"
                        break
        except (FloatingPointError, ValueError):
            stop_reason = "nonfinite"
            break
    if history[-1].updates != optimizer_state.step or history[-1].examples_processed != stream.examples_seen:
        try:
            observe()
        except (FloatingPointError, ValueError):
            stop_reason = "nonfinite"
    elapsed = prior_seconds + perf_counter() - started
    state = {"version": "linear-optimization-state-v1", "configuration": asdict(config),
             "resumable": stop_reason not in ("diverged", "nonfinite"),
             "parameters": theta.tolist(), "optimizer": optimizer_state.to_dict(),
             "batch_stream": stream.to_dict(), "elapsed_seconds": elapsed,
             "schedule": {"policy": "inverse_time_by_examples" if config.decay_examples else "constant",
                          "next_learning_rate": scheduled_rate(config, stream.examples_seen)},
             "recovery_scope": "Contains algorithm/order state; caller must also match saved source, data, feature order and scaler. Real fresh-process recovery remains separate."}
    return OptimizationTrial(theta.copy(), tuple(history), stop_reason, elapsed, state, tuple(first_batches), last_batch_size)
