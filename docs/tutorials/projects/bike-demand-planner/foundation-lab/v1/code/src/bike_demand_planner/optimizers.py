"""Explicit SGD, classical momentum and bias-corrected Adam updates."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizerConfig:
    kind: str = "sgd"
    momentum: float = .9
    beta1: float = .9
    beta2: float = .999
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.kind not in ("sgd", "momentum", "adam"):
            raise ValueError("Choose sgd, momentum or adam.")
        if any(not np.isfinite(value) or not 0 <= value < 1 for value in (self.momentum, self.beta1, self.beta2)):
            raise ValueError("Momentum and Adam decay factors must be finite and in [0, 1).")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Adam epsilon must be finite and positive.")


@dataclass(frozen=True)
class OptimizerState:
    step: int
    first: np.ndarray
    second: np.ndarray

    def to_dict(self) -> dict:
        return {"step": self.step, "first": self.first.tolist(), "second": self.second.tolist()}

    @classmethod
    def from_dict(cls, values: dict, size: int):
        first = np.asarray(values["first"], dtype=np.float64)
        second = np.asarray(values["second"], dtype=np.float64)
        step = values["step"]
        if type(step) is not int or step < 0 or first.shape != (size,) or second.shape != (size,):
            raise ValueError("Invalid optimizer state dimensions or update index.")
        if not np.isfinite(first).all() or not np.isfinite(second).all() or np.any(second < 0):
            raise ValueError("Optimizer moments must be finite; second moments cannot be negative.")
        return cls(step, first.copy(), second.copy())


def initial_state(size: int) -> OptimizerState:
    if type(size) is not int or size < 1:
        raise ValueError("Parameter count must be positive.")
    return OptimizerState(0, np.zeros(size, dtype=np.float64), np.zeros(size, dtype=np.float64))


def _update(parameters, gradient, state, config, learning_rate):
    """Internal vector update; callers validate inputs and guard finite results."""
    step = state.step + 1
    first, second = state.first, state.second
    if config.kind == "sgd":
        direction = gradient
    elif config.kind == "momentum":
        first = config.momentum * first + gradient
        direction = first
    else:
        first = config.beta1 * first + (1 - config.beta1) * gradient
        second = config.beta2 * second + (1 - config.beta2) * gradient ** 2
        first_corrected = first / (1 - config.beta1 ** step)
        second_corrected = second / (1 - config.beta2 ** step)
        direction = first_corrected / (np.sqrt(second_corrected) + config.epsilon)
    return parameters - learning_rate * direction, OptimizerState(step, first, second)


def optimizer_step(parameters, gradient, state: OptimizerState, config: OptimizerConfig, learning_rate: float):
    """Return new parameters/state; epsilon is outside Adam's square root."""
    theta = np.asarray(parameters, dtype=np.float64)
    grad = np.asarray(gradient, dtype=np.float64)
    if theta.ndim != 1 or not len(theta) or grad.shape != theta.shape:
        raise ValueError("Parameters and gradient must have the same nonempty vector shape.")
    if not np.isfinite(theta).all() or not np.isfinite(grad).all():
        raise ValueError("Parameters and gradients must be finite.")
    if isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    checked_state = OptimizerState.from_dict(state.to_dict(), len(theta))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        updated, next_state = _update(theta, grad, checked_state, config, learning_rate)
    if not np.isfinite(updated).all():
        raise ValueError("Optimizer update became nonfinite.")
    return updated, next_state


def optimizer_definition(config: OptimizerConfig) -> dict:
    return {**asdict(config), "momentum_convention": "velocity = momentum * previous_velocity + gradient; no dampening",
            "adam_convention": "bias correction at t=1,2,...; epsilon outside sqrt(corrected second moment)",
            "weight_decay": 0, "nesterov": False, "amsgrad": False}
