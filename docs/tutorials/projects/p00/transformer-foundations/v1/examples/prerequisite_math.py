"""Inspect arrays, probability arithmetic, derivatives and parameter updates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math


def linear(rows, weights, bias):
    if not rows or not rows[0] or not weights or not weights[0]:
        raise ValueError("Nonempty rectangular matrices required")
    inputs, outputs = len(rows[0]), len(weights[0])
    if (len(weights) != inputs or len(bias) != outputs
            or any(len(row) != inputs for row in rows)
            or any(len(row) != outputs for row in weights)):
        raise ValueError("Expected X[N,D], W[D,F], b[F]")
    result = []
    for row in rows:
        output_row = []
        for f in range(outputs):
            total = bias[f]
            for d in range(inputs):
                total += row[d] * weights[d][f]
            output_row.append(total)
        result.append(output_row)
    return result


def shape(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Nonempty rectangular matrix required")
    return len(matrix), len(matrix[0])


def transpose(matrix):
    rows, columns = shape(matrix)
    return [[matrix[r][c] for r in range(rows)] for c in range(columns)]


def matmul(left, right):
    _, columns = shape(right)
    return linear(left, right, [0] * columns)


def mean(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("A nonempty finite sequence is required")
    return sum(values) / len(values)


def population_variance(values):
    center = mean(values)
    return mean([(value - center) ** 2 for value in values])


def norm(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("A nonempty finite sequence is required")
    return math.sqrt(sum(value * value for value in values))


def standardize(values, epsilon=1e-5):
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    center, variance = mean(values), population_variance(values)
    return [(value - center) / math.sqrt(variance + epsilon) for value in values]


def softmax(logits):
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("A nonempty finite sequence of logits is required")
    largest = max(logits)
    terms = [math.exp(value - largest) for value in logits]
    denominator = sum(terms)
    return [value / denominator for value in terms]


def nll(logits, target):
    softmax(logits)
    if not isinstance(target, int) or not 0 <= target < len(logits):
        raise ValueError("Target must be an integer class index")
    largest = max(logits)
    return math.log(sum(math.exp(value - largest) for value in logits)) - (logits[target] - largest)


def scalar_derivatives(weight, x, target, bias=0.0):
    prediction = weight * x + bias
    error = prediction - target
    return {"prediction": prediction, "loss": error * error,
            "grad_weight": 2 * error * x, "grad_bias": 2 * error}


def central_difference(function, value, epsilon=1e-5):
    if epsilon <= 0:
        raise ValueError("Difference step must be positive")
    return (function(value + epsilon) - function(value - epsilon)) / (2 * epsilon)


def sgd(parameters, gradients, learning_rate):
    if len(parameters) != len(gradients) or not parameters or learning_rate <= 0:
        raise ValueError("Matching nonempty vectors and a positive learning rate required")
    return [parameter - learning_rate * gradient for parameter, gradient in zip(parameters, gradients)]


def affine_derivatives(rows, weights, bias, targets):
    predictions = linear(rows, weights, bias)
    batch, outputs = shape(predictions)
    if shape(targets) != (batch, outputs):
        raise ValueError("Targets must match every predicted output")
    errors = [[value - targets[r][c] for c, value in enumerate(row)] for r, row in enumerate(predictions)]
    denominator = batch * outputs
    loss = sum(error * error for row in errors for error in row) / denominator
    grad_outputs = [[2 * error / denominator for error in row] for row in errors]
    return {"predictions": predictions, "loss": loss,
            "grad_inputs": matmul(grad_outputs, transpose(weights)),
            "grad_weights": matmul(transpose(rows), grad_outputs),
            "grad_bias": [sum(row[c] for row in grad_outputs) for c in range(outputs)]}


def logit_gradient(logits, target):
    nll(logits, target)
    return [value - float(index == target) for index, value in enumerate(softmax(logits))]


@dataclass(frozen=True)
class AdamState:
    step: int
    first: list[float]
    second: list[float]


def adam(parameters, gradients, state=None, *, learning_rate=0.01,
         beta1=0.9, beta2=0.999, epsilon=1e-8):
    if not parameters or len(parameters) != len(gradients):
        raise ValueError("Matching nonempty parameter/gradient vectors required")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or min(learning_rate, epsilon) <= 0:
        raise ValueError("Invalid Adam learning rate, epsilon or beta")
    state = state or AdamState(0, [0.0] * len(parameters), [0.0] * len(parameters))
    if state.step < 0 or len(state.first) != len(parameters) or len(state.second) != len(parameters):
        raise ValueError("Adam state shape does not match parameters")
    step = state.step + 1
    first = [beta1 * old + (1 - beta1) * gradient for old, gradient in zip(state.first, gradients)]
    second = [beta2 * old + (1 - beta2) * gradient * gradient for old, gradient in zip(state.second, gradients)]
    changed = [parameter - learning_rate * (m / (1 - beta1 ** step)) /
               (math.sqrt(v / (1 - beta2 ** step)) + epsilon)
               for parameter, m, v in zip(parameters, first, second)]
    return changed, AdamState(step, first, second)


def report():
    scalar = scalar_derivatives(2.0, 3.0, 1.0)
    updated = sgd([2.0, 0.0], [scalar["grad_weight"], scalar["grad_bias"]], 0.01)
    affine = affine_derivatives([[1, 2], [3, 4]], [[1, 0], [0, 1]], [0, 0], [[0, 0], [0, 0]])
    first, state = adam([2.0], [30.0])
    second, state = adam(first, [0.0], state)
    numerical = central_difference(lambda weight: (3 * weight - 1) ** 2, 2.0)
    return {"linear": linear([[1, 2], [3, 4]], [[1, 0, 2], [0, 1, 3]], [0, 0, 1]),
            "rectangular": matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),
            "mean": mean([1.0, 3.0]), "variance": population_variance([1.0, 3.0]),
            "norm": norm([3.0, 4.0]), "probabilities": [round(p, 9) for p in softmax([0, math.log(3)])],
            "nll": round(nll([0, math.log(3)], 1), 9),
            "scalar_gradient": [scalar["grad_weight"], scalar["grad_bias"]],
            "sgd_parameters": updated, "sgd_loss": scalar_derivatives(updated[0], 3, 1, updated[1])["loss"],
            "affine_gradient_weights": affine["grad_weights"], "affine_gradient_bias": affine["grad_bias"],
            "adam_parameters": [round(first[0], 9), round(second[0], 9)],
            "adam_moments_after_two": [round(state.first[0], 9), round(state.second[0], 9)],
            "finite_difference_matches": math.isclose(numerical, 30.0, abs_tol=1e-7)}


def torch_report():
    import torch
    from torch import nn
    torch.set_num_threads(1)
    dtype = torch.float64
    weight = torch.tensor(2.0, dtype=dtype, requires_grad=True)
    loss = (3 * weight - 1).square()
    loss.backward()
    gradient = weight.grad.item()
    with torch.no_grad():
        weight -= 0.01 * weight.grad
    inputs = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype)
    mathematical_w = torch.tensor([[1., 0., 2.], [0., 1., 3.]], dtype=dtype)
    layer = nn.Linear(2, 3).to(dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(mathematical_w.T)
        layer.bias.copy_(torch.tensor([0., 0., 1.], dtype=dtype))
    orientation = torch.equal(layer(inputs), torch.tensor([[1., 2., 9.], [3., 4., 19.]], dtype=dtype))
    return {"gradient": gradient, "weight_after_sgd": round(weight.item(), 9),
            "loss_after_sgd": round((3 * weight - 1).square().item(), 9),
            "linear_orientation_matches": orientation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true", help="Compare scalar/linear arithmetic in an available CPU Torch runtime")
    print(json.dumps(torch_report() if parser.parse_args().torch else report(), sort_keys=True))
