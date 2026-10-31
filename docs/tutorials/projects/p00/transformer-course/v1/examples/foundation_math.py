"""Small, inspectable arithmetic and text contracts for Transformer foundations."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

Vector = list[float]
Matrix = list[Vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Dot-product vectors must have equal, nonzero lengths")
    return sum(a * b for a, b in zip(left, right))


def matrix_shape(matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("A matrix must have at least one row and one column")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("All matrix rows must have the same length")
    return len(matrix), width


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    rows, columns = matrix_shape(matrix)
    return [[matrix[r][c] for r in range(rows)] for c in range(columns)]


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    rows, inner = matrix_shape(left)
    right_rows, columns = matrix_shape(right)
    if inner != right_rows:
        raise ValueError(f"Inner dimensions differ: {inner} versus {right_rows}")
    return [
        [sum(left[r][k] * right[k][c] for k in range(inner)) for c in range(columns)]
        for r in range(rows)
    ]


def linear(inputs: Matrix, weights: Matrix, bias: Vector) -> Matrix:
    result = matmul(inputs, weights)
    if len(bias) != len(result[0]):
        raise ValueError("One bias value is required per output feature")
    return [[value + bias[c] for c, value in enumerate(row)] for row in result]


def linear_backward(inputs: Matrix, weights: Matrix, grad_output: Matrix) -> tuple[Matrix, Matrix, Vector]:
    rows, input_width = matrix_shape(inputs)
    weight_rows, output_width = matrix_shape(weights)
    gradient_rows, gradient_width = matrix_shape(grad_output)
    if (weight_rows, gradient_rows, gradient_width) != (input_width, rows, output_width):
        raise ValueError("Input, weight and output-gradient dimensions do not agree")
    grad_inputs = matmul(grad_output, transpose(weights))
    grad_weights = matmul(transpose(inputs), grad_output)
    grad_bias = [sum(row[c] for row in grad_output) for c in range(output_width)]
    return grad_inputs, grad_weights, grad_bias


def stable_softmax(logits: Sequence[float]) -> Vector:
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("Softmax requires a nonempty sequence of finite logits")
    largest = max(logits)
    numerator = [math.exp(value - largest) for value in logits]
    denominator = sum(numerator)
    return [value / denominator for value in numerator]


def categorical_loss_and_gradient(logits: Sequence[float], target: int) -> tuple[float, Vector, Vector]:
    probabilities = stable_softmax(logits)
    if not 0 <= target < len(logits):
        raise ValueError("The target must index one of the logits")
    largest = max(logits)
    # Compute -log p_target without taking log of an underflowed probability.
    loss = largest + math.log(sum(math.exp(value - largest) for value in logits)) - logits[target]
    gradient = [probability - float(index == target) for index, probability in enumerate(probabilities)]
    return loss, probabilities, gradient


def scalar_mse_and_gradients(inputs: Vector, weights: Vector, bias: float, target: float) -> tuple[float, float, Vector, float]:
    prediction = dot(inputs, weights) + bias
    error = prediction - target
    loss = 0.5 * error * error
    grad_weights = [error * value for value in inputs]
    return prediction, loss, grad_weights, error


@dataclass(frozen=True)
class AdamState:
    step: int
    first_moment: Vector
    second_moment: Vector


def adam_step(
    parameters: Vector,
    gradients: Vector,
    state: AdamState | None = None,
    *,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[Vector, AdamState]:
    if not parameters or len(parameters) != len(gradients):
        raise ValueError("Parameters and gradients must have equal, nonzero lengths")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or learning_rate <= 0 or epsilon <= 0:
        raise ValueError("Adam requires beta values in [0,1), positive rate and positive epsilon")
    if not all(math.isfinite(value) for value in parameters + gradients):
        raise ValueError("Parameters and gradients must be finite")
    if state is None:
        state = AdamState(0, [0.0] * len(parameters), [0.0] * len(parameters))
    if state.step < 0 or len(state.first_moment) != len(parameters) or len(state.second_moment) != len(parameters):
        raise ValueError("Adam state must match the parameter vector")
    if not all(math.isfinite(value) for value in state.first_moment + state.second_moment):
        raise ValueError("Adam moments must be finite")
    if any(value < 0 for value in state.second_moment):
        raise ValueError("Adam second moments cannot be negative")
    step = state.step + 1
    first = [beta1 * previous + (1 - beta1) * gradient for previous, gradient in zip(state.first_moment, gradients)]
    second = [beta2 * previous + (1 - beta2) * gradient * gradient for previous, gradient in zip(state.second_moment, gradients)]
    corrected_first = [value / (1 - beta1**step) for value in first]
    corrected_second = [value / (1 - beta2**step) for value in second]
    updated = [
        parameter - learning_rate * moment / (math.sqrt(variance) + epsilon)
        for parameter, moment, variance in zip(parameters, corrected_first, corrected_second)
    ]
    return updated, AdamState(step, first, second)


def finite_difference(function, value: float, step: float = 1e-5) -> float:
    if step <= 0:
        raise ValueError("The difference step must be positive")
    return (function(value + step) - function(value - step)) / (2 * step)


def utf8_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def utf8_text(ids: Sequence[int]) -> str:
    return bytes(ids).decode("utf-8")


END_WORD = "</w>"


def merge_pair(symbols: Sequence[str], pair: tuple[str, str]) -> tuple[str, ...]:
    result: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            result.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            result.append(symbols[index])
            index += 1
    return tuple(result)


def fit_word_bpe(word_counts: Mapping[str, int], merge_count: int) -> list[tuple[str, str]]:
    if merge_count < 0 or not word_counts or any(count <= 0 for count in word_counts.values()):
        raise ValueError("BPE needs positive word counts and a nonnegative merge count")
    if any(END_WORD in word for word in word_counts):
        raise ValueError("The end-of-word marker is reserved")
    vocabulary = {tuple(word) + (END_WORD,): count for word, count in word_counts.items()}
    merges: list[tuple[str, str]] = []
    for _ in range(merge_count):
        counts: Counter[tuple[str, str]] = Counter()
        for symbols, frequency in vocabulary.items():
            for pair in zip(symbols, symbols[1:]):
                counts[pair] += frequency
        if not counts:
            break
        # Break equal-frequency ties consistently so the fixture is reproducible.
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        vocabulary = {merge_pair(symbols, best): count for symbols, count in vocabulary.items()}
        merges.append(best)
    return merges


def encode_word_bpe(word: str, merges: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    if END_WORD in word:
        raise ValueError("The end-of-word marker is reserved")
    symbols = tuple(word) + (END_WORD,)
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    while len(symbols) > 1:
        available = [pair for pair in zip(symbols, symbols[1:]) if pair in ranks]
        if not available:
            break
        symbols = merge_pair(symbols, min(available, key=ranks.__getitem__))
    return symbols


def decode_word_bpe(symbols: Sequence[str]) -> str:
    joined = "".join(symbols)
    if not joined.endswith(END_WORD) or joined.count(END_WORD) != 1:
        raise ValueError("One terminal end-of-word marker is required")
    return joined[: -len(END_WORD)]


def visibility_mask(length: int, allowed_keys: Sequence[bool], *, causal: bool = False) -> list[list[bool]]:
    if length <= 0 or not allowed_keys:
        raise ValueError("Query and key lengths must be positive")
    if causal and length != len(allowed_keys):
        raise ValueError("This causal fixture requires matching query and key lengths")
    return [[bool(allowed) and (not causal or key <= query) for key, allowed in enumerate(allowed_keys)] for query in range(length)]


def assert_split_disjoint(groups_by_split: Mapping[str, Sequence[str]]) -> None:
    owner: dict[str, str] = {}
    for split, groups in groups_by_split.items():
        for group in groups:
            previous = owner.setdefault(group, split)
            if previous != split:
                raise ValueError(f"Group {group!r} occurs in both {previous!r} and {split!r}")


def demo(section: str = "all") -> None:
    if section in {"all", "matrices"}:
        result = matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
        assert result == [[58, 64], [139, 154]]
        print("Matrix product:", result)
        inputs = [[1, 0], [0, 1], [1, 1]]
        weights = [[1, 0], [1, 1]]
        print("Linear map:", linear(inputs, weights, [0.5, -0.5]))
    if section in {"all", "learning"}:
        loss, probabilities, gradient = categorical_loss_and_gradient([0, math.log(3)], 1)
        print("Probabilities:", [round(value, 6) for value in probabilities])
        print("Token loss:", round(loss, 6), "logit gradient:", gradient)
        prediction, loss, grad_weights, grad_bias = scalar_mse_and_gradients([2, -1], [0.5, -1], 0.0, 1.0)
        updated, state = adam_step([0.5, -1, 0], grad_weights + [grad_bias])
        print("Scalar prediction/loss:", prediction, loss)
        print("Adam parameters:", [round(value, 6) for value in updated], "step:", state.step)
    if section in {"all", "text"}:
        text = "café"
        ids = utf8_ids(text)
        assert utf8_text(ids) == text
        print("UTF-8 bytes:", ids)
        merges = fit_word_bpe({"low": 5, "lower": 2, "lowest": 1}, 3)
        print("BPE merges:", merges)
        print("Encoded lower:", encode_word_bpe("lower", merges))
    if section in {"all", "contracts"}:
        print("Encoder visibility:", visibility_mask(3, [True, True, False]))
        print("Decoder visibility:", visibility_mask(3, [True, True, False], causal=True))
        assert_split_disjoint({"train": ["pair-01", "pair-02"], "validation": ["pair-03"]})
        print("Corpus groups: disjoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["all", "matrices", "learning", "text", "contracts"], default="all")
    demo(parser.parse_args().section)
