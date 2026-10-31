"""Row-vector operations and the shared synthetic fixture; standard library only."""
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Dot product widths must agree')
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError('Expected a nonempty rectangular matrix')
    return [list(column) for column in zip(*matrix)]


def matmul(a, b):
    columns = transpose(b)
    return [[dot(row, column) for column in columns] for row in a]


def softmax(scores, allowed=None):
    if not scores:
        raise ValueError('Need at least one score')
    allowed = [True] * len(scores) if allowed is None else list(allowed)
    if len(allowed) != len(scores) or not any(allowed):
        raise ValueError('Every query needs at least one allowed key')
    if any(not math.isfinite(score) for score, use in zip(scores, allowed) if use):
        raise ValueError('Allowed scores must be finite')
    maximum = max(score for score, use in zip(scores, allowed) if use)
    numerators = [math.exp(score - maximum) if use else 0.0
                  for score, use in zip(scores, allowed)]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]




def rounded(value):
    return [rounded(x) for x in value] if isinstance(value, list) else round(value, 6)


X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
