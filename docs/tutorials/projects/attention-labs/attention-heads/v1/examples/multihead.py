"""Two independent scalar heads, with all helpers defined here."""
import json
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Mismatched widths')
    return sum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[dot(row, column) for column in zip(*b)] for row in a]


def head(x, wq, wk, wv):
    q, k, v = matmul(x, wq), matmul(x, wk), matmul(x, wv)
    outputs, patterns = [], []
    for i, query in enumerate(q):
        scores = [dot(query, key) / math.sqrt(len(key)) for key in k[:i + 1]]
        maximum = max(scores)
        numerators = [math.exp(s - maximum) for s in scores]
        weights = [s / sum(numerators) for s in numerators]
        outputs.append(matmul([weights], v[:i + 1])[0])
        patterns.append(weights + [0.0] * (len(x) - i - 1))
    return outputs, patterns


if __name__ == '__main__':
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    h1, a1 = head(x, [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [1.0]])
    h2, a2 = head(x, [[0.0], [1.0]], [[0.0], [1.0]], [[1.0], [-1.0]])
    joined = [first + second for first, second in zip(h1, h2)]
    output = matmul(joined, [[1.0, 0.5], [-0.5, 1.0]])
    def rounded(matrix):
        return [[round(v, 6) for v in row] for row in matrix]
    assert a1[-1] != a2[-1]
    assert output[0] == [0.5, 1.5]
    print(json.dumps({'head1_weights': rounded(a1), 'head2_weights': rounded(a2),
                      'concatenated': rounded(joined), 'output': rounded(output)}))
