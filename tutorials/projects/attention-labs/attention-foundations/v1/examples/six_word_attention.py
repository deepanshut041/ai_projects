"""The six-word, three-coordinate walkthrough, without trainable projections."""
import math
from row_math import dot, matmul, softmax, transpose

TOKENS = ['Dream', 'big', 'and', 'work', 'for', 'it']
INPUTS = [[0.72, 0.45, 0.31], [0.75, 0.20, 0.55],
          [0.30, 0.80, 0.40], [0.85, 0.35, 0.60],
          [0.55, 0.15, 0.75], [0.25, 0.20, 0.85]]


def attend(inputs, scale=False):
    scores = matmul(inputs, transpose(inputs))
    divisor = math.sqrt(len(inputs[0])) if scale else 1.0
    weights = [softmax([x / divisor for x in row]) for row in scores]
    return scores, weights, matmul(weights, inputs)


if __name__ == '__main__':
    scores, weights, contexts = attend(INPUTS)
    for label, vector in zip(TOKENS, INPUTS):
        print(label, 'norm', round(math.sqrt(dot(vector, vector)), 6))
    for label, matrix in [('scores', scores), ('weights', weights), ('contexts', contexts)]:
        print(label)
        for row in matrix:
            print(' '.join(f'{x:.6f}' for x in row))
    assert all(abs(sum(row) - 1.0) < 1e-12 for row in weights)
    assert all(abs(scores[i][j] - scores[j][i]) < 1e-12
               for i in range(6) for j in range(6))
    assert abs(weights[0][1] - weights[1][0]) > 1e-4
    assert abs(contexts[1][0] - 0.5981968369218149) < 1e-7
    scaled = attend(INPUTS, scale=True)
    print('scaled big weights', [round(x, 6) for x in scaled[1][1]])
    print('scaled big context', [round(x, 6) for x in scaled[2][1]])
    print('divide by sum toy', [x / 12 for x in [1, 2, 3, 6]])
    print('softmax toy', [round(x, 6) for x in softmax([1, 2, 3, 6])])
