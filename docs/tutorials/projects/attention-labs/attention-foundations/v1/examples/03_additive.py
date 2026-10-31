import json
import math
from row_math import X, matmul, rounded, softmax


def additive(query, states):
    # Project a scalar query to [query, -query]; project each key identically.
    # The output scoring vector is [1, -1]. No biases in this fixture.
    scores = [math.tanh(query + h[0]) - math.tanh(-query + h[1]) for h in states]
    weights = softmax(scores)
    context = matmul([weights], states)[0]
    return scores, weights, context


if __name__ == '__main__':
    scores, weights, context = additive(0.0, X)
    other = additive(1.0, X)
    assert abs(sum(weights) - 1.0) < 1e-14
    assert other[1] != weights
    assert softmax([1000.0, 1000.0]) == [0.5, 0.5]
    print(json.dumps({'scores': rounded(scores), 'weights': rounded(weights),
                      'context': rounded(context), 'new_query_weights': rounded(other[1])}))
