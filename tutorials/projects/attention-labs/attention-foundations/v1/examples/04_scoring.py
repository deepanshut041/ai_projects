import json
import math
from row_math import X, dot, rounded, softmax


if __name__ == '__main__':
    query = [1.0, 1.0]
    scores = [dot(query, key) for key in X]
    ordinary = softmax(scores)
    scaled = softmax([score / math.sqrt(2) for score in scores])
    sharp = softmax([0.0, 20.0])[-1]
    mild = softmax([0.0, 2.0])[-1]
    assert dot([1.0, 0.0], [2.0, 0.0]) > dot([1.0, 0.0], [1.0, 0.0])
    assert abs(sum(scaled) - 1.0) < 1e-14
    print(json.dumps({'scores': scores, 'unscaled': rounded(ordinary),
                      'scaled': rounded(scaled),
                      'mild_diagonal_derivative': round(mild * (1 - mild), 9),
                      'sharp_diagonal_derivative': round(sharp * (1 - sharp), 12)}))
