"""Projected single-head retrieval, using the earlier row operations."""
import math
from row_math import X, dot, matmul, rounded, softmax


def attention(q, k, v, causal=False, key_valid=None, scaled=True):
    if len(k) != len(v) or not k or not q:
        raise ValueError('Need queries and matching nonempty keys/values')
    if causal and len(q) != len(k):
        raise ValueError('This toy causal path assumes aligned self-attention')
    key_valid = [True] * len(k) if key_valid is None else list(key_valid)
    if len(key_valid) != len(k):
        raise ValueError('Key validity must match the key axis')
    scale = math.sqrt(len(k[0])) if scaled else 1.0
    scores = [[dot(query, key) / scale for key in k] for query in q]
    weights = [softmax(row, [valid and (not causal or j <= i)
                            for j, valid in enumerate(key_valid)])
               for i, row in enumerate(scores)]
    return scores, weights, matmul(weights, v)


WQ = [[1.0, 0.0], [0.0, 1.0]]
WK = [[1.0, 1.0], [0.0, 1.0]]
WV = [[2.0, 0.0], [0.0, -1.0]]
