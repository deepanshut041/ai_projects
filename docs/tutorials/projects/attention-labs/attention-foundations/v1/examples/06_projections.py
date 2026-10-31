import json
import math
from attention_math import X, WQ, WK, WV, attention, matmul, rounded


def forward(theta):
    wq = [[theta, 0.0], [0.0, 1.0]]
    q, k, v = matmul(X, wq), matmul(X, WK), matmul(X, WV)
    _, weights, output = attention(q, k, v)
    error = output[-1][0] - 2.0
    return error * error / 2, q, k, v, weights, output


if __name__ == '__main__':
    loss, q, k, v, weights, output = forward(1.0)
    a = weights[-1]
    d_output = output[-1][0] - 2.0
    d_weights = [d_output * value[0] for value in v]
    weighted_gradient = sum(p * g for p, g in zip(a, d_weights))
    d_scores = [p * (g - weighted_gradient) for p, g in zip(a, d_weights)]
    analytic = sum(g * key[0] / math.sqrt(2) for g, key in zip(d_scores, k))
    epsilon = 1e-5
    numeric = (forward(1 + epsilon)[0] - forward(1 - epsilon)[0]) / (2 * epsilon)
    assert abs(analytic - numeric) < 1e-9
    new_theta = 1 - 0.2 * analytic
    assert forward(new_theta)[0] < loss
    print(json.dumps({'q': q, 'k': k, 'v': v, 'last_weights': rounded(a),
                      'last_output': rounded(output[-1]), 'loss': round(loss, 9),
                      'gradient': round(analytic, 9), 'gradient_verified': True,
                      'updated_parameter': round(new_theta, 9),
                      'updated_loss': round(forward(new_theta)[0], 9)}))
