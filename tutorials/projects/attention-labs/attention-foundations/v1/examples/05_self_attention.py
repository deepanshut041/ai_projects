import json
from row_math import X, dot, matmul, rounded, softmax, transpose


def self_attention(inputs):
    scores = matmul(inputs, transpose(inputs))
    weights = [softmax(row) for row in scores]
    return scores, weights, matmul(weights, inputs)


if __name__ == '__main__':
    scores, weights, output = self_attention(X)
    permutation = [2, 0, 1]
    permuted = [X[i] for i in permutation]
    _, _, changed = self_attention(permuted)
    error = max(abs(changed[i][j] - output[old][j])
                for i, old in enumerate(permutation) for j in range(2))
    zero_weights = [softmax([dot([0.0, 0.0], key) for key in X])]
    assert error < 1e-14
    assert zero_weights[0] == [1 / 3] * 3
    assert weights[0][2] != weights[2][0]
    print(json.dumps({'scores': scores, 'weights': rounded(weights),
                      'output': rounded(output), 'permutation_equivariant': error < 1e-14,
                      'zero_query_weights': rounded(zero_weights[0])}))
