"""Rows are output channels; columns are input channels."""
import json
import math


def matvec(matrix, vector):
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("Input dimension mismatch")
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def main():
    weights = [[0.24, 1.02], [-0.76, 0.49]]
    restored = [[0.0, 1.0], [-1.0, 0.5]]
    x = [4.0, 1.0]
    error = [[q - w for q, w in zip(qrow, wrow)] for qrow, wrow in zip(restored, weights)]
    y, qy = matvec(weights, x), matvec(restored, x)
    delta = [a - b for a, b in zip(qy, y)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(delta, matvec(error, x)))
    error_norm = math.sqrt(sum(v*v for row in error for v in row))
    x_norm = math.sqrt(sum(v*v for v in x))
    delta_norm = math.sqrt(sum(v*v for v in delta))
    assert delta_norm <= error_norm * x_norm + 1e-12
    print(json.dumps({"float_output": [round(v, 4) for v in y],
                      "quantized_output": qy, "delta": [round(v, 4) for v in delta]}))


if __name__ == "__main__":
    main()
