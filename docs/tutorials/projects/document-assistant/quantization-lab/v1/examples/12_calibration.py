"""Calibration statistics in input-channel coordinates; no gradient training."""
import json


def gram(samples):
    if not samples or not samples[0]:
        raise ValueError("Need a rectangular nonempty sample matrix")
    n, d = len(samples), len(samples[0])
    if any(len(x) != d for x in samples):
        raise ValueError("Need a rectangular nonempty sample matrix")
    return [[sum(x[i] * x[j] for x in samples) / n for j in range(d)] for i in range(d)]


def quadratic(error, matrix):
    return sum(error[i] * matrix[i][j] * error[j]
               for i in range(len(error)) for j in range(len(error)))


def reconstruction(error, samples):
    return sum(sum(e*x for e, x in zip(error, sample))**2 for sample in samples) / len(samples)


def main():
    samples = [[4.0, 1.0], [4.0, -1.0]]
    matrix = gram(samples)
    a, b = [0.25, 0.0], [0.0, 0.25]
    assert matrix == [[16.0, 0.0], [0.0, 1.0]]
    assert quadratic(a, matrix) == reconstruction(a, samples) == 1.0
    assert quadratic(b, matrix) == reconstruction(b, samples) == 0.0625
    print(json.dumps({"gram": matrix, "same_weight_mse": True,
                      "output_mse": [quadratic(a, matrix), quadratic(b, matrix)]}))


if __name__ == "__main__":
    main()
