"""Diagonal equivalence and a tiny scale search, not the full AWQ algorithm."""
import json


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def transformed(weights, sample, scales):
    return [w*s for w, s in zip(weights, scales)], [x/s for x, s in zip(sample, scales)]


def ternary(values):
    scale = max(map(abs, values))
    return [round(v / scale) * scale for v in values]


def loss(weights, samples, scales):
    errors = []
    for x in samples:
        ws, xs = transformed(weights, x, scales)
        assert abs(dot(weights, x) - dot(ws, xs)) < 1e-12
        errors.append((dot(ternary(ws), xs) - dot(weights, x))**2)
    return sum(errors) / len(errors)


def main():
    weights, samples = [0.24, 1.02], [[4.0, 1.0], [4.0, -1.0]]
    scores = {str(s): round(loss(weights, samples, [s, 1.0]), 6) for s in [1, 2, 4, 8]}
    assert scores == {"1": 0.9216, "2": 0.9216, "4": 0.0036, "8": 0.81}
    print(json.dumps({"scale_loss": scores, "best_first_channel_scale": 4,
                      "scope": "ternary teaching grid"}))


if __name__ == "__main__":
    main()
