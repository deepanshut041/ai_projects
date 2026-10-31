"""Standard-library numerical trace. Predict values before running."""
import json
import math


def softmax(xs):
    peak = max(xs)
    terms = [math.exp(x - peak) for x in xs]
    return [x / sum(terms) for x in terms]


def attention(q, keys, values, allowed):
    scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q))
              for k in keys]
    selected = [s for s, ok in zip(scores, allowed) if ok]
    if not selected:
        raise ValueError("Every query needs at least one visible key")
    probs = iter(softmax(selected))
    weights = [next(probs) if ok else 0.0 for ok in allowed]
    return sum(w * v for w, v in zip(weights, values))


def main():
    p = softmax([1000.0, 1000.0])
    w, x, target = 2.0, 3.0, 1.0
    loss = lambda value: (value * x - target) ** 2
    eps = 1e-5
    analytic = 2 * (w * x - target) * x
    finite = (loss(w + eps) - loss(w - eps)) / (2 * eps)
    assert abs(analytic - finite) < 1e-7
    values = [2.0, 4.0, 8.0]
    causal = [attention([0.0], [[0.0]] * 3, values,
                        [j <= i for j in range(3)]) for i in range(3)]
    print(json.dumps({"stable_probabilities": p, "cross_entropy": round(-math.log(p[0]), 9),
                      "gradient": analytic, "updated_weight": w - 0.01 * analytic,
                      "causal": [round(v, 9) for v in causal],
                      "inputs": [257, 49, 50], "labels": [49, 50, 258]}))


if __name__ == "__main__":
    main()
