"""One constrained quadratic step; no claim to implement production GPTQ."""
import json


def cost(e1, e2, h):
    return h[0][0]*e1*e1 + 2*h[0][1]*e1*e2 + h[1][1]*e2*e2


def main():
    h, e1 = [[4.0, 2.0], [2.0, 2.0]], 0.25
    e2 = -h[1][0] * e1 / h[1][1]
    # Independent exhaustive scalar search includes the analytic minimum.
    candidates = [i / 1000 for i in range(-1000, 1001)]
    best = min(candidates, key=lambda e: cost(e1, e, h))
    assert abs(best - e2) < 1e-12 and e2 == -0.25
    epsilon = 1e-6
    slope = (cost(e1, e2 + epsilon, h) - cost(e1, e2 - epsilon, h)) / (2*epsilon)
    assert abs(slope) < 1e-9
    assert cost(e1, e2, h) < cost(e1, 0.0, h)
    print(json.dumps({"fixed_error": e1, "compensation": e2,
                      "before": cost(e1, 0, h), "after": cost(e1, e2, h)}))


if __name__ == "__main__":
    main()
