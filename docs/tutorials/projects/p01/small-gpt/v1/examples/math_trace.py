"""Run before setup: standard-library next-token, GELU and schedule arithmetic."""
import json
import math


def softmax(values):
    exps = [math.exp(x-max(values)) for x in values]
    return [value/sum(exps) for value in exps]


def main():
    probabilities = [0.5, 0.25, 0.125]
    nll = -sum(math.log(p) for p in probabilities)/len(probabilities)
    assert math.isclose(math.exp(nll), 4.0)
    # Two groups with unequal valid-token counts: (2*1 + 1*4)/3 = 2.
    assert (2*1.0 + 1*4.0)/3 == 2.0
    shifted = [257, 104, 105, 258]
    exact = -1 * 0.5*(1+math.erf(-1/math.sqrt(2)))
    approximate = -0.5*(1+math.tanh(math.sqrt(2/math.pi)*(-1-0.044715)))
    assert abs(exact-approximate) < 0.001
    p = softmax([0, math.log(3)])
    cold = softmax([0, math.log(3)/0.5])
    assert math.isclose(p[1], 0.75) and math.isclose(cold[1], 0.9)
    print(json.dumps({"x": shifted[:-1], "y": shifted[1:], "perplexity": round(math.exp(nll), 6),
                      "unequal_batch_nll": 2.0, "gelu_minus_one": round(approximate, 6),
                      "cold_probability": round(cold[1], 6)}))


if __name__ == "__main__":
    main()
