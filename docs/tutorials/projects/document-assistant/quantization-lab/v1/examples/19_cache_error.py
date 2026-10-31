"""Scalar attention with isolated key/value errors; not a cache kernel."""
import json
import math


def attend(query, keys, values):
    scores = [query*k for k in keys]  # head_dim=1, so sqrt(d)=1.
    maximum = max(scores)
    exps = [math.exp(s-maximum) for s in scores]
    probabilities = [e/sum(exps) for e in exps]
    return sum(p*v for p, v in zip(probabilities, values)), probabilities


def main():
    base, probabilities = attend(1, [0, math.log(3)], [0, 4])
    changed_key, _ = attend(1, [0, math.log(3) + math.log(3)], [0, 4])
    changed_value, same_probabilities = attend(1, [0, math.log(3)], [0, 5])
    assert probabilities == same_probabilities
    assert abs(base - 3) < 1e-12 and abs(changed_key - 3.6) < 1e-12
    assert abs(changed_value - 3.75) < 1e-12
    print(json.dumps({"base": base, "key_error_output": round(changed_key, 6),
                      "value_error_output": changed_value, "probabilities": probabilities}))


if __name__ == "__main__":
    main()
