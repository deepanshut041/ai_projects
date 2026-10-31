"""Autoregressive mechanics with explicitly supplied logits; no LLM."""
import json
import math


def softmax(logits):
    maximum = max(logits)
    exps = [math.exp(x - maximum) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def decode(steps, eos, limit):
    ids, reason = [], "length"
    for logits in steps[:limit]:
        token = max(range(len(logits)), key=lambda i: logits[i])
        if token == eos:
            reason = "eos"
            break
        ids.append(token)
    return ids, reason


def main():
    probabilities = softmax([0.0, math.log(3.0)])
    shifted = softmax([1000.0, 1000.0 + math.log(3.0)])
    assert all(abs(a - b) < 1e-12 for a, b in zip(probabilities, shifted))
    # IDs 0='3', 1=' hours', 2=EOS. The model would produce each next vector.
    tokens, reason = decode([[4, 1, 0], [0, 4, 1], [0, 1, 4]], eos=2, limit=4)
    assert (tokens, reason) == ([0, 1], "eos")
    assert decode([[0, 4, 1]], eos=2, limit=1)[1] == "length"
    print(json.dumps({"probabilities": [round(x, 6) for x in probabilities],
                      "ids": tokens, "finish_reason": reason}))


if __name__ == "__main__":
    main()
