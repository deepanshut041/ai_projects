"""Exhaustive selection on authored layer costs, not measured model quality."""
import itertools
import json
import math


def storage(n, bits, group):
    return math.ceil(n * bits / 8) + math.ceil(n / group) * 2


def main():
    choices = [(4, 32), (4, 128), (8, 32)]
    losses = [{(4, 32): .08, (4, 128): .12, (8, 32): .01},
              {(4, 32): .30, (4, 128): .40, (8, 32): .02}]
    candidates = []
    for recipe in itertools.product(choices, repeat=2):
        size = sum(storage(n, *choice) for n, choice in zip([1024, 256], recipe))
        loss = sum(table[choice] for table, choice in zip(losses, recipe))
        candidates.append({"recipe": recipe, "bytes": size, "loss": loss})
    feasible = [c for c in candidates if c["bytes"] <= 900]
    best = min(feasible, key=lambda c: (c["loss"], c["bytes"]))
    assert best["recipe"] == ((4, 32), (8, 32)) and best["bytes"] == 848
    print(json.dumps({"recipe": best["recipe"], "bytes": best["bytes"],
                      "loss": round(best["loss"], 4), "budget": 900}))


if __name__ == "__main__":
    main()
