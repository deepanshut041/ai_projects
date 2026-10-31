"""Inspect a weighted mixture before introducing learned projections."""

import json


def weighted_mix(weights, values):
    if not values or len(weights) != len(values):
        raise ValueError("one weight is required for each nonempty value row")
    width = len(values[0])
    if not width or any(len(row) != width for row in values):
        raise ValueError("value rows must have one common nonzero width")
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1) > 1e-9:
        raise ValueError("attention weights must be nonnegative and sum to one")
    return [sum(weight * row[j] for weight, row in zip(weights, values))
            for j in range(width)]


def main():
    values = [[2., 0.], [0., 3.], [4., 5.]]
    context = weighted_mix([.25, .25, .5], values)
    single = weighted_mix([1., 0., 0.], values)
    assert context == [2.5, 3.25]
    assert single == values[0]
    print(json.dumps({"context": context, "single": single}))


if __name__ == "__main__":
    main()
