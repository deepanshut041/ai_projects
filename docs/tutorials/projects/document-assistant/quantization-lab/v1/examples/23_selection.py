"""Development-only choice and exact paired bootstrap on a small fixture."""
import itertools
import json


def choose(candidates, memory_cap, minimum_support):
    feasible = [c for c in candidates if c["peak_gib"] <= memory_cap
                and c["support_rate"] >= minimum_support and c["reload_verified"]]
    return min(feasible, key=lambda c: (c["latency_s"], c["name"])) if feasible else None


def bootstrap_means(differences):
    # Exhaustive empirical bootstrap for n=4 only; 4**4 resamples.
    return sorted(sum(sample) / len(sample)
                  for sample in itertools.product(differences, repeat=len(differences)))


def main():
    candidates = [{"name": "q8", "peak_gib": 9, "support_rate": .95, "reload_verified": True, "latency_s": 4},
                  {"name": "q4", "peak_gib": 6, "support_rate": .90, "reload_verified": True, "latency_s": 3}]
    best = choose(candidates, memory_cap=8, minimum_support=.92)
    assert best is None  # Do not relax the threshold after seeing results.
    differences = [0, -1, 0, 1]  # Candidate-supported minus reference-supported, paired IDs.
    distribution = bootstrap_means(differences)
    assert sum(differences) == 0 and len(distribution) == 256
    print(json.dumps({"selected": best, "paired_mean": 0.0,
                      "bootstrap_range": [distribution[0], distribution[-1]], "resamples": 256}))


if __name__ == "__main__":
    main()
