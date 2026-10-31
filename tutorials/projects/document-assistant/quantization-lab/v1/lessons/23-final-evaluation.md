## 23. Freeze the choice before opening final results

Development work has many degrees of freedom: format, calibration set, scale search, grouping, context length and application thresholds. Repeatedly choosing whichever looks best on the same final questions turns that “final” set into development data. We separate selection from assessment and allow a truthful outcome of no feasible candidate.

### Make feasibility operational

Suppose the declared rule is peak memory at most 8 GiB, supported-answer rate at least 0.92 on development and verified fresh-process reload. Among feasible candidates, choose minimum median end-to-end latency under the matched workload. Thresholds are illustrative design choices; the real protocol must select them before observing candidate results and include uncertainty/coverage requirements.

Our authored table has Q8 at 9 GiB, support 0.95, latency 4 seconds and successful reload; Q4 at 6 GiB, support 0.90, latency 3 seconds and successful reload. Q8 fails memory; Q4 fails support. The correct selection is `None`. Relaxing the support threshold to 0.90 after seeing this table would create a revised design decision, not evidence that the original rule passed.

### Paired differences need uncertainty and failure review

For supported-answer indicators, paired differences $d_i\in\{-1,0,1\}$ compare the same question. With `[0,-1,0,1]`, mean difference is zero, but one regression and one improvement remain. Four questions cannot establish preserved quality with useful precision. Keep the individual cases and document-level grouping.

An empirical paired bootstrap samples the paired units with replacement and recalculates the mean. Our four-unit example enumerates all $4^4=256$ ordered resamples instead of using randomness. Its possible means range from -1 to +1. This full range is **not** a 95% confidence interval. A percentile interval would choose declared quantiles; its interpretation depends on assumptions and tiny samples are unstable. Questions sharing a document are correlated, so for the real study resample document groups or use another declared method that respects dependence. Do not pretend that 100 questions from one manual give 100 independent documents.

### Complete decision and resampling reference

Create `23_selection.py`. `choose()` applies feasibility first, then a deterministic tie rule. `bootstrap_means()` is exhaustive only for the tiny teaching case; its exponential cost makes it inappropriate for the real dataset. A scalable implementation uses a frozen random seed and a bounded number of group-level resamples, with the full method recorded.

[Download the complete reference](examples/23_selection.py)

```python
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
```

From the scratch directory containing that file, run:

```bash
python3 23_selection.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "selected": null,
  "paired_mean": 0.0,
  "bootstrap_range": [
    -1.0,
    1.0
  ],
  "resamples": 256
}
```

The output should show no selection and 256 resamples. If a candidate is selected, inspect whether the filter uses “or” where all constraints are required. If zero average difference is interpreted as proof of equality, inspect the uncertainty and regression cases. An equivalence/noninferiority claim needs a predeclared acceptable margin and a suitable interval/test, not merely a nonsignificant difference.

### Seal the final assessment

After choosing on development, freeze artifact, tokenizer/template, runtime/placement/cache policy, context/output limits, prompt/packing policy, parsing/scoring rubric, thresholds and report code. Record their digests before accessing final labels. Run the selected candidate and declared reference on all frozen final IDs, retain failed/partial records, then score without tuning. If the final result reveals a serious issue, report it. A subsequent repair uses a new development iteration and needs fresh final evidence or an honest reused-test limitation.

The final report includes raw counts/rates, denominators, paired changes, document/domain/length slices, uncertainty method, artifact bytes, host/device memory, latency/throughput boundaries, preparation cost, failures and reproducibility/recovery evidence. Distinguish unsupported combinations and missing data from poor numerical scores. A full benchmark name belongs only to a full run under its required protocol; a declared subset is named as a subset.

[[practice:selection-transfer]]

### A result can be useful without a speedup

A compressed artifact may save memory while losing support quality, improve context headroom without improving decode, or fail to reload on the target runtime. Those are practical findings if measured under a frozen protocol. The project succeeds as learning only when the mechanisms and evidence limits can be explained; it succeeds as an application only when its declared contract passes. Next we assemble the all-phase handoff without conflating those states.
