## 15. Spend precision where the declared budget permits

We have several mechanisms, but a practical assistant needs one feasible artifact. Smaller groups cost more metadata; higher precision costs more payload; preserving a sensitive layer may be worth more than preserving an insensitive one. Selection is a constrained decision, not simply “choose the lowest MSE.”

### Work the storage tradeoff before considering loss

For $n$ values, payload bits $b$, group size $g$ and two-byte group scales, our toy count is

$$B(n,b,g)=\lceil nb/8\rceil+2\lceil n/g\rceil.$$

For a 1024-value layer, four-bit/group-32 storage is $512+64=576$ bytes; group-128 storage is $512+16=528$. For a 256-value layer those numbers are $128+16=144$ and $128+4=132$. Eight-bit/group-32 storage is 1088 and 272 bytes respectively. Real presets may need different metadata; this count belongs to the stated toy layout.

With a 900-byte cap, four-bit/group-32 for the first layer plus eight-bit/group-32 for the second costs $576+272=848$ and fits. Eight bits for both costs 1360 and fails. Upgrading the larger first layer while keeping the second four-bit already costs at least 1220, so it also fails.

### Define an objective before looking for a winner

The authored loss table gives first-layer losses 0.08, 0.12 and 0.01 for `(4,32)`, `(4,128)` and `(8,32)`. The smaller second layer has 0.30, 0.40 and 0.02. These are invented diagnostics to expose a decision, not measurements of Gemma. Under an additive proxy, the feasible 848-byte recipe has loss $0.08+0.02=0.10$. Both group-32/four-bit layers give $0.08+0.30=0.38$ at 720 bytes. Spending 128 extra bytes on the sensitive small layer improves this proxy substantially.

We enumerate all nine choices in `15_precision_budget.py`. Exhaustive search is appropriate for two layers; the point is to make feasibility and the objective visible, not introduce an unnecessary optimizer. Ties use storage as a secondary criterion. The additive layer-loss assumption is a proxy: actual network errors interact, so a selected recipe must still pass task evaluation.

[Download the complete reference](examples/15_precision_budget.py)

```python
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
```

From the scratch directory containing that file, run:

```bash
python3 15_precision_budget.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "recipe": [
    [
      4,
      32
    ],
    [
      8,
      32
    ]
  ],
  "bytes": 848,
  "loss": 0.1,
  "budget": 900
}
```

The selected recipe should be four-bit/group-32 then eight-bit/group-32, at 848 bytes and loss 0.1. A result exceeding 900 means the budget filter is wrong. A different low-loss recipe may signal that bytes and bits were mixed or that per-layer group ceilings were omitted.

### Understand Pareto choices

A candidate dominates another if it is no worse on every declared objective and strictly better on at least one. Memory, supported-answer rate and latency can conflict, leaving several nondominated choices. An application still needs a declared decision rule: for example, enforce a peak-memory cap, minimum support quality and reload success, then minimize latency. That is different from inventing a weighted sum whose units and weights have no operational meaning.

Do not optimize storage alone when the actual constraint is peak device memory. A selectively higher-precision tensor may require a different kernel or dequantization workspace. A recipe that is small on disk can be slow or unsupported. Exporter support and measured peak memory are feasibility conditions, not optional footnotes after choosing the smallest file.

### How to turn this into phase-04 work

Change one factor at a time first: fix bit width and compare one supported group-size change, or hold grouping fixed and preserve one declared tensor class at higher precision. Retain actual tensor inventories and layer coverage. Only then consider a combined recipe, with a fresh matched baseline. Keep calibration budget and development membership fixed while comparing those recipes.

If no candidate satisfies the frozen constraints, report no selection. Options such as shorter context, another model, another placement or looser quality thresholds are new design decisions requiring a new protocol version; do not retroactively relabel a failing candidate successful.

[[practice:precision-transfer]]

The [comparison guide](project/comparisons.md) records which variable each study isolates. We have chosen how to investigate candidates; now we need a task evaluator that can expose changed facts, missing answers and unsupported claims without hiding them in one average.
