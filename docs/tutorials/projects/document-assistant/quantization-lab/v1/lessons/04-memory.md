## 4. Budget the memory that is live at the same time

We now know which source and runtime a reference would use. The next question is whether a complete request fits. A four-bit weight file may fit on disk while model loading, tokenization, context state or a large temporary operation exceeds the device budget. We need an arithmetic estimate and a measurement plan.

### Bits, bytes and the unit conversion

A bit has two possible values. A byte holds eight bits. For $N$ stored scalar parameters using exactly $b$ bits each, the packed payload is $\lceil Nb/8\rceil$ bytes. A decimal gigabyte is $10^9$ bytes; a gibibyte is $2^{30}=1,073,741,824$ bytes. Keep the label with every number. Dividing decimal parameter counts by binary memory capacities without converting units can hide a significant gap.

Consider a **hypothetical** inventory of eight billion stored values. At two bytes per BF16 value its raw payload is 16,000,000,000 bytes, about 14.901 GiB. Four-bit payload alone is 4,000,000,000 bytes. Suppose each group of 32 values also stores a two-byte scale: 250,000,000 groups add 500,000,000 bytes. Total toy storage is 4.5 billion bytes, or 4.191 GiB. It is a 3.56-fold payload-plus-scale reduction from BF16, not exactly fourfold. Real formats add metadata and may leave some tensors at higher precision.

### Separate lifetimes instead of adding unrelated peaks

For a fixed placement, approximate simultaneously live memory as

$$M_{\mathrm{live}}=M_{\mathrm{weights}}+M_{\mathrm{cache}}+M_{\mathrm{workspace}}+M_{\mathrm{other}}.$$

Weights are mostly persistent during inference. KV cache depends on retained sequence state. Workspace includes temporary activations, kernel scratch and possibly dequantization buffers. Other memory includes runtime/context objects and nonmodel allocations. The budget condition is $M_{\mathrm{live}}+R\le M_{\mathrm{capacity}}$, where $R$ is reserved headroom. This is meaningful only for allocations sharing a memory pool; host RAM and device-local memory need separate rows when a model is offloaded or mapped.

For a 12 GiB illustrative capacity, reserve 2 GiB and allow 1 GiB workspace plus 2 GiB cache. The 4.191 GiB toy weights give 9.191 GiB including reserve, so the plan fits. If cache grows to 8 GiB, the same model needs 15.191 GiB including reserve and fails. Reducing weight bytes did not remove the context constraint.

![Memory lifetimes and capacity](figures/memory-budget.svg)

The figure shows a budget, not a measured allocation trace. Do not add the largest loading transient to a later steady-state cache peak if they never overlap. Conversely, do not ignore conversion that holds both source and destination weights simultaneously. Keep acquisition/conversion/calibration/inference peaks as separate operations, with overlap documented inside each.

### Execute the arithmetic and change the context allowance

Create `04_memory_budget.py`. `weight_bytes()` rounds payload bytes and group counts upward. That matters for small/nondivisible tensors. Counting one giant parameter vector underestimates per-tensor padding, so this remains a conceptual model until a real tensor inventory is supplied.

[Download the complete reference](examples/04_memory_budget.py)

```python
"""Illustrative arithmetic; no device capacity or model memory is measured."""
import json
import math

GIB = 2 ** 30


def weight_bytes(parameters, bits, group, scale_bytes):
    return math.ceil(parameters * bits / 8) + math.ceil(parameters / group) * scale_bytes


def main():
    # Hypothetical 8-billion stored scalars, not a Gemma tensor inventory.
    count = 8_000_000_000
    bf16 = count * 2
    q4 = weight_bytes(count, 4, 32, 2)
    reserve, workspace, cache = 2 * GIB, 1 * GIB, 2 * GIB
    capacity = 12 * GIB
    total = q4 + reserve + workspace + cache
    assert q4 == 4_500_000_000 and total < capacity < bf16
    print(json.dumps({"bf16_gib": round(bf16 / GIB, 3), "q4_gib": round(q4 / GIB, 3),
                      "total_gib": round(total / GIB, 3), "fits": total <= capacity,
                      "longer_cache_fits": q4 + reserve + workspace + 8 * GIB <= capacity}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 04_memory_budget.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "bf16_gib": 14.901,
  "q4_gib": 4.191,
  "total_gib": 9.191,
  "fits": true,
  "longer_cache_fits": false
}
```

The output should report `fits: true` and `longer_cache_fits: false`. The inputs are visible assumptions, not detected hardware capacity. If your hand calculation differs, check decimal versus binary units, scale count, then whether reserve was included twice. None of those issues is a model-quality problem.

### What phase 00 must measure

Record available host RAM and device capacity before loading, peak during load, post-load steady state, peak after prefill, peak during decode and post-request retained allocation. State how each number is collected and whether it represents allocated, reserved or resident memory. RSS and a device allocator's reserved count answer different questions; summing them may double count shared memory. On a shared device record competing allocations rather than treating “capacity minus my tensors” as free memory.

Run a short context before a longer one, retain failed/OOM runs, and bound output tokens so a request cannot grow indefinitely. A fresh-process run distinguishes persisted allocator reservation from per-request state growth. Record placement and kernel behavior; compressed weights expanded at load time may save disk without saving the intended live memory.

[[scenario:budget-choices]]

[[practice:memory-transfer]]

This completes a useful first session: answer contract, runnable setup boundary, compatibility criteria and a calculable budget. The memory allowance now gives document preparation a concrete input-token constraint. Next we make document bytes and evidence IDs reproducible before testing compressed models.
