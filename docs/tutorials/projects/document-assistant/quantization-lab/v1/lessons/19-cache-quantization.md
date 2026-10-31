## 19. Change cache precision without changing the weight experiment

A smaller cache can leave room for longer documents, but it can also alter attention. Weight quantization and cache quantization act on different state. Freeze the chosen weight artifact and all task settings before comparing cache policies; otherwise an answer change has two possible causes.

### Keys change the mixture; values change what is mixed

For one query $q$, keys $k_t$ and values $v_t$, attention scores are $a_t=q^Tk_t/\sqrt d$, probabilities are $p_t=\operatorname{softmax}(a)_t$ and output is $o=\sum_t p_tv_t$. A key error $\Delta k_t$ changes score by $q^T\Delta k_t/\sqrt d$, which can change **all** probabilities after normalization. With keys fixed, value errors instead change output by $\Delta o=\sum_t p_t\Delta v_t$. These are different pathways, even if both stored tensors use the same bit width.

For small score changes, softmax has derivative $\partial p_i/\partial a_j=p_i(\mathbf1_{i=j}-p_j)$. It follows by differentiating the numerator and shared denominator. The diagonal is positive, while off-diagonals are negative: increasing one score redistributes probability away from other positions. This local linear approximation is a diagnostic, not an exact formula for large perturbations.

Our scalar-head fixture uses $q=1$, keys `[0,log(3)]` and values `[0,4]`. Head dimension is one, so the scale is one. Probabilities are `[0.25,0.75]` and output is 3. Add `log(3)` to the second key: exponentials become proportional to `[1,9]`, probabilities `[0.1,0.9]`, output 3.6. Restore the keys and increase only the second value from 4 to 5: probabilities stay `[0.25,0.75]`, output becomes 3.75.

### Complete isolated perturbation reference

Create `19_cache_error.py`. `attend()` defines score normalization and weighted sum before the main fixture calls it. The two changed cases deliberately isolate K and V; neither claims to simulate a full low-bit cache kernel. The same check would be harder to interpret if we changed query, key and value together.

[Download the complete reference](examples/19_cache_error.py)

```python
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
```

From the scratch directory containing that file, run:

```bash
python3 19_cache_error.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "base": 3.0,
  "key_error_output": 3.6,
  "value_error_output": 3.75,
  "probabilities": [
    0.25,
    0.75
  ]
}
```

If a value-only change modifies the recorded probabilities, the implementation has mixed up tensors or reused mutable state. If a key change simply adds to the output without renormalization, it is not the attention operation described above. When moving to multiple dimensions, restore the dot product and $\sqrt d$ scale with axes explicitly stated.

### Axes, groups and residual regions matter

[KIVI](source:kivi-paper) motivates asymmetric cache quantization with per-channel key treatment and per-token value treatment, plus a high-precision residual region. Its [implementation](source:kivi-code) has its own supported models and hardware requirements. This tutorial derives the attention-error pathways and storage budget; it does not implement KIVI or establish Gemma/Intel support.

“Per-channel” needs an axis definition. For a `[tokens,channels]` matrix it can mean one scale per channel across a group of tokens; “per-token” can mean scales for channel groups within a token. Write the reduction axes and group boundaries. A large key channel persisting across tokens can make one scaling direction useful; token-specific value ranges can favor another. These are data-distribution motivations requiring measurement for the selected model.

If $T$ tokens retain $R$ recent tokens in high precision, a simplified cache count is compressed storage for $T-R$ plus high-precision storage for $R$, scales/zero points and padding. With a 128-token residual, 1024 total tokens, two-byte original scalars and two-bit older payload, the scalar payload ratio is $((896)(0.25)+(128)(2))/(1024(2))=480/2048\approx0.2344$. Metadata and workspaces increase it; “two bits” does not imply an eightfold reduction in total model memory.

[[practice:cache-quant-transfer]]

### Build a controlled context study

Record weight hash, cache types for K/V, axes/groups, residual policy, actual applied layer coverage and runtime support. Test answerability, evidence at beginning/middle/end, distractors, repeated numbers and cases requiring two far-apart facts. Keep prompt content matched at each length. Distinguish a correct refusal after omitted evidence from a false refusal when evidence was supplied.

If the backend rejects a policy or falls back to float, retain that outcome and inspect actual allocation/logs. Do not label a requested but unused option a successful experiment. Once a candidate passes development constraints, it needs saved identities and restart behavior before serving. We address that persistence boundary next.
