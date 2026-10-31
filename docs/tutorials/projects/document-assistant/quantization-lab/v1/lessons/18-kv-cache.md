## 18. Derive cache growth from the state actually stored

Weights persist across requests; KV cache stores intermediate attention state for a sequence. During prefill the model processes prompt positions. During decode, retaining prior keys and values avoids recomputing the entire history for every token. This saves computation while consuming memory that depends on context length.

### Start with the logical tensors and axes

For a conventional unshared attention layer, a key tensor can have logical shape `[batch, kv_heads, retained_tokens, key_head_dim]`, with a corresponding value tensor. If key/value dimensions match at $d$ and both use $b$ bytes per scalar, logical bytes are $2BHTdb$. The factor 2 counts K and V; $H$ counts **KV heads**, not query heads. Grouped-query attention lets several query heads use the same key/value head. It reduces stored KV heads relative to ordinary multihead attention without reducing query heads to the same number.

This is a logical count, not a universal physical layout. A backend may transpose axes, pad dimensions, reserve a maximum capacity, page blocks, share caches, or keep separate key/value types. If K and V dimensions/types differ, count them separately. If a window retains only recent positions, use its actual retained count rather than full sequence length.

### Count distinct owners once

Our three-layer fixture has two sliding layers sharing one owner `s`, with window 512, two KV heads and dimension 64. A third layer owns a full-context cache `f`, two KV heads and dimension 128. Use batch 1 and two bytes per scalar. At $T=2048$, sliding storage is $2(1)(2)(512)(64)(2)=262,144$ bytes. Full storage is $2(1)(2)(2048)(128)(2)=2,097,152$. Total is 2,359,296 bytes. Counting the shared sliding owner twice would overestimate this logical fixture.

At lengths 512, 2048 and 4096, totals are 786,432; 2,359,296; and 4,456,448 bytes. The sliding term saturates at 512 while the full-context term grows. Batch two doubles these counts if each sequence has the same retained length. Different-length batches need per-sequence/block accounting.

![Shared and windowed cache owners](figures/cache-owners.svg)

### Complete owner-based reference

Create `18_cache_budget.py`. Each layer records owner, window, KV head count and dimension. The function deduplicates owners and counts retained tokens. The fixture assumes layers sharing an owner agree on storage shape/policy; a real manifest validator must reject conflicting owner descriptions instead of accepting the first silently.

[Download the complete reference](examples/18_cache_budget.py)

```python
"""Count explicit cache owners; architecture fixture is not Gemma's layout."""
import json


def cache_bytes(layers, length, batch, bytes_per_scalar):
    owners, total = set(), 0
    for layer in layers:
        owner = layer["owner"]
        if owner in owners:
            continue
        owners.add(owner)
        retained = min(length, layer["window"]) if layer["window"] else length
        total += 2 * batch * retained * layer["kv_heads"] * layer["head_dim"] * bytes_per_scalar
    return total


def main():
    # Layers 1 and 2 share owner 's'; layer 3 owns a full-context cache 'f'.
    layers = [{"owner": "s", "window": 512, "kv_heads": 2, "head_dim": 64},
              {"owner": "s", "window": 512, "kv_heads": 2, "head_dim": 64},
              {"owner": "f", "window": None, "kv_heads": 2, "head_dim": 128}]
    sizes = [cache_bytes(layers, n, 1, 2) for n in [512, 2048, 4096]]
    assert sizes == [786432, 2359296, 4456448]
    assert cache_bytes(layers, 2048, 2, 2) == 2*sizes[1]
    print(json.dumps({"lengths": [512, 2048, 4096], "bytes": sizes,
                      "scope": "deduplicated logical cache fixture"}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 18_cache_budget.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "lengths": [
    512,
    2048,
    4096
  ],
  "bytes": [
    786432,
    2359296,
    4456448
  ],
  "scope": "deduplicated logical cache fixture"
}
```

The expected sizes are logical scalars times storage bytes. They omit scales, residual buffers, page rounding, alignment, workspaces and allocator reservation. A measured number exceeding this estimate is not automatically a memory leak. Inspect the backend's allocation policy and compare changes with length. A static full-capacity allocation can be constant while the logical number of used tokens grows.

### Apply the reasoning to the selected architecture

The [pinned configuration](source:gemma-config) includes layer types, distinct head dimensions and KV-sharing metadata. The fixture is deliberately not a reconstruction of Gemma's cache-owner map. To obtain that map, inspect the exact runtime architecture implementation and observe allocation/ownership; the sharing-count field alone does not fully specify physical buffers or backend support. Record which layers own caches, which reuse them, local retention rules and actual allocation precision. Do not multiply every layer by one global head size and present that as measured Gemma memory.

For phase 05, choose a verified weight artifact first. Use bounded complete-prompt lengths such as short/medium/long values that fit the gate, keeping output allowance fixed. For each length record supplied evidence location, truncation/drop decisions, used/reserved/peak cache memory, total host/device memory, prefill/TTFT/decode and task support. A “needle” fact at one position is a useful diagnostic, not proof of robust long-document reasoning.

[[practice:cache-transfer]]

We now understand what grows with context and what may saturate. Next we hold the weight artifact fixed while changing supported cache precision/policy, paying attention to why key errors and value errors have different effects.
