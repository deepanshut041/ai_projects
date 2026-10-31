## 8. Let the decoder read the encoded source

Causal self-attention lets a decoder position use its allowed French prefix. That alone provides no path from the English sentence to the next prediction. We need a second attention operation that reads the encoder's source representations.

For the row predicting “lisons”, the receiver is a decoder state built from its allowed prefix. The available memory consists of all valid encoded source rows. **Cross-attention** uses the decoder state to form Q and encoder memory to form K and V.

### Follow the two streams into one output

```text
encoded English source rows ── W_K ── K
                            └─ W_V ── V
causal French prefix states ── W_Q ── Q

Q K^T → source weights for each target receiver → weights V
```

The receiver's target position survives. The source-position axis is summed out. The gathered source features continue through decoder computation and eventually a vocabulary projection to support the next-token prediction. Cross-attention itself still returns features.

![Two target receivers reading four source memory rows](images/26-cross-attention.svg)

### Why source and target lengths need not match

Our displayed source has four words while its target has five. Translation is not a one-source-row-to-one-target-row mapping. In the small program, use four source states and two target states to expose the rectangular geometry. The target states are chosen vectors for two receiving rows, not outputs of an implemented French decoder.

With Q shaped `(2,2)` and K shaped `(4,2)`, scores and weights have shape `(2,4)`. Multiplying by V shaped `(4,2)` returns `(2,2)`: one gathered vector for each of the two target receivers.

```python
import math
import torch

def attend(q, k, v, allowed=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    allowed = torch.broadcast_to(allowed, scores.shape)
    has_keys = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, -torch.inf)
    safe_scores = torch.where(has_keys, scores, torch.zeros_like(scores))
    shifted = safe_scores - safe_scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    weights = torch.where(has_keys, weights, torch.zeros_like(weights))
    return weights @ v, weights

torch.set_printoptions(precision=6, sci_mode=False)
src = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
tgt = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
w_q = torch.eye(2, dtype=src.dtype)
w_k = torch.eye(2, dtype=src.dtype)
w_v = 2 * torch.eye(2, dtype=src.dtype)

q = tgt @ w_q
k = src @ w_k
v = src @ w_v
context, weights = attend(q, k, v)
print(weights.shape)
print(context)

source_validity = torch.tensor([[True, True, True, False]])
masked_context, _ = attend(q, k, v, source_validity)
print(masked_context)
```

The first two familiar queries recover `[[1.339523,1.000000],[1.000000,1.339523]]`. If slot 3 is padding for a shorter source, excluding it gives `[[1.604448,1.197776],[1.197776,1.604448]]`. The numerical operation did not change; the origin and ownership of its vectors did.

### Which masks belong here?

The source key mask has four entries because there are four memory rows. A two-entry target-validity row cannot serve as that mask. Target padding and target-loss masking are separate responsibilities.

Do not apply a target-time triangle over source indices. Source index 2 is not a future French token. The entire English source is available before generation. The decoder state supplying the query must already respect target causality; cross-attention may then read the full valid source. This is the encoder–decoder attention role described in [the paper's section 3.2.3](https://arxiv.org/html/1706.03762v7).

### Connect the roles to the actual call

The project calls the class as `layer(src, tgt=target_states)`. Here `src` means the memory that supplies K and V; `tgt` supplies Q. Without tgt, the same src supplies all three and the operation is self-attention. Output length follows tgt when it is supplied.

The reusable class requires matching batch size, feature width, dtype and device, while source and target sequence lengths can differ. It rejects `causal=True` with tgt supplied because that flag implements a self-attention triangle, not a cross-sequence time relation.

We have now located the three uses of attention: source-to-source, target-prefix-to-target-prefix, and target-to-source. Next we connect the translation objective to the parameters that control their comparisons and contributions.

**Runnable copy:** [rectangular cross-attention and source mask](examples/walkthrough/08_cross.py).

[[practice:cross-check]]
