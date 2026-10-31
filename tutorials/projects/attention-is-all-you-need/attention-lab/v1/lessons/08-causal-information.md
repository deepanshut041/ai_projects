## 7. Decoder causality: match training to what generation can know

Return to our translation task. When generating “Nous lisons des livres quotidiennement”, the decoder has the source sentence, but it does not yet have the target tokens it is supposed to predict. After “Nous”, it must predict “lisons” using the prefix that exists so far.

During training, the complete reference translation is available. We would like to train all target positions together in a matrix, but we must prevent a position from using a future reference token as an input shortcut.

### Show the exact input and label alignment

BOS denotes a beginning token; EOS denotes an ending token. Continue to use whole displayed words only to show ownership; the eventual tokenizer decides actual token boundaries.

| Decoder position | Input token at this position | Desired next token | Allowed target-input prefix |
| --- | --- | --- | --- |
| 0 | BOS | Nous | BOS |
| 1 | Nous | lisons | BOS, Nous |
| 2 | lisons | des | BOS, Nous, lisons |
| 3 | des | livres | BOS, Nous, lisons, des |
| 4 | livres | quotidiennement | BOS, Nous, lisons, des, livres |
| 5 | quotidiennement | EOS | BOS, Nous, lisons, des, livres, quotidiennement |

At position 1, the label is “lisons”. Input position 2 contains exactly that token. Without a restriction, position 1 could read it through self-attention and make a prediction using information unavailable at generation time. This is the concrete leakage path the causal mask must remove.

The rule is `key_index <= query_index`. It allows the current input token because the prediction at that row is for the **next** token. The label shift and the mask work together; a mask does not repair unshifted labels.

![Training input, next-token labels and the allowed causal pairs](images/25-decoder-causality.svg)

### Apply the rule to the same four-row arithmetic fixture

To keep every number inspectable, use the first four-position mask pattern with our existing numerical X. These rows now stand in for decoder states; they are not literal embeddings of the tokens in the table.

```text
             key 0  key 1  key 2  key 3
query 0       True  False  False  False
query 1       True   True  False  False
query 2       True   True   True  False
query 3       True   True   True   True
```

Query 0 reads only value 0, so its gathered vector is `[2,0]`. Query 1 compares only keys 0 and 1 and receives `[0.660477,1.339523]`. Query 2 can use keys 0–2 and receives `[1.503490,1.503490]`. Query 3 can use all four keys and receives `[1,1]`. The algorithm is the masked attention from chapter 6 with a different allowed matrix.

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
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
positions = torch.arange(x.shape[0])
causal = positions[None, :] <= positions[:, None]
context, weights = attend(x, x, 2 * x, causal)
print(causal)
print(context)

changed = x.clone()
changed[2] = torch.tensor([1000., -1000.], dtype=x.dtype)
changed_context, _ = attend(changed, changed, 2 * changed, causal)
assert torch.allclose(changed_context[:2], context[:2])

valid_keys = torch.tensor([True, True, True, False])
combined = causal & valid_keys[None, :]
combined_context, _ = attend(x, x, 2 * x, combined)
print(combined_context)
```

### Check future independence separately from padding

The perturbation changes position 2 while keeping all four positions real. Outputs at positions 0 and 1 must remain unchanged. This tests causality itself. If padding already hid position 2, an unchanged result would not independently prove that the causal rule worked.

In a separate padding example, let position 3 be padding. Combine restrictions with AND: a key must be both real and no later than the query. Query 3 then reads positions 0–2 and returns `[1.333333,1.333333]`. The first three outputs keep their earlier causal results.

The source encoder usually can read the whole valid English sentence: those tokens were supplied before translation began. Target self-attention must obey the prefix boundary. Training can calculate many target rows together under that restriction; generation still constructs the next token sequentially. Parallel matrix computation during training does not imply that the complete translation is already known at inference time. [Paper, decoder stack and attention applications](https://arxiv.org/html/1706.03762v7).

**Runnable copy:** [causal pairs and future perturbation](examples/walkthrough/07_causal.py).

[[practice:causal-check]]
