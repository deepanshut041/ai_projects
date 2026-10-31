## Lesson 5 — let each position read its prefix

Each input position now carries an ID and predicts the next ID. Attention turns a position's feature vector into a mixture of earlier feature vectors. P00 supplied the mechanism; here you rebuild the causal-only path and verify that it cannot use future text.

### Read the dimensions before the equation

An embedding will turn IDs `[B,T]` into vectors `X[B,T,D]`. Three learned projections produce queries Q, keys K and values V. A query describes what this position seeks; a key provides features for matching; a value provides the content mixed into the result. These are learned vectors, not linguistic labels you assign by hand.

A linear layer computes `X @ W.T + bias` in PyTorch's stored weight convention. A fused D→3D projection simply puts Q, K and V outputs beside each other, then splits them. It changes how we arrange the calculation, not the causal rule.

With D=64 and H=4, reshape each projection from `[B,T,64]` to `[B,T,4,16]`, then transpose to `[B,4,T,16]`. Each head has its own 16-feature score calculation. Do not reshape directly into `[B,4,T,16]`: that would rearrange which features belong to a time position.

[[diagram:attention]]

### Work one head by hand

For a head with d features,

$$S_{ij}=\frac{\sum_{r=1}^{d}Q_{ir}K_{jr}}{\sqrt d},\qquad
A_{ij}=\frac{\exp S_{ij}}{\sum_{k\;\mathrm{allowed}}\exp S_{ik}},\qquad
O_i=\sum_{j\;\mathrm{allowed}}A_{ij}V_j.$$

Index i is the query position, j the key position, and r the feature being summed. A causal query at i may use keys j≤i. We include the diagonal: position i's input is known, and it predicts the **next** token. Masking the diagonal unnecessarily removes valid context.

The square-root scale comes from a variance argument: if d product terms are independent with unit variance, their sum has variance d and standard deviation √d. Dividing by √d keeps the score's scale roughly stable as d changes. Learned activations do not satisfy those assumptions exactly; this motivates the scaling, not a proof that every observed score has variance one.

Set all allowed scores to zero and values to `[2,4,8]`. Row zero can only see 2; row one averages 2 and 4; row two averages all three. Outputs are `[2,3,14/3]`. Future keys must have zero probability, not a score of zero. Setting forbidden scores to zero would give them positive exponential mass. Use negative infinity before softmax.

[[scenario:visibility]]
[[practice:mask]]

### Implement the score-to-mixture operation

Create `gpt_attention.py` with the imports and `attend` function. The explicit path is our reference. The optional library path uses PyTorch SDPA as an independent comparison under matching masks and dropout settings; `True` in its Boolean mask means allowed.

<!-- file:gpt_attention.py mode:write -->
```python
"""Readable causal attention and an independent library comparison path."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def attend(q, k, v, allowed, *, optimized=False, dropout=0.0, training=False):
    probability = dropout if training else 0.0
    if optimized:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                              dropout_p=probability)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = scores.softmax(dim=-1)
    weights = F.dropout(weights, p=probability, training=training)
    return weights @ v
```

The score matrix has `[B,H,T,T]` entries. Softmax reduces over its last axis, the keys. `weights @ v` sums those key contributions and returns `[B,H,T,d]`. The library API applies the probability passed in `dropout_p`; evaluation must explicitly pass zero. [The inspected API contract](source:sdpa) is why `probability` depends on training mode here.

### Add projections and head layout

Append the attention module:

<!-- file:gpt_attention.py mode:append -->
```python
class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)

    def forward(self, x, valid, *, optimized=False):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(tensor):
            return tensor.reshape(batch, time, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = map(heads, (q, k, v))
        causal = torch.ones(time, time, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        # Model input requires a real first token. Even padded queries see it;
        # they are discarded by the loss. No softmax row is entirely masked.
        out = attend(q, k, v, allowed, optimized=optimized,
                     dropout=self.dropout, training=self.training)
        out = out.transpose(1, 2).contiguous().view(batch, time, width)
        return F.dropout(self.output(out), p=self.dropout, training=self.training)
```

`valid[:,None,None,:]` broadcasts a per-example key mask across heads and query rows. `causal[None,None]` broadcasts across batch and heads. Their AND enforces both rules. The model will reject all-PAD and non-right-padded input before this function runs.

After mixing, the transpose puts time before heads again. `.contiguous()` arranges those values in contiguous memory before `.view()` merges head features into width D. The output linear layer lets channels from different heads interact. Residual dropout regularizes that update; it will be disabled during deterministic comparisons.

Check the handwritten average now:

```bash
python - <<'PYCODE'
import torch
from gpt_attention import attend
q=torch.zeros(1,1,3,1)
v=torch.tensor([2.,4.,8.]).reshape(1,1,3,1)
mask=torch.ones(3,3,dtype=torch.bool).tril()
print(attend(q,q,v,mask).flatten())
PYCODE
```

Expect approximately `tensor([2.0000,3.0000,4.6667])`. Change the last value to 80: the first two outputs must stay the same. This tests the direction of information flow, not just output shape.

### Compare gradients, not only outputs

The next lesson's `test_gpt_model.py` compares explicit attention with SDPA in float64, compares Q/K/V gradients, and uses centered finite differences through `gradcheck`. In a centered difference, perturb one value by +ε and −ε and divide the output difference by 2ε. Too large an ε adds approximation error; too small an ε loses precision. Float64 and tiny tensors make this diagnostic more reliable.

[[practice:gradient-check]]

Keep dropout off for comparisons. With random dropout, two equally correct calls may choose different masks. Do not loosen tolerance until every mismatch disappears; first compare dtype, mask polarity, scaling and head order.

Attention now provides a causal update. The next lesson adds the residual stream, per-position transformations and vocabulary projection to make a complete model.

[Continue to Lesson 6](lessons/06-complete-gpt.md)

### Complete files from this lesson

[gpt_attention.py](examples/gpt_attention.py)
