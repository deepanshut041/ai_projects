## 5. Why use several attention heads?

One head produces one distribution over source positions for each receiver. In our example, receiver 0 uses weights `[0.334881,0.165119,0.334881,0.165119]`. The first output feature and the second output feature both use that same distribution. W_V can change the content carried by a source row, but within this head the row still has one scalar contribution weight.

That is the restriction multiple heads relax. A layer can give different projected feature groups independently calculated distributions over positions, then combine the gathered features. It does not have to compress every relationship into one mixture.

### What a head owns

Each head has query, key and value features, its own scores and its own softmax distribution for each receiver. Every head can still read every permitted position. Heads divide projected **features**, not the sentence into segments.

In the implementation, a single full-width projection produces all head features together, then a reshape labels which features belong to which head. This is equivalent to collecting separate per-head projection matrices side by side. Since projection happens before splitting, each head's features can depend on the whole input row.

### Work the two heads for the same receiving position

Keep X and the identity/doubling projections, but use two heads of width one. For receiver 0:

| Calculation | Head 0 | Head 1 |
| --- | --- | --- |
| Query feature | `[1]` | `[0]` |
| Keys at positions 0, 1, 2, 3 | `[1], [0], [1], [0]` | `[0], [1], [1], [0]` |
| Values at positions 0, 1, 2, 3 | `[2], [0], [2], [0]` | `[0], [2], [2], [0]` |
| Scores, scaled by sqrt(1) | `[1,0,1,0]` | `[0,0,0,0]` |
| Weights | `[0.365529,0.134471,0.365529,0.134471]` | `[1/4,1/4,1/4,1/4]` |
| Gathered feature | `1.462117` | `1.000000` |

The second head averages because this particular receiver's second query feature is zero. It is not assigned a special language task. The first head prefers positions 0 and 2. The two gathered features therefore come from different mixtures of the same source positions.

Concatenate them to get `[1.462117,1.000000]`, then apply W_O, the output projection. W_O is identity in this check so the two gathered features stay visible. In a trained model it can mix them into the next layer's feature space. It does not recalculate their attention weights.

![One distribution in one head versus separately normalized distributions in two heads](images/24-head-layout.svg)

### Follow the axes through the implementation

Start with `(B,T,d_model)`. Reshape to `(B,T,H,d_head)` and transpose to `(B,H,T,d_head)`. Now each `(batch,head)` pair holds an ordinary position-by-feature attention problem. After gathering, transpose back to `(B,T,H,d_head)` **before** joining head features into each position's row.

```python
import math
import torch

def attend(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    return weights @ v, weights

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [0., 0.]]], dtype=torch.float64)
w_q = torch.eye(2, dtype=x.dtype)
w_k = torch.eye(2, dtype=x.dtype)
w_v = 2 * torch.eye(2, dtype=x.dtype)
w_out = torch.eye(2, dtype=x.dtype)
q, k, v = x @ w_q, x @ w_k, x @ w_v

batch, positions, width = x.shape
heads = 2
assert width % heads == 0
head_width = width // heads
q_heads = q.reshape(batch, positions, heads, head_width).transpose(1, 2)
k_heads = k.reshape(batch, positions, heads, head_width).transpose(1, 2)
v_heads = v.reshape(batch, positions, heads, head_width).transpose(1, 2)

head_context, head_weights = attend(q_heads, k_heads, v_heads)
joined = head_context.transpose(1, 2).reshape(batch, positions, width)
output = joined @ w_out

print(q_heads.shape)
print(head_weights[0, 0, 0])
print(head_weights[0, 1, 0])
print(output[0])
```

The output rows are approximately `[1.462117,1.000000]`, `[1.000000,1.462117]`, `[1.462117,1.462117]` and `[1,1]`. They differ from the one-head results because the comparison features, per-head scaling and normalization groups changed. The reshape alone would not justify an improvement claim.

If you flatten while heads still precede positions, elements can end up in the wrong token row even though the final shape looks plausible. This is why the inverse transpose is part of correctness rather than cosmetic layout code.

At fixed model width, increasing H reduces each head's width. More independently weighted mixtures come with narrower per-head feature spaces. Neither a larger head count nor a diagram assigning “grammar” to one head and “meaning” to another establishes a better model. The paper motivates multiple representation subspaces; useful specializations and task quality require trained-model evidence. [Paper, section 3.2.2](https://arxiv.org/html/1706.03762v7).

**Runnable copy:** [two heads with visible score and value paths](examples/walkthrough/05_heads.py).

[[practice:heads-check]]
