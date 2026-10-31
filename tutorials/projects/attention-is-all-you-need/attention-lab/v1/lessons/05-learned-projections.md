## 4. What the learned projections can change

We can now trace an input matrix to one output row per position. Our parameter choices were simple: Q and K copy X, while V doubles it. A useful model needs to adapt both its comparison rule and the information carried by each contribution. This is the responsibility of the projection parameters.

### Separate model parameters from this input's activations

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

W_Q, W_K and W_V persist between examples. Q, K, V, scores, attention weights A and context C are calculated anew for each input. In particular, the entries of A are **not a table of persistent parameters indexed by sentence positions**. A is produced by the current input and learned projections.

Each W is shared across positions. For a model width d and a single head with comparison width d_k, W_Q and W_K have shape `(d,d_k)`; W_V has shape `(d,d_v)`. The matrices do not acquire extra parameter rows when the sentence becomes longer. The sequence length controls the number of activation rows and pair comparisons.

### Why not use the same transformed vector for every job?

Consider one score before scaling:

$$q_i k_j^T = x_i W_Q W_K^T x_j^T.$$

The receiving row and available row enter through different matrices. This permits an asymmetric relation: position i reading j need not receive the same score as j reading i. If Q=K, the raw self-attention score matrix is symmetric; its row-normalized weights still need not be symmetric because different rows can have different denominators. Separate query/key maps remove the raw-score symmetry constraint as well.

The value projection has a different purpose. Once the comparison assigns weight A_ij, V decides which features travel from source j to receiver i. Requiring the content features to equal the comparison features would unnecessarily tie those choices together. Separate maps provide capacity to learn them separately; they do not guarantee a particular linguistic specialization.

### A controlled change that isolates the value path

Use the identity/doubling settings again so the starting output stays recognizable. Then halve only W_V. Q, K, scores and A are untouched. V halves, so every C row halves. That is a direct consequence of `C = A @ V`, rather than a new selection decision.

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
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
w_q = torch.eye(2, dtype=x.dtype)
w_k = torch.eye(2, dtype=x.dtype)
w_v = 2 * torch.eye(2, dtype=x.dtype)

q = x @ w_q
k = x @ w_k
v = x @ w_v
context, weights = attend(q, k, v)
print(q)
print(k)
print(v)
print(context[0])

half_context, same_weights = attend(q, k, x @ (w_v / 2))
assert torch.allclose(same_weights, weights)
assert torch.allclose(half_context, context / 2)
```

The original first context remains `[1.339523,1.000000]`. After the isolated value change it is approximately `[0.669762,0.500000]`, with exactly the same attention weights. Changing X instead would generally affect all three roles at once, so it would be a poor way to isolate the value projection.

Changing W_Q can change several receivers because the same matrix transforms all their rows. In chapter 9, the first input row `[1,0]` produces a gradient on W_Q's first row. That update can also affect the query for input row `[1,1]`; it is not a private parameter belonging only to token position 0.

### Connect the notation to the implementation

Our handwritten notation multiplies a row on the right, `x @ W`. PyTorch's `nn.Linear` stores its matrix in the transposed convention and evaluates `x @ weight.T + bias`. To copy a general handwritten W into the layer, copy `W.T`. Identity matrices hide that distinction, so it is worth making the orientation explicit now.

The project's `q_proj`, `k_proj`, `v_proj` and `out_proj` are affine layers with biases. Their parameters become useful only when a loss and optimizer adjust them. Calling a forward pass with identity maps demonstrates the computation; it does not train those maps. We will trace the learning path after locating encoder and decoder attention.

These separate maps also make multiple heads possible. Instead of forcing every output feature to use one shared distribution over source positions, we can compute several distributions in projected feature groups. That is the next design question.

**Runnable copy:** [projections and isolated value change](examples/walkthrough/04_projections.py).

[[practice:projections-check]]
