## Lesson 9 — Learned Q/K/V and scaled attention

**Goal:** calculate learned attention and derive the gradient paths for matching and payload transport. Prerequisites: embeddings, position coordinates and matrix gradients. Attention does not directly assign English meanings to features; task loss learns which comparisons and payloads help prediction.

The reference [translation_attention.py](examples/translation_attention.py) uses row vectors. We will first retain a fully worked three-token example, then distinguish rectangular cross-attention and its backward paths. [Inspected source](source:attention-code).

### Start with attention without learned projections

Let each row of a matrix be one token and each column one feature:

$$X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

If we use the same matrix as queries, keys and values, attention is

$$C=\operatorname{softmax}_{\text{row}}\!\left(\frac{XX^T}{\sqrt{2}}\right)X.$$

The square root is two's square root because the query and key vectors have two features. The transpose turns the three key rows into three columns. Consequently `X @ X.T` is a three-by-three table of dot products. A row asks how strongly one token matches all three tokens. The softmax converts that row into weights; multiplying those weights by `X` forms a weighted sum of input vectors.

In the downloaded course `examples/` folder, run this small probe with the Python environment from Lesson 2:

```python
import torch
from translation_attention import attention_math

x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
context, weights = attention_math(x, x, x)
print(weights[0, 0].round(decimals=3).tolist())
print(context[0, 0].round(decimals=3).tolist())
```

The first row is `[0.401, 0.198, 0.401]`; its context is `[0.802, 0.599]`. Both coordinates are combinations of actual input coordinates. This already retrieves information, but the model cannot learn *which representation* should be compared and which representation should be carried onward. It treats the raw input as all three.

### Queries, keys and values have separate jobs

A **query** is the representation used by the receiving position to select information. A **key** is the representation used by a candidate source position to offer a match. A **value** is the payload combined after those matches become weights. These are roles in a calculation. They are not three lists of English words, and a feature has no fixed interpretation such as “plural” before training.

We learn three different linear maps, applied to every token with shared parameters:

$$Q=XW_Q+b_Q,\qquad K=XW_K+b_K,\qquad V=XW_V+b_V.$$

Here `X` is `[T,D]`; each map is `[D,D]` for this example; each bias has `D` entries and is added to every row. The source vectors, not the attention probabilities, feed these projections. Training changes the maps through the final task loss.

For our example use zero biases and these matrices in the *mathematical input-by-output orientation*:

$$W_Q=\begin{bmatrix}1&0\\1&1\end{bmatrix},\quad
W_K=\begin{bmatrix}1&1\\0&-1\end{bmatrix},\quad
W_V=\begin{bmatrix}2&0\\0&3\end{bmatrix}.$$

Then all three products are visible:

$$
\begin{aligned}
Q&=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
\begin{bmatrix}1&0\\1&1\end{bmatrix}
=\begin{bmatrix}1&0\\1&1\\2&1\end{bmatrix},\\
K&=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
\begin{bmatrix}1&1\\0&-1\end{bmatrix}
=\begin{bmatrix}1&1\\0&-1\\1&0\end{bmatrix},\\
V&=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
\begin{bmatrix}2&0\\0&3\end{bmatrix}
=\begin{bmatrix}2&0\\0&3\\2&3\end{bmatrix}.
\end{aligned}
$$

PyTorch stores `nn.Linear.weight` as `[out_features,in_features]` and computes `X @ weight.T + bias`. For example the stored query weight is `[[1,1],[0,1]]`. Transposing it produces the `W_Q` printed above. Mixing those two conventions is a common source of apparently reasonable but wrong results. [Linear contract](source:torch-linear).

[[diagram:qkv-matrices]]

### Read the score table one cell at a time

The transpose of the key matrix and the product are

$$K^T=\begin{bmatrix}1&0&1\\1&-1&0\end{bmatrix},\qquad
R=QK^T=\begin{bmatrix}1&0&1\\2&-1&1\\3&-1&2\end{bmatrix}.$$

For example `R[1,1]` pairs query row `[1,1]` with key row `[0,-1]`: `1×0 + 1×(-1) = -1`. The output's row still identifies the receiving token; its column identifies the source token. A negative score is valid. It becomes a smaller positive probability after softmax.

Divide every score by `sqrt(2)`. The first row becomes approximately `[0.707,0,0.707]`. Exponentiation gives `[2.028,1,2.028]`; dividing by their sum `5.056` gives `[0.401,0.198,0.401]`. Softmax runs across **source keys**, the last dimension. It does not normalize across receiving tokens.

Scaling has a statistical motivation. If each query/key coordinate is independent with mean zero and variance one, each coordinate product has variance one and the sum of `d_k` such products has variance `d_k`. Dividing the sum by `sqrt(d_k)` returns variance near one under those assumptions. It limits a tendency toward extreme logits and saturated softmax. It does not force every trained query to have unit norm. When heads are introduced next, `d_k` becomes the width of one head. [Attention derivation](source:attention-paper).

### The values supply the output coordinates

The first context row is the probability row times the value matrix:

$$[0.401,0.198,0.401]
\begin{bmatrix}2&0\\0&3\\2&3\end{bmatrix}
\approx[1.604,1.797].$$

The first coordinate is `0.401×2 + 0.198×0 + 0.401×2`; the second is `0.401×0 + 0.198×3 + 0.401×3`. These are payload mixtures. Scores decide where to read; values determine what is read.

Without dropout, an allowed attention row sums to one, so its context lies in the convex hull of its value rows. That statement concerns the intermediate context. A subsequent learned output projection can place the final output elsewhere. Attention weights also need not be symmetric: the second query's preference for the first key can differ from the first query's preference for the second key.

### Implement and inspect the core operation

Open `translation_attention.py`. Its `attention_math` is the complete explicit operation. Keep `math` and `torch` imported at the top:

```python
def attention_math(q, k, v, allowed=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if allowed is None:
        weights = scores.softmax(dim=-1)
    else:
        scores = scores.masked_fill(~allowed, -torch.inf)
        nonempty = allowed.any(dim=-1, keepdim=True)
        safe_scores = torch.where(nonempty, scores, torch.zeros_like(scores))
        weights = safe_scores.softmax(dim=-1).masked_fill(~allowed, 0)
    return weights @ v, weights
```

The last two branches reserve a safe convention for a wholly blocked row: zero probabilities and zero context, rather than softmax of only negative infinities. We will explain actual masks in Lesson 11. For today's unmasked call, only the first branch is used. `transpose(-2,-1)` swaps the final two axes and preserves any batch/head axes.

Run the module's controlled learned example:

```bash
python translation_attention.py
```

It prints `Q`, `K`, `V`, `Q @ K.T`, `A`, and `A @ V`. The three context rows round to `[[1.604,1.797],[1.851,1.140],[1.924,1.067]]`. All the fixed projections and the identity output projection are set inside `main()` for this inspection. Real training initializes them and updates them; it does not replace them with these numbers.

To verify the independent known-answer case without running the later training checks:

```bash
python translation_checks.py TranslationChecks.test_dot_product_and_weighted_value_known_answer
```

That case computes the two-key softmax by hand with `math.exp`, then checks the weighted payload against the attention function. A wrong softmax axis, missing scale or elementwise multiply fails it.

### Rectangular attention preserves the receiving positions

For cross-attention, receiving states Y have T rows and source memory X has S rows. Queries Q=YW_Q have shape [T,d], while K=XW_K and V=XW_V have [S,d]. Scores and probabilities have [T,S]; context has [T,d]. The output follows the query axis even when T≠S. A changed source value affects transported payloads without changing weights if Q/K are held fixed. Changing the source representation normally changes both K and V, so it can alter both paths.

### Derive the backward operation

Let C=AV and let G=∂L/∂C be an upstream [T,d] gradient. Matrix multiplication first gives ∇V=AᵀG and ∇A=GVᵀ. For one probability row a with upstream row gₐ, the softmax Jacobian is diag(a)−aaᵀ. Therefore each score derivative is aⱼ(gₐⱼ−Σₖaₖgₐₖ). This subtraction couples the keys competing in a row.

$$
\begin{aligned}
R&=QK^T/\sqrt d,\\
\nabla_Q L&=(\nabla_R L)K/\sqrt d,\\
\nabla_K L&=(\nabla_R L)^TQ/\sqrt d.
\end{aligned}
$$

Continue through the learned projections: ∇W_Q=Yᵀ∇Q and ∇W_K=Xᵀ∇K, with analogous value gradients. The source-state gradient contains **both** the key and value routes; the receiving state has the query route and a residual route in the full block. This explains why treating attention weights as detached constants trains only part of the intended model.

A two-key example makes the competing gradients visible. Take d=1, q=0, keys [1,−1], values [2,6], and L equal to the scalar context. Both weights are .5; context is 4. The probability upstream gradient is [2,6], so score gradients are [−1,1]. The query gradient is −2; key gradients are zero because q=0; value gradients are [.5,.5]. A zero query gradient is not implied by equal weights.

```python
import torch
from translation_attention import attention_math
q = torch.tensor([[0.]], dtype=torch.float64, requires_grad=True)
k = torch.tensor([[1.],[-1.]], dtype=torch.float64, requires_grad=True)
v = torch.tensor([[2.],[6.]], dtype=torch.float64, requires_grad=True)
context, weights = attention_math(q,k,v)
context.sum().backward()
assert context.item() == 4 and q.grad.item() == -2
assert torch.equal(k.grad, torch.zeros_like(k))
assert torch.equal(v.grad, torch.full_like(v,.5))
```

Masked entries have zero probability and score derivative under the reference convention. Dropout and the output projection add their own backward operations; the formulas above describe the undropped core. The new [applied probes](examples/model_learning_probes.py) compare analytic projected gradients with autograd and finite differences.

[[practice:qkv-practice-1]]
[[practice:qkv-transfer]]

### Notes and implementation

In my notes, reproduce Q/K/V, scores, probabilities and context from the worked matrices. Define the row/column meaning for unequal lengths, then derive the two key/value gradient routes. Change a payload and a key separately before running a probe. Continue with [multi-head implementation](lessons/08-heads.md).
