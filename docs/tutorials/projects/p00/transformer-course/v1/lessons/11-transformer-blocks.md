## Lesson 14 — Residual and normalization blocks

**Goal:** derive the exact post-norm residual computation, normalization gradient and dropout ownership. Prerequisites: attention and the FFN. Correct sublayers connected in the wrong order implement a different architecture.

### Addition creates a direct gradient route

A residual operation is u=x+f(x), requiring equal shapes. Its Jacobian is I+J_f, so an upstream gradient travels through both the identity route and the sublayer. This helps optimization by preserving a direct route; it does not guarantee healthy gradients through arbitrary subsequent normalization or depth.

The source uses post-norm:

$$
 y=\operatorname{LN}(x+\operatorname{Dropout}(f(x))).
$$

Pre-norm instead uses `x + Dropout(f(LN(x)))` and commonly has an additional final normalization after the stack. These graphs have different gradients and saved normalization locations. The Small GPT course later uses its own stated pre-norm graph; swapping orders here is an architectural change.

[[diagram:residual-normalization]]

### LayerNorm operates on one token's features

For u∈ℝᴰ,

$$
\begin{aligned}
\mu&=D^{-1}\sum_j u_j,\\
\sigma^2&=D^{-1}\sum_j(u_j-\mu)^2,\\
\hat u_j&=(u_j-\mu)/\sqrt{\sigma^2+\epsilon},\\
y_j&=\gamma_j\hat u_j+\beta_j.
\end{aligned}
$$

The variance denominator is D. With u=`[1,3]`, μ=2, variance 1 and ε=10⁻⁵, initial γ=1/β=0 gives approximately `[-.999995,.999995]`. Normalizing over the token axis would couple different positions and could leak future information. `nn.LayerNorm(D)` uses the last D features separately for each batch/token row; it behaves consistently with batch size one. It differs from BatchNorm and from clipping parameter gradients after backward. [Official LayerNorm definition](source:torch-layernorm).

Equal-feature u has zero centered values and produces β. Epsilon prevents division by zero, but is not a switch that skips the operation. Learned γ/β provide per-feature scale and shift; they add 2D parameters for each separate normalization module.

### Follow its backward signal

Given upstream g=∂L/∂y, define r=g⊙γ and s=√(σ²+ε). Differentiating the centering and variance gives

$$
\nabla_u L=\frac{r-\operatorname{mean}(r)-\hat u\operatorname{mean}(r\odot\hat u)}{s}.
$$

Means are over features of that token. The parameter gradients sum over uses: ∇γ=Σ(g⊙û), ∇β=Σg. The derivative is not merely division by s: every input coordinate influences mean and variance. For u=`[1,2,3]`, g=`[1,0,0]`, γ=1 and the ε→0 limit, the input gradient is approximately `[.204124,-.408248,.204124]`. Its entries sum to zero, reflecting invariance to a constant input shift.

```python
import torch
u = torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True)
norm = torch.nn.LayerNorm(3,eps=1e-5).double()
y = norm(u)
g = torch.tensor([1.,0.,0.],dtype=u.dtype)
(y*g).sum().backward()
centered = u.detach()-u.detach().mean()
s = (centered.square().mean()+norm.eps).sqrt()
hat = centered/s
expected = (g-g.mean()-hat*(g*hat).mean())/s
assert torch.allclose(u.grad,expected,atol=1e-12,rtol=1e-12)
assert abs(u.grad.sum().item()) < 1e-12
```

In a post-norm block, even f(x)=0 leaves y=LN(x), which is usually not x. The direct route passes through LayerNorm. A pre-norm block with a zero sublayer update retains x exactly, before any final stack normalization. This is a useful changed-condition diagnosis of the graph.

### Trace dropout and module ownership

An encoder block executes:

```text
source x
  → self-attention(x, source-key mask)
  → attention-output dropout
  → add x → LayerNorm
  → FFN (including its two dropout sites)
  → add current state → final LayerNorm
```

The decoder has three separate additions/norms: causal self-attention, source cross-attention and FFN. Cross-attention keys/values use encoder memory, but every decoder residual adds to the current **target** stream. Source and target lengths can differ, so adding memory directly would violate that interface.

Inverted dropout uses m/(1−p), with Bernoulli mask m, to preserve an activation's conditional expectation. It does not preserve each sample or make nonlinear normalized outputs equal in expectation. Attention-probability dropout, attention-output dropout and FFN dropout are distinct sites. The reference has no embedding dropout; defaults set the existing dropout probabilities to zero for its deterministic miniature.

The block constructors register unique attention, FFN and norm modules. `ModuleList` later registers each block in the stack; repeating the same object would share weights. Equal initialization values do not imply shared parameters. Do not instantiate trainable modules inside forward or keep them in an unregistered plain list.

### Verify the assembled block

Inspect [TransformerEncoderLayer and TransformerDecoderLayer](examples/translation_model.py). Count two LayerNorm modules (128 default parameters) in an encoder layer and three (192) in a decoder layer. Run the source gradient/future/padding controls once the complete model is assembled. Check `.eval()` disables registered dropout while autograd can remain enabled; `no_grad()` alone does not select evaluation behavior.

[[practice:transformer-blocks-practice-1]]
[[practice:transformer-blocks-transfer]]

### Notes and implementation

Draw post-norm versus pre-norm graphs in my notes, derive the three-feature gradient and mark all dropout/norm sites and unique parameter owners. Predict a zero-sublayer output before calculating it. Continue with the [encoder stack](lessons/15-encoder-stack.md).
