## Lesson 20 — Layer normalization and its gradients

**Goal:** calculate normalization over a token's feature axis and understand the coupling it introduces. Prerequisites: mean/population variance, derivatives and residual maps. Normalization acts on a representation; it does not normalize vocabulary probabilities or replace a loss.

### Normalize one feature vector at a time

For one token x with D features, define

$$\begin{aligned}
\mu&=\frac1D\sum_ix_i,\qquad v=\frac1D\sum_i(x_i-\mu)^2,\\
s&=\sqrt{v+\epsilon},\qquad \hat x_i=(x_i-\mu)/s,\\
y_i&=\gamma_i\hat x_i+\beta_i.
\end{aligned}$$

Gamma and beta are learned featurewise scale and offset, shape `[D]`, shared across positions/examples. Epsilon is inside the square root. D counts features, not valid tokens or batches. PyTorch LayerNorm uses population variance; its normalized_shape defines the last axes being reduced.

For `[B,T,D]` and `LayerNorm(D)`, calculate independent μ,v for each `(b,t)`. Do not reduce across T or B. Choosing `LayerNorm((T,D))` normalizes the whole trailing T×D grid and has a different affine parameter shape; it changes the model and often fixes sequence length unnecessarily.

[[diagram:feature-normalization]]

### Hand-compute the output

For x=`[1,3]`, μ=2 and v=1. With ε=1e−5, xhat≈`[−.999995000,.999995000]`. With gamma=`[2,1]`, beta=`[0,1]`, output≈`[−1.999990000,1.999995000]`. Ignoring epsilon gives the simplified values `[−2,2]`, useful for intuition but not the exact runtime fixture.

A constant row `[7,7]` has v=0 and xhat=`[0,0]`, so y=beta. Epsilon avoids division by zero; it is not an added constant to the output. Normalized coordinates have mean0; their variance is `v/(v+epsilon)`, slightly below1. Learned gamma/beta further change their mean and variance.

Adding a constant c to all coordinates leaves xhat unchanged. Positive scale invariance is approximate when epsilon>0, because epsilon does not scale with x. These properties hold for the selected axes, not for arbitrary per-coordinate shifts.

### Derive why features interact in backward

Changing xⱼ changes not only its numerator, but also μ and v used by every feature. Let upstream gᵢ=∂L/∂yᵢ and uᵢ=gᵢγᵢ. The chain rule yields

$$\frac{\partial L}{\partial x_i}=
\frac1s\left(u_i-\operatorname{mean}(u)-\hat x_i\operatorname{mean}(u\hat x)\right).$$

The means are over the D features. The formula accounts for epsilon through s and xhat. Gradients for learned affine parameters sum over all batch/position uses:

$$\begin{aligned}
(\nabla\gamma)_i&=\sum_{b,t}g_{bti}\hat x_{bti},\\
(\nabla\beta)_i&=\sum_{b,t}g_{bti}.
\end{aligned}
$$

A manual backward that divides by D−1 or treats μ as a detached constant computes a different gradient.

The input gradients sum to zero over the normalized features, matching shared-shift invariance. A loss that simply sums all normalized outputs with gamma1 is constant, so its zero gradient is expected; use a nonuniform upstream vector for meaningful finite-difference tests.

### A standalone PyTorch comparison

```python
import torch
from torch import nn
x = torch.tensor([[[1.,3.],[7.,7.]]], dtype=torch.float64)
mu = x.mean(-1, keepdim=True)
v = ((x-mu)**2).mean(-1, keepdim=True)
gamma = torch.tensor([2.,1.], dtype=x.dtype)
beta = torch.tensor([0.,1.], dtype=x.dtype)
manual = (x-mu)/torch.sqrt(v+1e-5)*gamma+beta
layer = nn.LayerNorm(2, eps=1e-5).double()
with torch.no_grad():
    layer.weight.copy_(gamma); layer.bias.copy_(beta)
assert torch.allclose(manual, layer(x), atol=1e-12, rtol=1e-12)
assert manual[0,1].tolist() == [0.,1.]
```

### LayerNorm versus BatchNorm

BatchNorm typically estimates feature statistics across batch/spatial examples and maintains running statistics for evaluation. LayerNorm computes its specified per-example trailing-axis statistics in both training and evaluation; it has no default running batch mean/variance. Therefore ordinary LayerNorm's arithmetic does not switch to historical statistics when `eval()` is called. Dropout elsewhere in the same model does change behavior.

Small batches and variable sequence lengths motivate per-token normalization, but an attention mask does not participate in this feature reduction. A padded row can still have a LayerNorm output beta. Mask/gate it if your downstream contract requires zero padding representations. LayerNorm also does not make inputs or output logits categorically normalized probabilities.

The reusable [foundation probes](examples/transformer_foundation_probes.py) implement these small calculations; [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:normalization-transfer]]

### Notes and sources

Calculate both rows, label every reduced axis and derive the two correction terms in ∇x. Compare `LayerNorm(D)` with a whole-sequence normalization and explain train/eval behavior. Reference arithmetic is in [prerequisite_tensors.py](examples/prerequisite_tensors.py).

Sources: [LayerNorm API](source:layernorm), [original Layer Normalization paper](source:layernorm-paper). Continue with [regularization](lessons/21-regularization.md).
