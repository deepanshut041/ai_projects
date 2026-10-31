## Lesson 19 — Residual connections, depth and normalization order

**Goal:** derive the identity path and distinguish the original post-norm layer from a later pre-norm design. Prerequisites: tensor maps, Jacobians, attention and FFNs. A residual connection is an addition with a shape contract, not a special kind of learned weight.

### Learn a change to an existing representation

For sublayer F, a residual map is y=x+F(x). The input and output of F must have the same shape, here `[B,T,D]`. If the sublayer uses an internal different width, its final projection must restore D. Expanding features to F inside an FFN is compatible because the second affine map returns to D.

The identity branch carries x directly. The sublayer learns a change relative to that representation. For x=`[1,3]` and F(x)=`[2,−1]`, y=`[3,2]`. Concatenating x and F(x) would give four features, not a residual sum; broadcasting a width1 output would be legal arithmetic but a different architectural choice.

For row-vector gradients written in column form, differentiating gives

$$J_y=I+J_F,\qquad \nabla_xL=g+J_F^Tg,$$

where g is the upstream gradient of L with respect to y. The direct contribution g avoids requiring every gradient path to multiply through J_F. Across several residual blocks, there are many routes through identity and learned branches. This explains a possible benefit for optimization, not a guarantee that all gradients remain healthy: J_F can amplify or cancel. If F(x)=−x, y=0 and the derivative is0. Residual connections do not eliminate initialization, normalization or optimizer considerations.

[[diagram:residual-path]]

### Normalization order changes the function

The original architecture uses post-norm:

$$y=\operatorname{LN}(x+\operatorname{Dropout}(F(x))).$$

A common later pre-norm variant is

$$y=x+\operatorname{Dropout}(F(\operatorname{LN}(x))).$$

LayerNorm is explained fully in the next lesson. In post-norm, even the direct branch then passes through normalization's derivative. In pre-norm, the outer residual path directly adds x. Final stack normalization and other conventions also vary across architectures; “pre-norm” alone does not specify a whole model.

Set F=0 and dropout off. Post-norm produces LN(x), while pre-norm produces x. For x=`[1,3]`, standard LayerNorm with gamma1,beta0 gives approximately `[−1,1]`; the pre-norm result remains `[1,3]`. Thus changing normalization order changes forward outputs even before training, not just the learning rate or implementation efficiency.

```python
import torch
from torch import nn
x = torch.tensor([[1.,3.]], dtype=torch.float64, requires_grad=True)
ln = nn.LayerNorm(2, eps=1e-5).double()
post = ln(x + torch.zeros_like(x))
pre = x + torch.zeros_like(ln(x))
assert torch.equal(pre, x)
assert not torch.allclose(post, x)
(pre * torch.tensor([[2.,4.]])).sum().backward()
assert x.grad.tolist() == [[2.,4.]]
```

This controlled zero-sublayer check distinguishes the contracts. It does not benchmark the two architectures' training stability. Do not reproduce the original paper while silently using pre-norm because another tutorial's code happens to do so.

### Residual dropout has a specific location

Dropping the sublayer output before the addition makes training y=x+M⊙F(x)/(1−p), followed by LN in the original order. Dropping the whole sum also removes part of the identity signal and defines another operation. The original embedding-plus-position input also uses dropout before entering the stack. Attention-weight dropout is another distinct placement; its row sums after random dropping need not remain1 even with inverted scaling.

Training/evaluation controls must consistently disable stochastic dropout during a numerical comparison. A causality test with active dropout can show random differences unrelated to forbidden information. Check modes before drawing an architectural conclusion.

### Layer and stream boundaries

An encoder layer uses residual attention then residual FFN. A decoder layer inserts a separate cross-attention residual sublayer between causal self-attention and FFN. Encoder memory supplies K/V, but the residual added around cross-attention is the decoder query-stream representation, shaped `[B,T,D]`. Adding source memory `[B,S,D]` instead is wrong even if a toy example happens to have T=S.

Do not reuse one normalization module across several sublayers accidentally: each owns learned gamma/beta in the standard design. Module registration and state inspection expose unintended sharing. Sequence padding may need output gating for declared invariants; a key mask alone does not zero a residual query.

The reusable [foundation probes](examples/transformer_foundation_probes.py) implement these small calculations; [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:residual-transfer]]

### Notes and sources

Draw both branches, derive I+J_F and produce a cancellation counterexample. Work the zero-sublayer pre/post-norm case and name the residual stream in cross-attention. Sources: [original architecture](source:transformer-paper), [residual-network paper](source:residual-paper), [pre/post-norm analysis](source:prenorm-paper). Continue with [LayerNorm](lessons/20-normalization.md).
