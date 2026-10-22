## Lesson 15 — Queries, keys, values and scaled attention

**Goal:** calculate attention as a differentiable retrieval operation, with explicit axes and backward paths. Prerequisites: dot products, probabilities, embeddings and variance scaling. Self-attention and cross-attention differ in where their inputs come from, not in the basic weighted-sum formula.

### Separate matching from transported information

A query describes what one output position is looking for. A key provides a matching representation for a candidate input position. A value is the information that candidate contributes. Learned maps create Q=XqWq, K=XkWk and V=XvWv. These descriptions explain roles; they do not imply trained coordinates have human-readable meanings.

For one head, let Q `[T,dₖ]`, K `[S,dₖ]` and V `[S,dᵥ]`. T counts queries, S counts key/value positions. Then

$$\begin{aligned}
R&=QK^T/\sqrt{d_k}\quad[T,S],\\
A_{ts}&=\frac{\exp R_{ts}}{\sum_{j=0}^{S-1}\exp R_{tj}},\\
O&=AV\quad[T,d_v].
\end{aligned}$$

Each row of A is a distribution over **source positions**. It is not a distribution over vocabulary classes. The softmax axis is S. Dot products sum feature coordinates; the subsequent weighted value sum aggregates positions. Conflating those two sums causes incorrect shapes and interpretations.

[[diagram:attention-flow]]

### A rectangular hand calculation

Choose one query q=`[1,0]`, dₖ=2 and three keys:

$$K=\begin{bmatrix}0&0\\\sqrt2\log2&0\\\sqrt2\log3&0\end{bmatrix},\quad
V=\begin{bmatrix}6&0\\0&3\\2&2\end{bmatrix}.$$

Raw dot products are `[0,sqrt2 log2,sqrt2 log3]`; scaled scores are `[0,log2,log3]`. Exponentials are `[1,2,3]`, so weights are `[1/6,2/6,3/6]`. The output is

$$\begin{aligned}
O_0&=(1/6)6+(2/6)0+(3/6)2=2,\\
O_1&=(1/6)0+(2/6)3+(3/6)2=2.
\end{aligned}$$

Changing only the third value to `[4,0]` leaves scores and weights unchanged and gives `[3,1]`. Changing its key can change every normalized weight in the row. Removing a candidate requires renormalization over allowed keys; simply zeroing its value leaves probability mass assigned to it and gives a different operation.

The sqrt(dₖ) factor is about score variance; it is not an average over keys or values. dᵥ can differ from dₖ in the mathematical interface. In the common multi-head architecture both are chosen as D/H.

### All the attention operations are differentiable

Let upstream G=∂L/∂O `[T,dᵥ]`. Ignoring masks for this derivation,

$$\begin{aligned}
\nabla V&=A^TG,\qquad U=\nabla A=GV^T,\\
\nabla R_{ts}&=A_{ts}\left(U_{ts}-\sum_jA_{tj}U_{tj}\right),\\
\nabla Q&=(\nabla R)K/\sqrt{d_k},\\
\nabla K&=(\nabla R)^TQ/\sqrt{d_k}.
\end{aligned}$$

The subtraction term couples scores within a softmax row. Increasing one score changes the distribution across all candidates, not only that candidate. Gradients then continue into the projection weights using affine backward rules. If an implementation detaches A, it still transports values but loses the score-learning path.

### A runnable explicit calculation

```python
import math, torch
q = torch.tensor([[1.,0.]], dtype=torch.float64)
k = torch.tensor([[0.,0.],[math.sqrt(2)*math.log(2),0.],
                  [math.sqrt(2)*math.log(3),0.]], dtype=q.dtype)
v = torch.tensor([[6.,0.],[0.,3.],[2.,2.]], dtype=q.dtype)
scores = q @ k.T / math.sqrt(q.shape[-1])
a = scores.softmax(dim=-1)
out = a @ v
assert torch.allclose(a, torch.tensor([[1/6,2/6,3/6]], dtype=q.dtype))
assert torch.allclose(out, torch.tensor([[2.,2.]], dtype=q.dtype))
```

This is a numerical probe, not a complete trained language model. Stable mask handling is added in the next lesson. Optimized SDPA may avoid materializing the full score matrix, but its intended mathematical operation still follows this interface; compare explicit and optimized paths with tolerances.

Passing this lesson’s Q/K/V into `explicit_attention` in the [foundation probes](examples/transformer_foundation_probes.py) reproduces `[2,2]`. The module’s CLI summary uses an **additional independent fixture**, with one-feature Q/K and different values `[[0,0],[6,0],[4,2]]`, whose unmasked output is `[4,1]`; it is not the same hand example. The [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:attention-transfer]]

### Notes and sources

Write all shapes, compute the changed-value output and derive the row-softmax backward rule. Explain which input supplies Q versus K/V in cross-attention. A large attention weight is a model computation, not by itself a causal explanation of a prediction.

Sources: [attention equation](source:transformer-paper), [PyTorch SDPA](source:torch-sdpa), [D2L attention scoring](source:d2l-attention). Continue with [masking](lessons/16-masks.md).
