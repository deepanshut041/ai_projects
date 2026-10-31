## Lesson 3 — Model tensor interfaces and resource budgets

**Goal:** trace the complete architecture's axes and distinguish learned parameters from runtime work/memory. Prerequisites: Tutorial 0 tensor/matrix contracts and this course's source layout. Use unequal dimensions while checking interfaces; the equal lengths in a tiny fixture can hide errors.

### A shape is a typed coordinate contract

Let B=batch, S=source positions, T=decoder positions, D=model features, H=heads, d=D/H, F=FFN hidden width and Vₛ/Vₜ=separate vocabulary sizes. Translation IDs are integer `[B,S]` and `[B,T]`; hidden states are floating `[B,S,D]` and `[B,T,D]`. Cross-attention scores are `[B,H,T,S]`. Vocabulary logits are `[B,T,Vt]`. QA endpoint logits are `[B,L]`, where L is its packed length, not a vocabulary size.

[[table:tensor-contracts]]

For B=2,S=3,T=4,D=8,H=2, query heads have shape `[2,2,4,4]`, source keys/values `[2,2,3,4]`, and cross scores `[2,2,4,3]`. Dot products contract the last feature axis; softmax normalizes the last key-position axis; weighted sums return one value vector per query/head. Reshape and transpose are distinct, as developed in Tutorial 0.

An affine layer mathematically uses X[N,D]W[D,F]+b[F], while PyTorch stores its weight `[F,D]`. Bias is shared across positions/examples. The indexed forward and backward contracts are

$$\begin{aligned}
Y_{nf}&=\sum_iX_{ni}W_{if}+b_f,\\
\nabla W&=X^TG,\quad \nabla b=\sum_nG_{n,:},\quad\nabla X=GW^T.
\end{aligned}$$

These rules also determine projection and FFN parameter counts. For sequence tensors, the shared bias gradient sums batch and position uses; it does not average automatically.

[[diagram:matrix-product]]

### Derive the actual reference's parameter budget

For D=32,F=128, each biased attention module has `4D²+4D=4224` entries, and each biased FFN has `2DF+F+D=8352`. A LayerNorm has2D=64 parameters. One encoder has1 attention,1FFN,2norms:12704. One decoder has2attentions,1FFN,3norms:16992.

Separate source/target tables at Vₛ=10,Vₜ=12 contain `(10+12)×32=704` entries. The separate target output head adds `32×12+12=396`. Therefore

$$\begin{aligned}
P&=12704+16992+704+396\\
 &=30796.
\end{aligned}$$

Fixed sinusoidal buffers are state-dict tensors, not learned parameters. Tying the target output matrix would remove384 independent entries, but the actual reference leaves it separate. Do not subtract parameters for a tying feature that has not been implemented.

### Parameters do not count activations or total process memory

In FP32,30796 parameters occupy123184 bytes. Gradients add storage, Adam's two moment arrays add about twice the parameter elements, and optimizer/scalar/runtime overhead adds more. Activations/saved backward values depend on B,S,T and depth. Whole-process RAM includes Python/PyTorch and allocation behavior, not just tensor payloads.

With the unequal miniature dimensions above, explicit encoder/decoder-self/cross score arrays contain

$$BH(S^2+T^2+TS)=2\cdot2(9+16+12)=148$$

scalars,592FP32 bytes for those arrays alone. Softmax weights, gradients and saved values may coexist. Optimized SDPA can avoid full score materialization; do not count an explicit array as a guaranteed allocation in every kernel.

At fixed D,H, doubling sequence length approximately quadruples self-attention score elements. FFN work grows linearly with length. Projection work is order sequence×D²; attention score/value work is order TSD; FFN work is order sequence×DF. Operation order is not a measured runtime or a promise that the attention component dominates every short sequence.

### Verify interfaces and counts against the source

```python
import torch
from translation_model import Transformer, TransformerConfig
m = Transformer(TransformerConfig())
assert sum(p.numel() for p in m.parameters()) == 30796
buffers = dict(m.named_buffers())
assert any('encodings' in name for name in buffers)
batch, source_len, target_len = 2,3,4
src = torch.tensor([[2,4,3],[2,5,3]])
tgt = torch.tensor([[2,4,5,3],[2,6,7,3]])
logits = m(src,tgt)
assert logits.shape == (batch,target_len,12)
```

Run from `examples/` with the Lesson 2 interpreter. This is a contract probe, not a trained-quality test. [foundation_math.py](examples/foundation_math.py) has independent matrix/backward calculations, including different rectangular dimensions.

[[practice:tensors-practice-1]]
[[practice:tensors-transfer]]

### Notes and checks

In my notes: label every tensor in the unequal-length path, derive30796 from component terms and calculate parameter/score bytes separately. Predict changing H at fixed D and changing T at fixed model parameters. Identify a same-shape wrong-axis operation that a naive shape assertion would miss. Continue with [translation likelihood](lessons/04-learning.md).
