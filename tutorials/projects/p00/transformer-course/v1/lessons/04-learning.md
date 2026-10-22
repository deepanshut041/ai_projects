## Lesson 4 — Translation likelihood and supervised losses

**Goal:** derive the actual sequence objective and its gradients before connecting them to a trainer. Prerequisites: probability/logits/chain rule from Tutorial 0 and the model tensor interfaces. Loss choice and reduction are part of the specification, not incidental arguments to a library call.

### Autoregressive likelihood does not assume independent tokens

For source x and target y₁…yₜ including EOS,

$$\begin{aligned}
p_\theta(y\mid x)&=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}),\\
-\log p_\theta(y\mid x)&=\sum_{t=1}^{T}-\log p_\theta(y_t\mid x,y_{<t}).
\end{aligned}$$

The conditional factors include the prefix, so target tokens are not declared independent. During teacher forcing, decoder input is `[BOS,y1,...,y(T−1)]` and labels `[y1,...,EOS]`. A causal mask and that one-token shift enforce the factorization while permitting parallel training. A decoder that reads its own unshifted target can minimize an easier leaked problem.

Suppose a target has probabilities `.5,.25,.8`. Its sequence NLL is log10≈2.302585; its mean over three labels is≈.767528. A second target has one label with probability `.5`, loss log2≈.693147. A combined token mean is `(log10+log2)/4≈.748933`; averaging the two record means gives≈.730338. Both are mathematically defined, but weight records differently. The reference chooses an unweighted valid-token mean.

### Stable cross-entropy and its exact gradient

For vocabulary logits z and hard target k,

$$\begin{aligned}
\ell(z,k)&=\operatorname{LSE}(z)-z_k,\\
\operatorname{LSE}(z)&=m+\log\sum_j e^{z_j-m},\qquad m=\max_jz_j,\\
\frac{\partial\ell}{\partial z_j}&=\operatorname{softmax}(z)_j-\mathbf1[j=k].
\end{aligned}$$

With logits `[0,log3]` and target1, loss log(4/3)≈.287682 and gradient `[.25,−.25]`. Adding1000 to both logits leaves the loss/distribution unchanged. Passing probabilities into `cross_entropy` is a different operation: it interprets them as logits and applies another log-softmax.

For valid labels across a batch, total L=Σℓ/K gives each logit row gradient `(p−one_hot)/K`. Ignored rows have zero loss contribution and zero logit gradient. This is exclusion from supervision, not exclusion of a token's representation from attention.

[[diagram:learning-update]]

### Trace the source implementation

In [translation_train.py](examples/translation_train.py), `token_statistics(logits,labels)` flattens `[B,T,V]` to `[B*T,V]`, flattens labels in the same order, computes sum CE with ignore_index−100 and returns `(loss_sum,count,correct)`. It rejects zero valid targets rather than dividing by zero. Training normalizes by the total valid count across all micro-batches; evaluation accumulates totals before division.

```python
import math, torch
from translation_train import token_statistics
z = torch.tensor([[[0.,math.log(3)],[1000.,1000.]]],
                 dtype=torch.float64, requires_grad=True)
y = torch.tensor([[1,-100]])
loss_sum, count, correct = token_statistics(z,y)
(loss_sum/count).backward()
assert count.item() == 1 and correct.item() == 1
assert abs(loss_sum.item()-math.log(4/3)) < 1e-12
assert torch.allclose(z.grad[0,0], torch.tensor([.25,-.25], dtype=z.dtype))
assert torch.equal(z.grad[0,1], torch.zeros(2,dtype=z.dtype))
```

Run from `examples/` with the recorded Torch interpreter. The padded logit row can be arbitrarily wrong and still have zero supervised gradient; this does not prove the model hides padded keys or future tokens. Those require separate perturbation tests.

### Smoothing is an explicit alternative target

A uniform-mixture smoothing convention uses q=(1−ε)one_hot(k)+ε/V. Its CE gradient is p−q. For V=4,ε=.1,k=2, q=`[.025,.025,.925,.025]`. Excluding the gold class or PAD from the smoothing distribution gives another formula and must be declared. The original Transformer reports smoothing; the inspected translation reference here uses plain hard-label CE. Do not silently reinterpret its training loss as a smoothed objective.

Training-smoothed CE is not the same quantity as held-out hard-target NLL used for perplexity. Perplexity is exp(mean token NLL) under a fixed tokenizer/corpus/conditioning contract. A lower training mean caused by changing the valid count or evaluation corruption is not evidence of a better model.

### Connect output gradients to learned state

With vocabulary head z=hU+c, ∇U=hᵀδ, ∇c sums δ across uses, and ∇h=δUᵀ. That signal then traverses decoder FFN/self/cross-attention and encoder memory. A nonzero head gradient alone does not prove source embeddings or encoder projections receive a gradient; inspect those paths separately in the complete-model lesson.

[[practice:learning-practice-1]]
[[practice:learning-transfer]]

### Notes and checks

In my notes: derive the conditional factorization, p−one_hot and the valid-token denominator. Work the two different record/token means, then change one target length and predict weighting. Draw which tensor rows are ignored and which learned modules must receive gradients. Continue with [source/target tokenizers](lessons/05-tokenization.md). API references: [CE](source:torch-ce) and [the original paper](source:attention-paper).
