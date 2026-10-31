## Lesson 25 — bidirectional encoder and tied MLM head

**Goal:** trace the full embedding–encoder–vocabulary path, derive the MLM head and its shared gradient paths, and count a complete miniature model. Prerequisites: learned positions, dynamic corruption, post-norm blocks and GELU. The reference classes remain in [roberta_model.py](examples/roberta_model.py); helper/data logic stays outside the model.

### Trace registered modules and their tensor contracts

`RobertaModel` owns `RobertaEmbeddings` and a `ModuleList` of distinct `RobertaEncoderLayer` objects. Each layer has learned `q_proj/k_proj/v_proj/out_proj`, an attention dropout/normalization branch, and a two-map `RobertaFeedForward` branch. Registered ownership makes parameters visible to optimizers, device movement and state serialization. Equal layer specifications do not mean the layers share weights.

For $X[B,T,D]$ and $H$ heads, projected Q/K/V become $[B,H,T,D/H]$. `attention_mask[B,T]` expands to permitted-key shape $[B,1,1,T]`; SDPA Boolean `True` means allowed. There is no target-style causal triangle. The mask excludes keys, not the computation of padded query vectors. The attention call explicitly sets dropout probability to zero when the module is in evaluation mode; a functional SDPA call does not infer that mode itself. [SDPA contract](source:torch-sdpa).

After merging heads and output projection, the actual block computes

$$
\begin{aligned}
U&=\operatorname{LN}_1(X+\operatorname{Dropout}(\operatorname{Attention}(X))),\\
Y&=\operatorname{LN}_2(U+\operatorname{Dropout}(W_2\operatorname{GELU}(W_1U+b_1)+b_2)).
\end{aligned}
$$

These formulas use column-map notation inside the FFN; source PyTorch weights are stored as output-by-input matrices. Both residual tensors preserve $[B,T,D]$. Each LayerNorm reduces the last feature axis independently for each token; it does not average token positions. The original hidden dropout is placed after the FFN output, before residual addition. Substituting pre-norm changes this forward map and checkpoint semantics.

[[diagram:roberta-block]]

### Develop GELU's value and derivative

The default `F.gelu` here uses $g(x)=x\Phi(x)$, where $\Phi$ is the standard normal CDF and $\phi(x)=e^{-x^2/2}/\sqrt{2\pi}$ its density. Product rule gives $g'(x)=\Phi(x)+x\phi(x)$. At $x=1$, $\Phi(1)≈0.841345$, $\phi(1)≈0.241971$, so $g(1)≈0.841345$ and $g'(1)≈1.083315$. At $x=-1$, $g(-1)≈-0.158655$ and $g'(-1)≈-0.083315$.

GELU is a smooth gating transform, not ReLU's exact zeroing of all negative values; its local derivative can be negative. An optional tanh approximation is another numerical specification. The FFN expands $D\to F=4D$ and contracts back, applying the same feature maps independently at every position. Earlier attention mixes contextual information; the FFN itself does not mix positions. [GELU](source:torch-gelu).

### The MLM head is more than a vocabulary matrix

For a contextual row $h\in\mathbb R^D$, the source head applies a $D\to D$ affine map, GELU, feature LayerNorm, then a $D\to V$ vocabulary map:

$$u=\operatorname{LN}(\operatorname{GELU}(hA+a)),\qquad z=uE^\top+c.$$

Here $E[V,D]$ is the **same Parameter object** as the input token table; $c[V]$ is a separate learned vocabulary bias. Two equal-valued tensors are not tying. `RobertaMLMHead` assigns `decoder.weight = word_embeddings.weight`, so optimization accumulates both uses into one table and serialization preserves a compatible mapping.

For illustrative transformed vector $u=[1,1]$ and table $E=[[1,0],[0,2]]$, logits are `[1,2]`; probabilities are `[0.268941,0.731059]`. With target class 1, output-table gradients are $0.268941[1,1]$ for row 0 and $-0.268941[1,1]$ for row 1 before any averaging denominator. This toy starts **after** the head transform; it does not claim a raw hidden vector bypasses GELU/LayerNorm.

More generally the classifier contribution is

$$\frac{\partial L}{\partial E_j}\bigg|_{\mathrm{output}}=\sum_{t\in M}\frac{p_{t,j}-\mathbf1[j=x_t]}{|M|}\,u_t.$$

Add the input-lookup/contextual contribution to obtain the complete tied gradient. Even an unselected vocabulary row can receive output-softmax gradient. In particular `padding_idx=1` suppresses lookup contributions but does not suppress the tied output-classifier contribution or an explicit matrix-decay update. A PAD row initialized to zero is not globally frozen forever.

### Count the complete model under explicit dimensions

Let vocabulary $V=30$, width $D=8$, context capacity $C=8$, one layer, two heads and $F=32$. Include biases and affine LayerNorm parameters:

| Component | Formula | Count |
| --- | --- | --- |
| Token, position tables and embedding LayerNorm | $VD+(C+2)D+2D$ | $240+80+16=336$ |
| One attention module | $4D^2+4D$ | 288 |
| One FFN | $2DF+F+D$ | 552 |
| Two block LayerNorms | $4D$ | 32 |
| Additional tied MLM head parameters | $D^2+D+2D+V$ | $72+16+30=118$ |

Backbone count is $336+288+552+32=1208$; MLM total is 1,326. The tied decoder does not allocate a second $VD$ weight matrix. Heads divide attention width but do not create extra full-width Q/K/V matrices beyond these projections. Changing $V$ or $C$ changes different parameter groups, while changing batch/sequence length affects working tensors rather than learned count.

### Run full forward, loss and backward checks

From `examples/` with the frozen interpreter:

```python
import torch
from roberta_model import RobertaConfig, RobertaForMaskedLM

torch.set_num_threads(1)
torch.manual_seed(10)
c = RobertaConfig(vocab_size=30, embedding_dimension=8,
    num_transformer_blocks=1, num_attention_heads=2, context_length=8)
m = RobertaForMaskedLM(c)
assert sum(p.numel() for p in m.parameters()) == 1326
assert m.mlm_head.decoder.weight is m.roberta.embeddings.word_embeddings.weight
ids = torch.tensor([[0,4,8,2]])
labels = torch.tensor([[-100,6,-100,-100]])
r = m(ids, labels=labels)
assert r['logits'].shape == (1,4,30) and r['target_count'] == 1
r['loss'].backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
assert m.roberta.embeddings.word_embeddings.weight.grad[1].abs().sum() > 0
print('parameters:', sum(p.numel() for p in m.parameters()), 'selected:', r['target_count'])
```

Expect 1,326 parameters and one supervised target, with a nonzero PAD-row tied gradient. This does not mean PAD is a correct label; it arises through the softmax vocabulary denominator. The separate numerical suite compares attention with an explicit float64 score/value computation, checks permitted right context and tests trailing blocked keys. Preserve real position IDs when constructing the latter perturbation.

### Changed-condition practice and notes

[[practice:roberta-encoder-and-mlm-head-practice-1]]

[[practice:roberta-encoder-and-mlm-head-transfer]]

In [my study notes](project/study-notes.md), trace one selected-position gradient through vocabulary bias, tied table, head transform and encoder. Derive the count when the decoder is untied, and explain how that differs from the inspected demonstration and official checkpoint mapping. Next, [MLM pretraining and measurement](lessons/20-mlm-training-and-counting.md) connects this complete model to sampled presentations and counted updates.
