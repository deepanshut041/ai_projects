## Lesson 22 — encoder–decoder learning and inference

**Goal:** construct the source stream, shifted decoder stream, supervision and masks for conditional sequence learning; explain why parallel teacher-forced training and sequential generation implement the same factorization. Prerequisites: tokenization, embeddings, positions, attention masks and token loss. This is a sequence contract study; the complete translation implementation belongs to Course 1.

### Define the conditional probability before the architecture

Let $x=(x_1,\ldots,x_S)$ be source token IDs and $y=(y_1,\ldots,y_M)$ the target content. Append an end token $y_{M+1}=\mathrm{EOS}$. A conditional autoregressive model represents

$$
p_\theta(y,\mathrm{EOS}\mid x)
=\prod_{t=1}^{M+1}p_\theta(y_t\mid x,y_{<t}).
$$

This follows the probability chain rule; it does not assume the target tokens are independent. The model uses the complete source and only earlier target tokens for each factor. The negative log-likelihood sums $-\log p_\theta(y_t\mid x,y_{<t})$. EOS contributes a factor because stopping is a learned prediction. BOS supplies an initial decoder input; in this contract it is not an output target. [Encoder–decoder framework](source:d2l-seq2seq).

If the three factors for content `[A,B]` followed by EOS are $[0.8,0.5,0.9]$, their product is 0.36 and sequence NLL is $-\log0.36\approx1.021651$. The per-target mean is $1.021651/3\approx0.340550$, counting EOS. Omitting EOS changes both the sequence score and the denominator. A loss averaged over sequences gives a different weighting from one averaged over all valid tokens when lengths differ.

### Two streams, three attention roles

Define batch size $B$, padded source length $S$, padded decoder length $T$, model width $D$, head count $H$ and target vocabulary size $V_t$. Source IDs have shape $[B,S]$; the encoder turns them into contextual memory $E\in\mathbb R^{B\times S\times D}$. Decoder IDs have shape $[B,T]$; the decoder emits hidden vectors $Z\in\mathbb R^{B\times T\times D}$ and logits $[B,T,V_t]$.

| Operation | Query stream | Key/value stream | Required permission rule |
| --- | --- | --- | --- |
| Encoder self-attention | Source states | Source states | Valid source keys; no target-style causal triangle |
| Decoder self-attention | Target-prefix states | Target-prefix states | Valid decoder keys at indices no greater than the query index |
| Decoder cross-attention | Target-prefix states | Encoder memory | Valid source keys; source and target lengths may differ |

For cross-attention, scores have shape $[B,H,T,S]$. Applying a $T\times T$ triangle to that tensor has the wrong meaning even if equal lengths conceal the error. The decoder's queries decide which source information to retrieve; its keys and values are projected from encoder memory. The original Transformer uses those three attention roles in separate sublayers. [Architecture](source:transformer-paper).

[[diagram:seq2seq-contract]]

### Shift inputs and labels exactly once

Use a pedagogical target vocabulary with `PAD=0`, `BOS=1`, `EOS=2`, and content IDs starting at 3. These IDs are local to this probe; they do not replace the byte-tokenizer contract from Lesson 12. For target content `[4,5,6]`:

| Decoder index | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Decoder input | BOS | 4 | 5 | 6 |
| Label predicted at that index | 4 | 5 | 6 | EOS |
| Legal target information | Empty prefix | 4 | 4,5 | 4,5,6 |

A shorter target `[7]` in the same batch becomes decoder input `[BOS,7,PAD,PAD]` and labels `[7,EOS,PAD,PAD]`. The first two positions are supervised. Key-padding exclusion and ignored labels solve different problems: one restricts information flow, the other removes unused positions from the loss. Padded query rows may still have legal keys and produce nonzero outputs. Exclude their labels and apply any declared zero-output policy explicitly.

The following complete probe constructs both examples and checks target alignment, masks and loss accounting. Run it with the frozen interpreter from the course `examples/` directory.

```python
import math
import torch
from torch.nn import functional as F

PAD, BOS, EOS = 0, 1, 2
targets = [[4, 5, 6], [7]]
B, T, V = 2, max(len(row) + 1 for row in targets), 8
decoder = torch.full((B, T), PAD, dtype=torch.long)
labels = torch.full((B, T), PAD, dtype=torch.long)
for b, content in enumerate(targets):
    n = len(content) + 1
    decoder[b, :n] = torch.tensor([BOS] + content)
    labels[b, :n] = torch.tensor(content + [EOS])

source = torch.tensor([[3, 4, 0], [5, 6, 7]])
source_valid = source != PAD
causal = torch.ones(T, T, dtype=torch.bool).tril()
allowed_self = causal[None, None] & (decoder != PAD)[:, None, None, :]
allowed_cross = source_valid[:, None, None, :].expand(B, 1, T, 3)
assert allowed_self.shape == (2, 1, 4, 4)
assert allowed_cross.shape == (2, 1, 4, 3)
assert allowed_self[0, 0, 1].tolist() == [True, True, False, False]
assert allowed_cross[0, 0, 3].tolist() == [True, True, False]
assert labels.tolist() == [[4, 5, 6, 2], [7, 2, 0, 0]]

logits = torch.zeros(B, T, V, requires_grad=True)
loss_sum = F.cross_entropy(
    logits.reshape(-1, V), labels.reshape(-1),
    ignore_index=PAD, reduction='sum')
count = int((labels != PAD).sum())
assert count == 6
assert math.isclose(float(loss_sum.detach()) / count, math.log(V), rel_tol=1e-6)
(loss_sum / count).backward()
assert torch.equal(logits.grad[1, 2:], torch.zeros(2, V))
print('decoder:', decoder.tolist())
print('valid targets:', count, 'uniform mean NLL:', float(loss_sum.detach()) / count)
```

Here `True` explicitly means **allowed**. Translate it when an API uses `True` to mean blocked. The uniform baseline is $\log8\approx2.079442$ per valid target; flattening preserves the matching $B,T$ order. This probe verifies preparation and the loss denominator; it does not execute a trained decoder or prove its causality.

### Teacher forcing permits parallel computation without future leakage

During teacher forcing, the decoder receives the recorded previous target IDs. All shifted inputs are available at once, so masked attention computes the factors in parallel. The causal mask prevents a query from reading later shifted positions. At query index 1, the current input token is the first target token; that row predicts the second target token. It never receives its own label through a legal path.

Using unshifted labels as decoder inputs exposes the answer at the same position through the embedding and residual path. A causal mask including the diagonal cannot fix that. Correct shifting without a causal mask also leaks later labels via later positions. Test both contracts in the full model: with dropout off, replace a future decoder token and require earlier logits to stay unchanged; separately change the source and require output behavior to reflect source conditioning. Shape correctness alone proves neither property.

### Generation constructs the prefix instead of receiving it

At inference, encode the source, initialize the decoder prefix to BOS, calculate next-token logits at its final valid position, select a token and append it. Stop on EOS or a declared maximum generated length. Batch decoding needs one finished flag per record: a finished record must not continue generating meaningful tokens while others run. Reset prefixes and any caches for each new request. Caches are an implementation optimization with their own mask/position contract; they are not needed to understand the factorization.

Greedy decoding chooses the highest-probability token at each step. It need not find the most probable complete sequence. Suppose first-token probabilities are $p(A)=0.6,p(B)=0.4$, while $p(\mathrm{EOS}\mid A)=0.51$ and $p(\mathrm{EOS}\mid B)=0.9$. Greedy chooses `[A,EOS]` with probability $0.306$; `[B,EOS]` has probability $0.36$. A width-two beam retains both initial hypotheses and can recover the latter in this short tree. Keep scores as sums of log probabilities; do not multiply many floating-point probabilities directly. [Beam search](source:d2l-beam-search).

Beam search retains a bounded set of candidate prefixes and expands them; finite width remains approximate. EOS-completed hypotheses need separate handling. A length penalty changes the ranking objective, so report its rule and the length being counted. Stopping at the first EOS among all candidates can discard a better completion. Use a defined completion/bound rule and a hard length cap rather than an unbounded loop. Probability score is also not a translation-quality metric: a decoder and an evaluation metric answer different questions.

### Exercise and notes

[[practice:seq2seq-transfer]]

In [my study notes](project/study-notes.md), shift the target `[9,8]`, count its supervised positions and draw its legal decoder self-attention matrix. Annotate one cross-attention row when the source has five real tokens plus two PAD positions. Explain why teacher-forced token accuracy and free-running sequence accuracy can diverge: generated mistakes change the later prefixes the model sees.

Continue with [evaluation and debugging](lessons/09-training.md), applying the same discipline to a complete small training experiment before a full translation build.
