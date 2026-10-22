## Lesson 29 — QA endpoint head, loss and fine-tuning

**Goal:** derive both endpoint losses/gradients, distinguish the current loss's classification domain from constrained decoding, and train the transferred backbone with a new QA objective. Prerequisites: checked backbone transfer, question/context features and original character alignment. The complete `RobertaForQuestionAnswering` implementation stays in [roberta_model.py](examples/roberta_model.py).

### A shared contextual vector produces two endpoint scores

The backbone returns $H[B,T,D]$ for the complete packed question/context window. A single `Linear(D,2)` maps every row to start and end scores:

$$s_{b,t}=h_{b,t}\cdot w_s+b_s,\qquad e_{b,t}=h_{b,t}\cdot w_e+b_e.$$

These are unnormalized **position logits**, not probabilities or character offsets. `unbind(-1)` separates two $[B,T]$ arrays. At $D=32$, the head has $2D+2=66$ scalars, shared across all positions and records.

For rows $h_0=[1,0],h_1=[0,1],h_2=[1,1]$, choose $w_s=[1,2],b_s=0.5$ and $w_e=[-1,1],b_e=-0.5$. Start scores are `[1.5,2.5,3.5]`; end scores are `[-1.5,0.5,-0.5]`. Separate argmax picks start 2 and end 1, a reversed pair. Training marginal classifiers does not itself impose ordering; Lesson 30's decoder imposes joint span constraints.

[[diagram:qa-head]]

### Derive the counted two-endpoint objective

Let supervised window $b$ have inclusive token endpoints $(a_b,z_b)$ and let $N$ count windows with both valid labels. The reference computes summed start/end cross-entropies, averages the two tasks, then divides by $N$:

$$
L=-\frac{1}{2N}\sum_{b\ \mathrm{valid}}\left(\log p^s_{b,a_b}+\log p^e_{b,z_b}\right),\qquad
p^s_{b,t}=\frac{e^{s_{b,t}}}{\sum_{u=0}^{T-1}e^{s_{b,u}}}.
$$

End probabilities normalize analogously. With four equiprobable positions and both correct endpoints assigned 0.25, $L=(-\log0.25-\log0.25)/2=\log4≈1.386294$. `target_count` is one valid window, not four sequence tokens or two endpoints. For unequal microbatch window counts, sum the already half-averaged endpoint losses and divide by their combined count.

The derivatives are $(p^s_{b,t}-\mathbf1[t=a_b])/(2N)$ and $(p^e_{b,t}-\mathbf1[t=z_b])/(2N)$ on valid windows, zero on ignored windows. For one uniform four-position example with start 1/end 2, start derivatives are `[.125,-.375,.125,.125]`; end derivatives are `[.125,.125,-.375,.125]`. The hidden gradient combines both learned columns, $\partial L/\partial h_t=w_s\,\partial L/\partial s_t+w_e\,\partial L/\partial e_t$, then continues into every used backbone layer.

### Name the current endpoint domain accurately

The present loss normalizes across **all packed positions**, including question, special and padded query positions. The attention key mask does not remove their endpoint logits from this denominator. Context-only label construction chooses correct gold endpoints, and loss discourages competing wrong endpoints, but this is different from structurally restricting the endpoint softmax to context.

At uniform scores on a 64-position packed window, baseline loss is $\log64$, even if only 20 positions are context candidates. A context-only-normalized objective would instead have baseline $\log20$ and different gradients. That alternative requires an explicit tested loss change; it is not something the unchanged source already does. Inference uses context-only ordered spans via offsets, as the next lesson explains.

Both targets must be supplied together with shape $[B]$. Missing-window labels are `(-100,-100)`, and the model rejects only-one-ignored endpoint or a valid target that is reversed/out of sequence bounds. An all-ignored batch returns count zero and finite zero loss; `training_step` skips optimizer/scheduler progress. Ignored windows are not negative/no-answer examples.

### Probe the derivatives and an actual backbone backward pass

From `examples/` using the frozen interpreter:

```python
import math
import torch
from torch.nn import functional as F
from roberta_model import RobertaConfig, RobertaForQuestionAnswering

torch.set_num_threads(1)
s = torch.zeros(1,4, dtype=torch.float64, requires_grad=True)
e = torch.zeros(1,4, dtype=torch.float64, requires_grad=True)
loss = (F.cross_entropy(s,torch.tensor([1])) + F.cross_entropy(e,torch.tensor([2]))) / 2
loss.backward()
assert math.isclose(float(loss.detach()), math.log(4))
torch.testing.assert_close(s.grad,torch.tensor([[.125,-.375,.125,.125]],dtype=torch.float64))
torch.testing.assert_close(e.grad,torch.tensor([[.125,.125,-.375,.125]],dtype=torch.float64))

torch.manual_seed(10)
c = RobertaConfig(vocab_size=30, embedding_dimension=8,
    num_transformer_blocks=1, num_attention_heads=2, context_length=8)
model = RobertaForQuestionAnswering(c)
ids = torch.tensor([[0,6,7,2], [0,9,2,1]])
r = model(ids, start_positions=torch.tensor([1,-100]), end_positions=torch.tensor([2,-100]))
r['start_logits'].retain_grad()
r['end_logits'].retain_grad()
r['loss'].backward()
assert r['target_count'] == 1
assert torch.equal(r['start_logits'].grad[1],torch.zeros(4))
assert model.roberta.layers[0].attention.q_proj.weight.grad.abs().sum() > 0
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('uniform endpoint loss:', float(loss.detach()), 'supervised windows:', r['target_count'])
```

The ignored row has no direct endpoint derivative, while the supervised row supplies gradients to the encoder, not only its head. The same parameter objects are shared across windows, so parameter gradients accumulate across valid examples.

### Complete the fine-tuning stage deliberately

The QA branch of `build_training_bundle` builds features for eight authored training questions, keeps answer-containing windows, verifies coverage, optionally copies a compatible MLM backbone and creates fresh AdamW/schedule state. Default fine-tuning trains the entire backbone and head. Freezing the backbone would require `requires_grad`/optimizer-group changes and a separate protocol; calling it “transfer” does not imply automatic freezing.

For a checked artifact initialized in Lesson 27, continue that new QA stage through `course_session.py` and use its full-state resume procedure. For the independent nonpersistent mechanics demonstration, `python roberta_train.py --task both --steps 120` runs MLM, copies its contextual weights, fine-tunes QA and decodes the eight training questions. A falling training loss and correct answers on those **seen** questions demonstrate fitting and decoding; they do not supply held-out SQuAD evidence. [Reference outcomes](project/reference-runs.md).

The source retains separate helpers `qa_batch`, `to_model_device`, `training_step` and `predict_qa`, so a notebook calls the same source rather than hiding model/training definitions in cell state. The GPU path is a declared option whose actual environment/fit/recovery needs measurement; the current evidence is CPU. [QA source](source:roberta-code), [official task preprocessing](source:hf-qa).

### Changed-condition practice and notes

[[practice:qa-head-and-fine-tuning-practice-1]]

[[practice:qa-head-and-fine-tuning-transfer]]

In [my study notes](project/study-notes.md), derive the uniform endpoint gradients, explain what the current denominator contains, and draw gradient flow from both heads into one contextual vector. Predict what freezing the backbone changes and identify the separate artifact checks needed for an official-tokenizer experiment. Next, [constrained decoding and metrics](lessons/23-span-decoding-and-metrics.md) turns those scores into a legal original-text answer.
