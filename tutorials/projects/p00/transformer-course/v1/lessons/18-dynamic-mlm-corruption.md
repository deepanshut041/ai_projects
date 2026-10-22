## Lesson 24 — dynamic MLM corruption and selected-target supervision

**Goal:** derive the selection/replacement distribution, preserve original labels across all branches, and handle zero-target presentations without inventing optimizer progress. Prerequisites: byte-token identity, categorical loss and independent RNG state. “Masked token” is an informal name: the supervised set includes mask, random and unchanged inputs.

### Assign three masks to three different operations

The attention mask says which input keys are real. The eligibility mask identifies ordinary tokens that may be selected, excluding BOS/PAD/EOS/UNK/MASK IDs. The selected-target mask says which positions contribute vocabulary loss. A real unselected token can be an attention key and receive indirect gradients without supplying its own classifier label.

For original IDs $x_t$, define eligibility $e_t$, independent selection draw $u_t$, rate $q$ and selection $m_t=e_t\mathbf1[u_t<q]$. Labels are $x_t$ where $m_t=1$ and `-100` elsewhere. Clone the input before corruption; do not mutate the originals used to construct labels. `-100` belongs only to supervision—it is not a vocabulary ID that can enter an embedding.

### Work the conditional 80/10/10 distribution

For each selected position draw $v_t\sim U[0,1)$:

| Condition | Corrupted input | Original label |
| --- | --- | --- |
| $v_t<0.8$ | MASK ID | $x_t$ |
| $0.8\leq v_t<0.9$ | Random ordinary vocabulary ID | $x_t$ |
| $v_t\geq0.9$ | Unchanged $x_t$ | $x_t$ |

The 80/10/10 fractions are conditional on selection. With $N=1,000$ eligible tokens and $q=0.15$, expected selected count is 150, mask replacements 120, random replacements 15 and unchanged selected inputs 15. These are expectations, not exact batch quotas. Selection count $K$ has mean $Nq$ and variance $Nq(1-q)$ under independent Bernoulli selection. The reference trainer deliberately uses $q=0.35$ for the tiny fixture, giving expected counts 350/280/35/35 on the same hypothetical $N$.

A random replacement can equal the original token. With 256 ordinary tokens, that happens with probability $1/256$ inside the random branch. Thus visibly unchanged selected inputs have probability $0.1+0.1/256$, not exactly 0.1, even though the branch probabilities remain 80/10/10. Counting visually changed cells cannot recover the supervised count.

[BERT's corruption objective](source:bert-paper) motivates these branches; [the RoBERTa study](source:roberta-paper) examines dynamic resampling. The actual course [corrupt_mlm](examples/roberta_data.py) excludes all five specials and samples random replacements only from ordinary IDs. Those explicit choices, its higher tiny-training rate and its own tokenizer distinguish this fixture from an unspecified official collator.

[[diagram:mlm-corruption]]

### Dynamic means redraw on presentation

Store original sequences and invoke the collator each time they are sampled. Dynamic corruption changes selected positions/replacements across presentations, but two draws may coincide. Forcing every new mask to differ introduces a new distribution. The trainer has separate CPU `mask_generator` and `batch_generator` objects; source selection, branch sampling and random replacement consume the mask stream, while example sampling consumes the other stream.

Even positions that do not ultimately become random replacements have generated random-token candidates in this implementation. Restoring only the seed or the selected-label matrix does not restore the next corruption draw. Complete checkpoints save the named generator states and sampling/progress identity. Fixed evaluation uses a separate known generator so validation does not consume the training corruption stream.

### Selected-only logits do not mean selected-only backbone gradients

For selected count $K>0$, the mean loss derivative is

$$\frac{\partial L}{\partial Z_{t,j}}=\frac{m_t}{K}\left(p_{t,j}-\mathbf1[j=x_t]\right).$$

At an unselected output position, this derivative is zero. At a selected position with probabilities `[0.2,0.7,0.1]` and target class 1, its contribution is `[0.2,-0.3,0.1]/K`. The encoder's attention routes its upstream signal into context keys/values and earlier layers; those input positions need not be selected. Restricting every backbone gradient to mask-replaced cells would sever useful context learning.

`masked_token_loss` in [roberta_model.py](examples/roberta_model.py) returns a loss sum and count. The wrapper divides by `max(count,1)` to return finite zero when all labels are ignored. That is a defined zero-target interface, not a zero-valued empirical mean over observations.

### Zero targets are a real outcome, not an error to hide

For $N$ eligible positions, $P(K=0)=(1-q)^N$. At $N=3,q=0.15$, this is $0.85^3=0.614125$. The reference does **not** force at least one selection. A policy that adds one target whenever $K=0$ would change the expected count from 0.45 to $0.45+0.614125=1.064125$ under that particular rule; it is a different objective distribution.

The actual `training_step` samples its complete microbatch group first. If total selected count is zero, it reports `updated=False`, leaves the optimizer/scheduler/global-update counter unchanged, and preserves the consumed RNG progression. The session distinguishes attempts/examples/input tokens from completed updates and supervised targets. A zero-target attempt must not become a fictitious training step or a scheduler tick. Valid-target microbatches in another group use their true common denominator.

### Probe all branches and the zero-target case

From `examples/` using the frozen interpreter:

```python
import torch
from roberta_data import ByteBPETokenizer, corrupt_mlm
from roberta_train import build_training_bundle, training_step

torch.set_num_threads(1)
tok = ByteBPETokenizer()
original = torch.tensor([[0,5,6,7,1,2]])
branches = torch.tensor([[0.,.1,.85,.95,0.,0.]])
corrupted, labels = corrupt_mlm(original, tok, torch.Generator().manual_seed(4),
    probability=1., selection_draws=torch.zeros_like(branches), replacement_draws=branches)
assert original.tolist() == [[0,5,6,7,1,2]]
assert labels.tolist() == [[-100,5,6,7,-100,-100]]
assert corrupted[0,1] == 4 and corrupted[0,3] == 7
assert 5 <= int(corrupted[0,2]) < tok.vocab_size
bundle = build_training_bundle('mlm', total_steps=10)
bundle.protocol['masking_probability'] = 0.0  # Separate diagnostic, not resume identity.
weights = [p.detach().clone() for p in bundle.model.parameters()]
epoch = bundle.scheduler.last_epoch
rng = bundle.mask_generator.get_state().clone()
report = training_step(bundle)
assert not report['updated'] and report['target_count'] == bundle.step == 0
assert bundle.scheduler.last_epoch == epoch
assert all(torch.equal(old,p) for old,p in zip(weights,bundle.model.parameters()))
assert not torch.equal(rng,bundle.mask_generator.get_state())
print('selected labels:', labels.tolist(), 'zero-target update:', report)
```

The labels preserve three targets despite only one explicit MASK in this row. The zero-target diagnostic proves no parameter movement while the corruption stream advances. It does not modify source or a saved run. The tests in [roberta_checks.py](examples/roberta_checks.py) also exercise random replacement counts larger than vocabulary size, excluding an accidental sampling-without-replacement limit.

[[scenario:corruption-branch]]

### Changed-condition practice and notes

[[practice:dynamic-mlm-corruption-practice-1]]

[[practice:dynamic-mlm-corruption-transfer]]

In [my study notes](project/study-notes.md), draw original IDs, eligibility, selection, branch action, labels and attention visibility as separate rows. Calculate expected counts and zero-target probability for a short sentence; predict what restoring only the seed loses. Explain the course's .35 rate and ordinary-only replacement policy without attributing them to every official RoBERTa run. Next, [the encoder and tied head](lessons/19-roberta-encoder-and-mlm-head.md) implement the vocabulary distribution used by those labels.
