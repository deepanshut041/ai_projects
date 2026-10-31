## Lesson 31 — integrated debugging and profiling

**Goal:** diagnose translation, MLM and QA with task-specific invariant probes, inspect gradients/activations and measure an explicit resource profile. Prerequisites: the complete models, losses, data contracts and decoding rules. A falling scalar loss does not locate a bug; a reproducible changed-condition test can isolate the operation that failed.

### Trace the first incorrect boundary

Start with a tiny record whose IDs, positions, labels and expected legal information you can inspect. Check the input contract before investigating optimizer scale: a wrong label shift, answer offset or corruption target can produce apparently successful fitting of the wrong task. Log shapes **and axis meaning**, relevant IDs, dtype/device, source/config/tokenizer identity and counts.

For translation, inspect source/target tokenization, shifted decoder input/labels, source padding, target causality, cross-attention, vocabulary loss and autoregressive stopping. For MLM, inspect original IDs, eligibility, selected labels, corruption branches, bidirectional visibility and selected counts. For QA, inspect original answer characters, window ownership, token endpoints, full-packed loss and constrained original-text extraction. The same attention kernel appears in all paths; those task boundaries do not become interchangeable.

[[diagram:model-diagnosis]]

### Make hypotheses falsifiable

| Observation | Candidate boundary | Distinguishing small probe |
| --- | --- | --- |
| Translation training is suspiciously easy | Unshifted labels or future visibility | Change a future decoder token with dropout off; earlier logits must stay unchanged |
| Generated translations ignore the source | Cross-attention/source use or memorized targets | Compare aligned versus changed/shuffled sources on controlled held-out inputs; retain generation and NLL |
| MLM loss counts do not match visible MASKs | Selected-target versus corruption branch confusion | Supply controlled 80/10/10 draws; count labels including unchanged/random targets |
| MLM earlier logits change with right context | Expected bidirectional visibility | Compare with the intended encoder contract; right context is allowed here |
| Real outputs change after PAD perturbation | Key masking or shifted learned positions | Hold mask **and real position addresses** fixed; mutate only a trailing blocked region first |
| QA training loss is low but answer is wrong | Alignment, domain, ordering, length or window choice | Extract gold from offsets, enumerate legal pairs and check cross-window scores separately |
| Resume diverges after reload | Optimizer/schedule/generator/identity state | Compare the same next batch/corruption/rate/update in another process |

A source perturbation can prove a computational dependence while still failing to show that a trained model uses source information helpfully. Assess the change against the task's expected behavior. A future-target perturbation proves a causality invariant only for tested inputs/settings; broad component checks and independent numerical comparisons strengthen that evidence.

### Distinguish mode, graph and parameter trainability

`model.eval()` changes dropout behavior; `torch.no_grad()` disables graph recording; `requires_grad=False` changes whether a parameter participates in autograd. They answer different questions. A training-mode model inside `no_grad()` can still apply random dropout. An evaluation-mode model can still backpropagate if graph recording and parameter gradients are enabled.

RoBERTa SDPA receives an explicit dropout probability based on `self.training`. `fixed_mlm_evaluation`, `fill_mask` and `predict_qa` leave the model in evaluation mode; `training_step` switches it back to training. The translation evaluation helper restores its previous mode. Check the actual helper rather than assuming one universal convention. The checkpoint stores model mode, and the trainer's next step must select its intended mode explicitly.

The MLM/QA update currently has no explicit finite-loss or finite-gradient rejection before stepping, unlike the translation update. Gradient clipping limits a finite vector's norm; it does not repair NaNs. Inspect loss, activations and gradients **before** changing parameters. If a learner adds a fail-closed finite guard, test its counter/schedule/persistence behavior as part of that implementation; do not describe it as already present in the unchanged reference.

### Use gradients and activations to localize numerical problems

Record per-layer activation mean/variance, finite fraction and relevant gradient norms. A single global norm can conceal an inactive layer or an exploding subbranch. For squared norm $\lVert g\rVert_2=(\sum_i g_i^2)^{1/2}$, clipping with limit $c$ scales by $\min(1,c/\lVert g\rVert_2)$, preserving direction when finite. With gradient `[3,4]` and limit 2.5, the result is `[1.5,2]`; its reported preclip norm remains 5.

For a residual branch, gradients add through the identity and learned paths; their existence does not guarantee good conditioning. For selected MLM logits, unselected direct output derivatives are zero while context/embedding gradients can remain nonzero. For QA, both endpoint tasks contribute to the shared contextual row. These mechanisms tell you which zeros are expected before you label them a disconnected model.

Overfit a small consistent fixture with dropout disabled and a fixed target construction before expanding data or capacity. For MLM that can mean one fixed-corruption batch; redrawing masks changes the target experiment. For QA use one fully covered answer span. For translation compare teacher-forced loss with actual free-running outputs. Successful tiny fitting tests machinery; it does not establish held-out quality or official benchmark reproduction.

### Profile defined tensors and actual execution separately

For $B=1,T=5,D=8,H=2,V=30$, FP32 hidden storage is $BTD×4=160$ bytes. One explicit attention-score array would contain $BHT^2=50$ scalars, or 200 bytes. Vocabulary logits need $BTV×4=600$ bytes. Those are tensor-size calculations, not whole-model peak RAM; SDPA may avoid materializing the full score array.

Parameter storage counts unique tensors, so tied output weights are not double-counted. Adam moments, gradients, saved activations, allocator/runtime overhead and checkpoints add different costs. Doubling $T$ quadruples explicit self-attention score storage while leaving parameter count fixed. For a CPU timing, state threads, batch/context, dtype, warmup and whether forward-only or backward/update was measured. GPU timing additionally requires synchronization and its own peak-memory accounting; CPU numbers cannot be relabeled GPU measurements.

### Run a clean invariant/gradient/profile probe

From `examples/` with the frozen interpreter:

```python
import time
import torch
from roberta_model import RobertaConfig, RobertaForMaskedLM

torch.set_num_threads(1)
torch.manual_seed(10)
c = RobertaConfig(vocab_size=30, embedding_dimension=8,
    num_transformer_blocks=1, num_attention_heads=2, context_length=8,
    hidden_dropout_p=.2, attention_dropout_p=.2)
m = RobertaForMaskedLM(c).eval()
ids = torch.tensor([[0,4,8,2,1]])
mask = ids.ne(1)
changed = ids.clone(); changed[0,-1] = 20
stats = {}
def inspect_layer(module, inputs, output):
    stats['mean'] = float(output.detach().mean())
    stats['variance'] = float(output.detach().var(unbiased=False))
hook = m.roberta.layers[0].register_forward_hook(inspect_layer)
with torch.no_grad():
    original = m(ids,mask)['logits']
    alternate = m(changed,mask)['logits']
    torch.testing.assert_close(original[:,:4],alternate[:,:4])
    for _ in range(3): m(ids,mask)
    started = time.perf_counter()
    for _ in range(25): m(ids,mask)
    milliseconds = 1000*(time.perf_counter()-started)/25
r = m(ids,mask,labels=torch.tensor([[-100,6,-100,-100,-100]]))
r['loss'].backward()  # eval mode still permits gradients.
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
hook.remove()
print('CPU eval-forward milliseconds:',milliseconds,'layer stats:',stats)
print('hidden bytes:',1*5*8*4,'explicit score bytes:',1*2*5*5*4)
```

The timing is a local measured observation, not a required speed threshold. The trailing perturbation preserves all earlier real position addresses and excludes the changed key, isolating the intended invariant. Do not copy this assertion to an interior-padding mutation without accounting for the position rule.

### Changed-condition practice and notes

[[practice:diagnosis-transfer]]

[[practice:diagnosis-contract-case]]

In [my study notes](project/study-notes.md), retain one concrete observation, competing hypotheses and the probe that separated them. Calculate a tensor budget independently from process memory, and identify the source/official-profile differences relevant to your failed case. Next, [readiness and the GPT handoff](lessons/24-ready-for-gpt.md) assemble the complete task traces and remaining evidence.
