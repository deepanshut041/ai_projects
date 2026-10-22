## Lesson 26 — MLM pretraining, counted updates and measurement

**Goal:** connect sampled corruption to a correctly weighted optimizer update, inspect the actual training recipe, and measure a fixed reconstruction protocol without overstating language quality. Prerequisites: selected-target loss, tied head, optimizer/schedule state and corpus boundaries. [roberta_train.py](examples/roberta_train.py) owns this complete experiment; [course_session.py](examples/course_session.py) adds bounded persistence.

### Declare data construction and sampling

`build_training_bundle('mlm')` learns 80 byte merges from eight authored `TRAIN_TEXTS` documents **plus the training QA questions**. `VALIDATION_TEXTS` is excluded. This shared fixture makes tokenizer identity consistent for the downstream demonstration; it is not an independently isolated pretraining-only corpus experiment. Do not carry that convenience into a benchmark while hiding task-data exposure.

The default MLM data keeps each document as one encoded sequence, including BOS/EOS, and rejects sequences longer than context capacity. `next_batch` samples four indices independently with replacement; it is not one shuffled epoch traversal. A repeated document is a new presentation with a new corruption draw. Count presentations, input tokens and supervised selected tokens separately from unique documents and completed optimizer updates.

An optional `pack_documents` helper concatenates already-split encoded documents and divides the stream into bounded chunks, retaining the final tail. For byte-only texts `['ab','c']` and capacity four it produces `[[0,102,103,2],[0,104,2]]`. Flattening recovers the entire original stream. BOS/EOS boundaries remain, but attention can cross them inside a chunk and chunks can cut a document. The default trainer does not silently use this packing path. Split/deduplicate records before fitting or packing; separator IDs do not prevent data leakage.

### Derive one group denominator

For microbatch $r$ with selected count $K_r$ and loss sum $S_r$, the intended token-mean objective is

$$L=\frac{\sum_r S_r}{K},\qquad K=\sum_r K_r.$$

Backpropagate each $S_r/K$, accumulate, then step once. Suppose selected counts are 2 and 6 and batch means are 1 and 3: $S_1=2,S_2=18$, combined mean $20/8=2.5$. Equal averaging of batch means gives 2 and overweights the smaller selected set. Averaging by input length or visible MASK count defines yet another objective.

For $K=0$, this reference consumes the sampled/corruption RNG draws and reports no update. A nonzero group can contain a zero-target microbatch; its sum is zero and the common denominator remains the valid targets in other microbatches. `accumulation_steps` belongs to run identity because changing it changes effective examples, corruption and update semantics.

### Inspect the actual optimizer and schedule

The source uses AdamW with rate 0.003, betas `(0.9,0.98)`, epsilon `1e-6`, global norm clipping at 1.0, and weight decay 0.01 for parameters of rank at least two. Rank-one parameters—including biases and LayerNorm scale/shift—have zero decay. Embedding and position matrices belong to the decay group; tying does not justify adding the same parameter twice. [Optimizer reference](source:torch-adam).

Let $U$ be total updates, $w=\min(10,U-1)$ warmup indices and $k$ the scheduler index. The source factor is

$$f(k)=\begin{cases}(k+1)/w,&k<w,\\(U-k)/(U-w),&k\geq w,\end{cases}$$

with a floor at zero after the planned goal. `LambdaLR` initializes the rate using index zero: for $U=120,w=10$, the first update uses $0.003/10=0.0003$. The factor equals one at both indices 9 and 10, then decreases linearly to zero at index 120. Log the rate **before** the optimizer step to identify the rate used, and advance the scheduler once per completed update. This is warmup plus linear decay, not the original Transformer's inverse-square-root schedule or generic Adam defaults.

`training_step` sets training mode, clears existing gradients, divides each loss sum by the common target count, backpropagates, clips, updates AdamW, advances the schedule and increments `step`. Unlike the translation reference, it currently lacks an explicit finite-loss/finite-gradient rejection before stepping. Inspect finiteness during development; do not claim clipping alone prevents nonfinite parameter corruption. The bounded session protects persistence and update/time limits, which are separate safeguards.

### Fix evaluation corruption independently of training

`fixed_mlm_evaluation` encodes two declared validation documents, uses a separate generator seeded 900 and corruption probability 0.35, selects evaluation mode and disables autograd. Repeating it under identical weights/data produces the same reconstruction targets, enabling a meaningful before/after loss comparison. It reports selected count and correct selected predictions; accuracy is `correct/target_count`, not divided by all padded cells.

For uniform vocabulary scores, baseline selected NLL is $\log V$; at $V=341$, approximately 5.831882. A lower fixed-corruption NLL indicates improvement on those particular targets. It does not average all possible corruption patterns or establish general-text likelihood. Report additional mask seeds and separate held-out documents for a broader assessment. This helper leaves `model.eval()` active; the next `training_step` explicitly returns to training mode. Do not silently assume every evaluation helper restores prior mode.

### Probe a real update and repeatable evaluation

From `examples/` using the frozen interpreter:

```python
import math
import torch
from roberta_train import build_training_bundle, training_step, fixed_mlm_evaluation

torch.set_num_threads(1)
b = build_training_bundle('mlm', seed=7, total_steps=20)
assert b.tokenizer.vocab_size == 341
assert sum(p.numel() for p in b.model.parameters()) == 39957
first = fixed_mlm_evaluation(b)
again = fixed_mlm_evaluation(b)
assert first == again and not b.model.training
rate_used = b.optimizer.param_groups[0]['lr']
r = training_step(b, accumulation_steps=2)
assert r['updated'] and b.step == 1 and b.model.training
assert r['target_count'] > 0 and math.isfinite(r['loss'])
assert math.isclose(rate_used, 0.0003)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.model.parameters())
print('fixed targets:', first['target_count'], 'group targets:', r['target_count'], 'rate used:', rate_used)
```

The default complete model has 39,957 unique parameters; the selected group count is an actual corruption measurement, not a planned fraction rounded from sequence length. The probe's two-microbatch configuration is a separate diagnostic profile, so it must not be inserted into an existing run with another accumulation identity.

For a complete nonpersistent mechanics demo, run `python roberta_train.py --task mlm --steps 120`. For resumable pretraining, use the common session and choose a new run directory:

```bash
python course_session.py --task mlm --mode new --run-directory /tmp/course-mlm-study \
  --total-steps 120 --additional-updates 25 --export-backbone
```

The latter writes verified full-state generations and a separate `backbone.pt` transfer artifact. Existing run directories are rejected in `new` mode. Later slices use explicit `resume` with the same total goal/source/runtime; a weight-only backbone export is not a recovery checkpoint. [Reference runs](project/reference-runs.md) labels actual author outcomes separately from learner evidence.

### Inspect fill-mask output as a token ranking

`fill_mask` requires exactly one literal `<mask>` marker, inserts one structural MASK ID, predicts that slot, and returns highest-scoring ordinary token candidates with their vocabulary probabilities. It excludes special candidates without renormalizing their remaining mass. A candidate may be a byte/subword fragment rather than a whole word; arbitrary single-byte decoding can show a replacement character. A single-token ranking is not a general multiword fill or generated QA answer.

### Changed-condition practice and notes

[[practice:mlm-training-and-counting-practice-1]]

[[practice:mlm-training-and-counting-transfer]]

In [my study notes](project/study-notes.md), work unequal selected counts, draw the scheduler index before/after one update, and distinguish document sampling, input-token budget and supervised-token budget. Record the exact corruption/data protocol beside every result and compare it with the original RoBERTa recipe without claiming an official pretrained artifact. Next, [backbone transfer](lessons/27-backbone-transfer.md) starts a new supervised stage with compatible contextual weights.
