## Lesson 23 — evaluation, debugging and a complete small training experiment

**Goal:** run a complete nonlinear classification experiment, interpret its measurements, and diagnose learning failures using falsifiable checks. Prerequisites: losses, backpropagation, optimizers, training-step accounting, data splits and evaluation modes. Attention components are already explained; this small experiment isolates the training foundations before Course 1 adds translation architecture and data.

### Write the task, baseline and evidence boundary

The existing [shared trainer](examples/prerequisite_mlp.py) predicts whether two nonzero real numbers have opposite signs. Features $X$ have shape $[N,2]$, dtype FP32 and CPU device. Labels $y$ have shape $[N]$, dtype int64, and class 1 means $x_1x_2<0$. The model is `Linear(2,8) → Tanh → Linear(8,2)`, with $2×8+8+8×2+2=42$ learned scalars. Its logits have shape $[N,2]$; the model contains no output softmax.

The training grid uses coordinates $±0.3,±0.7,±1.1,±1.5$ on both axes: 64 points. Validation uses $±0.5,±0.9,±1.3$: 36 different points. Both are balanced. A constant class prediction gets 50% accuracy. Uniform two-class probabilities give mean NLL $\log2≈0.693147$. These simple baselines establish what a useful fit must beat.

The validation grid checks interpolation in a small structured domain. It does not establish behavior at zero, far outside the grid, on natural-language text, or on translation. This experiment has no independent final test split; its `validation` name is deliberate. If you tune many settings on these 36 points, acquire a separately declared final assessment before claiming broad generalization.

### Aggregate counts before interpreting averages

For valid labeled records $i$, define $\ell_i=-\log p_{i,y_i}$. Report the loss sum $S=\sum_i\ell_i$, count $N$ and number correct $C$; then mean NLL is $S/N$ and accuracy is $C/N$. Two evaluation chunks with counts 2 and 6 and respective mean losses 1 and 3 combine to $(2×1+6×3)/8=2.5$. Averaging the chunk means gives 2 and overweights the smaller chunk. Translation uses valid target-token counts instead of this classifier's record counts.

Accuracy discards confidence after the highest-logit choice. A correct prediction assigning the target probability 0.51 contributes $-\log0.51≈0.673345$; a correct probability 0.99 contributes $≈0.010050$. Both are one correct record. A confident wrong probability 0.01 contributes $≈4.605170$, so a few such mistakes can dominate NLL.

For a fixed tokenizer and ordinary hard-target token likelihood, perplexity is

$$
\operatorname{PPL}=\exp\left(\frac{-\sum_i\log p_{i,y_i}}{N}\right).
$$

It is the inverse geometric mean assigned target probability. If two target probabilities are 0.5 and 0.25, mean NLL is $\log\sqrt8≈1.039721$ and perplexity is $\sqrt8≈2.828427$. Changing vocabulary/tokenization changes the unit being predicted, so perplexities are not directly comparable across those changes. Exponentiating label-smoothed loss yields an objective transformation, not that ordinary hard-target perplexity.

### Inspect one update before asking for a thousand

Work in the course `examples/` directory using the frozen interpreter selected in Lesson 2. `python` below means that interpreter; the source reference does not replace your foundations-lab implementation files. Run this complete probe:

```python
import math
import torch
from prerequisite_mlp import make_training_state, evaluate, update

torch.set_num_threads(1)
state = make_training_state(seed=7, total_steps=1000)
assert sum(p.numel() for p in state.model.parameters()) == 42
train_x, train_y = state.data['train']
valid_x, valid_y = state.data['validation']
assert len(train_y) == 64 and len(valid_y) == 36
assert not (set(map(tuple, train_x.tolist())) & set(map(tuple, valid_x.tolist())))
assert int(train_y.sum()) == 32 and int(valid_y.sum()) == 18
before = evaluate(state)
first = update(state)
assert first['step'] == 1 and first['examples'] == 16
assert math.isclose(first['learning_rate_used'], 0.0015)
assert not any(p.grad is not None for p in state.model.parameters())
while state.step < state.config.total_steps:
    update(state)
after = evaluate(state)
print('validation before:', before)
print('validation after:', after)
print('updates/examples:', state.step, state.examples_seen)
assert state.step == 1000 and state.examples_seen == 16000
assert after['loss'] < before['loss']
assert after['correct'] == 36
```

The seed-7 declared CPU reference reaches 36/36 on the held-out grid after 1,000 updates. Retain your actual loss and runtime identity instead of copying an author's result into your learning record. A failure is evidence to investigate, not a reason to edit the validation labels. The first update uses rate $0.03/20=0.0015$ during linear warmup; this trainer keeps the rate constant after warmup. It is not the paper's inverse-square-root schedule.

The update implementation chooses microbatches, clears gradients, backpropagates loss sums divided by the common record count, clips the global gradient norm, applies Adam, advances the schedule and counters, then clears completed gradients. Its report contains the rate actually used and the norm **before** clipping. A thousand batch-size-16 updates consumes 16,000 record presentations, not 16,000 unique records.

### Use a small overfit probe as a diagnostic

A model with sufficient capacity should fit a tiny consistent fixture when optimization works and regularization is disabled. Before a large experiment, repeatedly train on the four corners `[-1,-1],[-1,1],[1,-1],[1,1]` with labels `[0,1,1,0]`. In this reference, create a separate state with the same total goal and pass the same `(features,labels)` batch to `update(state,batches=[...])`. This bypasses the sampler deliberately; it is a diagnostic experiment, not a resumed version of the sampled run.

If that tiny fit fails, inspect one operation at a time: output axes, labels, stable loss, nonzero finite gradients, trainable parameter registration, actual rate and parameter changes after `step()`. Use the complete known-answer backward trace from Lesson 7 to distinguish an algorithm bug from a hard optimization problem. The reference's [numerical/training checks](examples/prerequisite_mlp_checks.py) also include a hidden-weight finite-difference comparison, unequal-microbatch equivalence and nonfinite-loss rejection.

### Measure errors and confidence separately

For binary labels, construct a confusion matrix $M_{a,b}$ counting true class $a$ and predicted class $b$. Its diagonal sum is $C$; row totals show whether one class is being ignored. Retain failed coordinates and predicted target probabilities. On a sequence task, also inspect length, EOS failures, source perturbations and free-running outputs; teacher-forced accuracy can look good while errors accumulate during generation.

A probability 0.9 is a confidence statement, not proof of calibration. Calibration asks whether many comparable predictions reported near 0.9 are correct near 90% of the time. Thirty-six structured points cannot support strong calibration claims. High validation accuracy, low NLL and calibration are distinct measurements.

BLEU measures clipped n-gram overlap with reference translations, with a brevity penalty; it is not the percentage of sentences correctly translated or a guarantee of semantic accuracy. Corpus BLEU is not the mean of independently scored sentence BLEUs. Record the reference corpus/split, language direction, tokenization and casing, smoothing, metric version/signature and decoding settings. Reference preprocessing can materially change scores. This classifier has no BLEU result, and its 36/36 grid accuracy cannot be compared with a published translation score. [Metric-reporting paper](source:bleu-reporting), [SacreBLEU reference](source:sacrebleu).

### Diagnose failures with an explicit observation and probe

| Observation | Candidate causes | Small check that distinguishes them |
| --- | --- | --- |
| Loss does not change | Missing gradients, disconnected tensor, wrong parameter group, zero/tiny rate | Compare a parameter before/after one step; inspect its finite gradient and logged rate |
| Tiny training fixture fits, validation does not | Overfitting, mismatched task distribution or split/label error | Inspect held-out errors and split overlap before changing capacity |
| Sequence training is suspiciously easy | Unshifted targets, future-token leakage or duplicate records | Perturb future target IDs with dropout off; check earlier logits and source/target split hashes |
| Loss becomes nonfinite | Invalid probability calculation, exploding values or all-masked softmax | Find the first nonfinite activation/loss/gradient; test extreme-logit and empty-row fixtures |
| Evaluation varies unexpectedly | Dropout still active, changing samples or preprocessing | Fix data order, use `eval()` and `no_grad()`, compare repeated logits |
| Resume diverges | Lost optimizer/sampler/RNG state or changed identity | Compare the next batch, rate and update in a new process as Lesson 24 requires |

These are hypotheses. Record the observed tensor/metric and the result of the distinguishing probe; do not declare “overfitting” from one curve alone. Layer summaries can include activation means/variances, gradient norms and saturation rates. For Tanh, a hidden output close to ±1 has derivative $1-h^2$ close to zero. A norm alone does not locate which layer or parameter produced the problem.

### Run the complete session and record resources

For training with verified generations, choose a new empty run path and execute:

```bash
python prerequisite_mlp.py --mode new --run-dir /tmp/tutorial0-evaluation-run \
  --total-steps 1000 --max-updates 1000 --max-seconds 3600
python prerequisite_mlp_checks.py
```

The CLI refuses to overwrite a nonempty new-run directory. The final JSON reports identity, before/after evaluation, counters, checkpoint generations, elapsed time and process peak RAM. FP32 parameters occupy $42×4=168$ bytes, but optimizer moments, activations and the Python/PyTorch runtime require additional memory. Whole-process peak RAM is a different measure from parameter storage. Record this CPU measurement; it is not a Colab/GPU throughput result. [Reproducibility limits](source:randomness).

### Exercise and notes

[[practice:evaluation-transfer]]

In [my study notes](project/study-notes.md), write the task/domain/split before recording results, calculate the constant-class and uniform-loss baselines, and retain one failed or deliberately corrupted-case diagnosis. Explain what a completed tiny training run proves and what remains unmeasured for translation. Keep author checks, your implementation, your explanation and hosted capability as separate evidence.

Continue with [checkpointing and reproducible recovery](lessons/10-recovery.md), preserving the exact state that produced the next update.
