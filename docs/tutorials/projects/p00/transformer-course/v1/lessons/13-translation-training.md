## Lesson 18 — Translation optimization and complete training updates

**Goal:** implement the reference's token-weighted objective, accumulation, AdamW groups, schedule and update boundaries, then measure a bounded fit. Prerequisites: [complete translation model](lessons/12-translation-model.md) and the earlier supervised-loss lesson. Work in your source copy's `examples/` with Lesson 2's compatible interpreter. [translation_train.py](examples/translation_train.py) owns `token_statistics`, `train_update`, `make_state`, `update` and `evaluate`.

### Define the supervised objective before reducing a batch

For valid shifted label $y_{bt}$ and logits $z_{bt}[V_t]$, the token loss is $\ell_{bt}=\operatorname{logsumexp}(z_{bt})-z_{bt,y_{bt}}$. Its logit derivative is softmax(z)−one_hot(y). The reference uses ordinary unweighted cross-entropy, includes EOS and ignores−100 labels:

$$L=\frac{\sum_{b,t:y_{bt}\ne-100}\ell_{bt}}{N},\qquad N=\sum_{b,t}\mathbf1[y_{bt}\ne-100].$$

The six-pair fixture has thirty labels, so uniform logits over twelve classes yield mean NLL log 12 ≈ 2.484907. This is a baseline, not the exact random model's loss. A valid token has gradient `(p−one_hot)/N`; an ignored position has zero supervised contribution. Target-key padding exclusion remains a separate attention contract.

`token_statistics` flattens B,T in logits and labels consistently, requests `reduction='sum'`, and returns sum/count/correct count. It rejects a microbatch with no supervised target. Aggregating counts once is essential: adding padding cannot improve the objective by enlarging its denominator. Accuracy uses argmax and discards confidence, so a falling NLL need not immediately change the number correct.

### One accumulation group produces one update

Suppose microbatch A contains two valid targets and summed loss 4; B contains five targets and summed loss 5. The token objective is 9/7 ≈ 1.285714. Equal averaging of their means gives `(2+1)/2=1.5`, overweights A, and produces another gradient. Backpropagate S_A/7 then S_B/7; derivatives add to the intended combined mean before changing parameters.

[[diagram:token-weighted-update]]

The source's `train_update(model,optimizer,batches,scheduler=...,max_grad_norm=1)` follows this order:

| Boundary | Actual operation and reason |
|---|---|
| Start group | `model.train()` and `optimizer.zero_grad(set_to_none=True)` once |
| Each microbatch | `logits_for` passes source/prefix IDs and both key masks; sum loss divided by the group's valid count calls backward |
| After accumulation | Global gradient norm clipping; abort nonfinite norms before parameter or scheduler changes |
| Commit update | One `optimizer.step()`, then one `scheduler.step()` when supplied |
| Reporting | Detached sum/count, argmax accuracy, valid-token count and **preclip** norm |

Clearing between microbatches loses earlier contributions. Stepping between them changes the parameters inside the intended group. The helper leaves completed gradients present; the next update clears them. Saving mid-group would require those partial gradients and its denominator, whereas the session protocol saves after complete updates.

Without dropout and with controlled batch-dependent behavior, accumulated and combined gradients agree up to floating-point ordering. Run `python translation_checks.py TranslationChecks.test_unequal_microbatch_accumulation_matches_full_batch`: it uses unequal target lengths and compares every parameter after one SGD update. A low loss alone cannot expose this weighting error.

### AdamW groups and the exact protocol

`adamw_groups` decays trainable parameters with rank≥ 2, including embeddings and matrix weights, and excludes lower-rank biases/LayerNorm scales. The grouping is dimensional, not a substring search for a particular normalization spelling. `make_state` uses base rate .01, betas(.9,.98), epsilon 1 e−6 and matrix decay .001.

For one decayed coordinate θ=2, gradient 2 and an initial Adam state, m=.2,v=.08; corrected moments are 2 and 4. The adaptive movement is approximately−.01. Decay separately shrinks θ by `(1−.01×.001)`, yielding approximately 1.98998 after both operations. Injecting λθ into the gradient would alter the moments instead, so it is not the same adaptive update. [AdamW API](source:torch-adamw).

The paper's Adam/inverse-square-root schedule and smoothing differ from both this reference and the pinned source's AdamW/cosine/plain-CE recipe. The reference's rate .01 and short duration are chosen for its miniature diagnostic task; the pinned large training script uses rate 1 e−4 and warmup 2000. Retain those source/scale distinctions rather than calling this a WMT reproduction.

### The schedule is indexed by completed optimizer updates

Let U be the fixed total goal, w warmup and k the LambdaLR index. The source multiplier is

$$\alpha(k)=\begin{cases}(k+1)/w,&k<w,\ w>0,\\
\tfrac12[1+\cos(\pi\min(1,(k-w)/(U-w)))],&k\ge w.
\end{cases}$$

For U=8,w=2, indices 0,1,2,5,8 give multipliers .5,1,1,.5,0. The `(k+1)` convention reaches the peak at index 1 and again at index 2; state it rather than hiding an off-by-one comparison. With the default 400-update goal, w=20 and the **first used** rate is .01/20=.0005.

`LambdaLR` is initialized before training and advanced after optimizer steps. Session JSON lists `get_last_lr()` after that advance, describing the next scheduled rate, not necessarily the rate just used. Record the optimizer rate before `update` if comparing used rates. A reconstructed LambdaLR closure needs the same total/warmup settings; loading only its index does not restore a different schedule function.

### Inspect real update state before a longer run

Run this standalone probe from `examples/`:

```python
import math
import torch
from translation_train import make_state,update,adamw_groups,warmup_cosine

torch.set_num_threads(1)
state = make_state(seed=7,steps=400)
used = state.optimizer.param_groups[0]['lr']
old = state.model.head.weight.detach().clone()
result = update(state)
assert math.isclose(used,.0005)
assert result['tokens'] == 30 and state.step == 1
assert torch.isfinite(state.model.head.weight).all()
assert not torch.equal(old,state.model.head.weight)
assert math.isclose(state.scheduler.get_last_lr()[0],.001)
groups = adamw_groups(state.model)
assert all(p.ndim>=2 for p in groups[0]['params'])
assert all(p.ndim<2 for p in groups[1]['params'])
for k,expected in [(0,.5),(1,1.),(2,1.),(5,.5),(8,0.)]:
    assert math.isclose(warmup_cosine(k,total_steps=8,warmup_steps=2),expected,abs_tol=1e-12)
print('used rate:',used,'next rate:',state.scheduler.get_last_lr()[0],'targets:',result['tokens'])
```

Then run `python translation_train.py --steps 400 --seed 7`, optionally with `--report-json /tmp/tutorial1-translation-fit.json`. This command executes one fixed six-pair batch per update:400 updates mean 2400 pair presentations and 12000 supervised labels, not 2400 unique sentences. `evaluate` disables dropout/graphs, aggregates totals and restores caller mode. The reference checks require sharply reduced training loss and exact generation of the six training sentences; two held-out recombinations remain a narrow diagnostic, not a benchmark-quality estimate. Lesson 20 evaluates those distinctions and source controls.

The direct `update` helper does not enforce the total budget itself; the CLI's fixed loop and `course_session` stop at the recorded goal. Use the session wrapper for resumable work. The inspected source also discusses distributed training: if R DDP ranks average gradients, each rank's token-loss sum scaled by R/N_global yields a global token mean under a shared reduced count. Accumulation wrappers can add scaling; this successful CPU reference does not establish a distributed execution result.

### Practice and notes

[[practice:translation-training-practice-1]]
[[practice:translation-training-schedule-transfer]]

In my notes, derive the 2-versus 5 token weighting, calculate the first used/next rates, annotate where gradients remain after commit, and list every optimizer-group/schedule field recovery must preserve. Record actual loss, token accuracy, free-running outputs and protocol identity; do not paste an author's threshold as my result. Continue with [autoregressive decoding](lessons/14-translation-generation.md).
