## Lesson 30 — constrained span decoding and QA metrics

**Goal:** choose an ordered context span jointly, recover original text, and calculate normalized exact match/token F1 with explicit window and no-answer limits. Prerequisites: packed offsets, endpoint logits and positive-answer QA training. A high endpoint score is not yet a valid answer.

### Derive joint constrained ranking

For start logits $s_i$ and end logits $e_j$, independent endpoint distributions give a fixed-window pair score

$$\log p_s(i)+\log p_e(j)=s_i+e_j-\log\sum_u e^{s_u}-\log\sum_v e^{e_v}.$$

The two normalization terms are constant across pairs **within one window**, so ranking valid pairs by $s_i+e_j$ is equivalent there. The valid set requires context-owned nonempty endpoint offsets, $i\leq j$, no noncontext position inside the span, and length $j-i+1\leq L_{\max}$. A probability factorization does not make independent argmax obey those constraints.

Consider a question/special position 0 with scores 100/100, followed by context `a b c` with start `[5,1,9]` and end `[8,7,1]`. Global independent argmax selects noncontext position 0. Even after restricting endpoints to context, it selects start 3/end 1, reversed. With maximum length two, valid scores include `(1,1):13`, `(1,2):12`, `(2,2):8`, `(2,3):2`, `(3,3):10`. The joint best span is `(1,1)` and answer `a`.

[[diagram:qa-joint-span]]

[best_span](examples/roberta_data.py) enumerates starts and ends subject to that maximum, excludes `None` endpoint/interior offsets, and chooses the first encountered maximum under its strict `>` comparison. This bounded search costs order $T L_{\max}$ candidate inspections; its current interior-ownership check adds work proportional to candidate span length. It returns `None` when no valid context candidate exists. It does not add a trained null-answer score.

### Slice original characters rather than reconstructed tokens

For selected inclusive token span $(i,j)$ with endpoint offsets $[l_i,r_i),[l_j,r_j)$, return `context[l_i:r_j]`. The end character remains exclusive even though the token end index is inclusive. This preserves original casing, accents and interior whitespace. Token decoding may normalize spacing in another tokenizer or show invalid single-byte fragments in this byte tokenizer; it is not the authoritative original answer rendering.

Repeated character offsets from UTF-8 byte symbols are legitimate. They can make multiple token pairs extract the same original substring. Context whitespace symbols may have empty offsets inside a span but cannot be chosen as endpoints. Question, separator and PAD offsets are `None`; a predicted high score there cannot override context ownership.

A 20-token gold answer is impossible when the declared decoder cap is 16, regardless of training quality. Report coverage lost to window capacity and length constraints separately from model scoring errors. Changing the cap or window construction changes the evaluated system and requires rerunning the measurement.

### Preserve record ownership and acknowledge cross-window scores

`predict_qa` constructs all features for one original question/context record, evaluates each, decodes its best valid span and chooses the greatest raw start-plus-end score. When batching multiple records, retain a record ID or official `overflow_to_sample_mapping`; do not compare a window belonging to one question with another's candidates. Evaluate one final prediction per original question, not one metric per overflow feature.

The fixed-window ranking equivalence does **not** establish cross-window calibration. Adding constant $c_s$ to every start logit and $c_e$ to every end logit leaves each endpoint distribution unchanged but shifts every raw pair score by $c_s+c_e$. For example window A's uniform `[0,0]`/`[0,0]` has pair score 0; window B's uniform `[0.5,0.5]` twice has score 1 although both distributions are identical. Shifting A's two arrays by 2 makes its score 4 and reverses raw ranking without changing its probabilities.

This reference uses that raw-score heuristic explicitly. Normalized scores, maximum-context heuristics or other aggregation/calibration rules are different policies to test, not guarantees that every window then becomes directly comparable. Inspect repeated-answer and boundary cases before attributing cross-window failures entirely to the backbone.

### Calculate normalized EM and multiset F1

`normalize_answer` lowercases, removes ASCII punctuation, removes English articles `a/an/the`, and collapses whitespace. It does not stem words or remove accents. Exact match is equality after this normalization. Thus `The RED apple!` and `red apple` have EM one; a semantically plausible paraphrase with different normalized tokens need not.

For prediction token multiset $P$ and gold multiset $G$, overlap is $C=\sum_w\min(\operatorname{count}_P(w),\operatorname{count}_G(w))$. Then precision is $C/|P|$, recall $C/|G|$, and, for nonempty multisets, $F1=2C/(|P|+|G|)$. Prediction `red red blue` versus gold `red blue blue` has overlap two, precision/recall $2/3$, F1 $2/3$ and EM zero. Using set intersection would incorrectly ignore repeated-token counts.

With multiple gold answers, the helper takes the best EM and best F1 over supplied alternatives. Its per-question values lie in `[0,1]`; report their dataset means or explicitly multiply by 100 for percentage scores. Do not average overflow-window metrics as if they were separate questions. The declared normalization/scale and gold corpus remain part of the evaluation contract. [Maintained SQuAD metric implementation](source:hf-squad-metric).

### Probe legal pairs, normalization and the score-shift limit

From `examples/` with the frozen interpreter:

```python
import math
import torch
from roberta_data import best_span, answer_metrics

result = best_span([100,5,1,9], [100,8,7,1],
    [None,(0,1),(2,3),(4,5)], 'a b c', max_answer_tokens=2)
assert result['text'] == 'a' and (result['start_token'],result['end_token']) == (1,1)
assert answer_metrics('The RED apple!', ['red apple']) == {'exact_match':1.,'f1':1.}
r = answer_metrics('red red blue',['red blue blue'])
assert r['exact_match'] == 0 and math.isclose(r['f1'],2/3)
a = torch.tensor([0.,0.])
b = torch.tensor([.5,.5])
assert torch.equal(a.softmax(-1),b.softmax(-1))
assert torch.equal(a.softmax(-1),(a+2).softmax(-1))
assert float(2*a[0]) < float(2*b[0]) < float(2*(a+2)[0])
print('legal answer:', result['text'], 'multiset F1:', r['f1'])
```

The probe has no trained model; it independently verifies decoding/metric arithmetic and demonstrates why probability-invariant shifts matter across windows. The complete reference's `predict_qa` calls the same decoder and obtains original-text answers after full QA fine-tuning.

### Keep positive-only QA separate from abstention

The metric helper defines sensible empty-string comparison behavior, but the model has no trained answerability score or calibrated no-answer threshold. Ignoring an unavailable window is not a negative training example, and returning no valid candidate due to missing context is not learned abstention. A SQuAD 2.0-style system needs its own data, null/answerability objective, decision rule and held-out calibration/evaluation.

The tiny run reports predictions on eight **training** questions. Perfect seen-fixture EM tests fitting/alignment/decoding mechanics, not unseen QA quality. Keep errors by type: missing coverage, wrong coordinates, special-token peak, reversed pair, length exclusion, cross-window choice and lexical metric mismatch. Those distinctions lead to different fixes.

[[scenario:span-validity]]

### Changed-condition practice and notes

[[practice:span-decoding-and-metrics-practice-1]]

[[practice:span-decoding-and-metrics-transfer]]

In [my study notes](project/study-notes.md), list all valid pairs for the toy case, derive EM/F1 with duplicate tokens, and show a window-wide constant shift. Explain which guarantees come from offsets/constraints and which need learned evidence. Compare the own-format positive-only task with an official dataset/answerability profile. Next, [integrated debugging and profiling](lessons/31-debugging-and-profiling.md) tests these contracts across all three task paths.
