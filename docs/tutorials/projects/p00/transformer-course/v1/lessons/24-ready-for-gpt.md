## Lesson 32 — explain the complete models and prepare the Small GPT handoff

**Goal:** explain all three task paths from data to gradient/inference, distinguish the evidence each demonstrates, and identify what changes for a decoder-only language model. Prerequisites: the complete translation and RoBERTa/QA lessons, debugging and full-state recovery. Readiness means concrete implementation/checks and explanation, not simply having visited 32 pages.

### Draw complete task paths with their real class axes

For translation: source text → frozen source IDs → source embeddings/sinusoidal positions → bidirectional encoder memory. Shifted target IDs → target embeddings/positions → causal self-attention → source cross-attention → position-wise FFN/residual blocks → target-vocabulary logits. Training uses recorded earlier target tokens and valid next-target labels; inference feeds back generated tokens, with independent EOS completion and capacity/budget limits.

For MLM: original text → own-format byte IDs → eligible/selected corruption with original labels retained → token plus learned-position embeddings → bidirectional encoder → dense/GELU/LayerNorm transform → tied vocabulary scores. Selected-target loss updates head/table/backbone; real unselected context can receive gradients through attention. It reconstructs selected inputs under a corruption protocol rather than generating an unrestricted continuation.

For extractive QA: original question/context/answer characters → pair layout and overlapping windows → aligned inclusive context endpoints → compatible transferred encoder plus new two-column head → start/end logits → context-only joint constrained span → original-text slicing. Training fine-tunes backbone/head with new optimization state; inference returns supplied-context text, not a new free-form answer or a learned no-answer decision.

[[diagram:two-builds-to-gpt]]

| Task operation | Output | Classification or reduction axis | Permitted information |
| --- | --- | --- | --- |
| Translation encoder | $[B,S,D]$ memory | Feature transforms, source-key attention | All real source tokens |
| Translation target head | $[B,T,V_t]$ | Target vocabulary | Target prefix plus source memory |
| RoBERTa MLM head | $[B,T,V]$ | Vocabulary; loss only selected labels | Full corrupted real input |
| QA start/end head | Two $[B,T]$ arrays | All packed positions in current CE; context-only constraints in decoding | Question plus context window |
| Small GPT next-token head | $[B,T,V]$ | One text vocabulary | Token prefix under causal visibility |

An identical attention formula or scalar loss does not establish identical tasks, normalization domains or checkpoint compatibility.

### Explain counted loss through a changed numerical example

Suppose translation has loss sum 10 over five valid targets, MLM sum 6 over three selected targets, and QA has start sum 6/end sum 2 over two valid windows. Their mean objectives are respectively $10/5=2$, $6/3=2$, and $(6+2)/(2×2)=2$. The equal numerical value does not make these equally good models: the prediction units, vocabulary/position domains and supervised examples differ.

For each, derive the local class gradient $p-y$ with its correct denominator, then identify how that derivative reaches earlier layers. Translation cross-attention routes target gradients into source memory. MLM selected logits route gradients into contextual keys/values and both tied-table uses. QA combines start/end signals into each shared contextual row. A direct gradient zero at one output position is not necessarily a zero upstream parameter gradient.

In each trace, label integer IDs, floating representations, allowed-key masks, ignored targets and original coordinates separately. Explain one failure concretely: wrong target shift, mistaken Boolean mask, replacement ID used as its own MLM label, changed tokenizer under old weights, byte offset used as a character offset, or independent reversed QA endpoints. Name the check that exposes it.

### Verify the three actual interfaces

From `examples/` using the frozen interpreter:

```python
import torch
from translation_model import Transformer, TransformerConfig
from roberta_model import RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering

torch.set_num_threads(1)
torch.manual_seed(7)
t = Transformer(TransformerConfig(src_vocab_size=10,tgt_vocab_size=12,
    embedding_dimension=8,num_attention_heads=2,max_src_len=4,max_tgt_len=4)).eval()
c = RobertaConfig(vocab_size=30,embedding_dimension=8,
    num_transformer_blocks=1,num_attention_heads=2,context_length=8)
m, q = RobertaForMaskedLM(c).eval(), RobertaForQuestionAnswering(c).eval()
with torch.no_grad():
    translation = t(torch.tensor([[2,4,0]]),torch.tensor([[2,4,3]]))
    ids = torch.tensor([[0,4,8,2]])
    mlm = m(ids)['logits']
    qa = q(ids)
assert translation.shape == (1,3,12)
assert mlm.shape == (1,4,30)
assert qa['start_logits'].shape == qa['end_logits'].shape == (1,4)
assert 10/5 == 6/3 == (6+2)/(2*2) == 2
print('translation:',tuple(translation.shape),'MLM:',tuple(mlm.shape),'QA:',tuple(qa['start_logits'].shape))
```

These independent miniature configurations verify interfaces; they are untrained and do not share compatible vocabularies or weights. The matching width is deliberate for shape inspection, not a conversion scheme.

### Run the reference checks and assess their actual scope

The complete local reference uses these unchanged commands from `examples/`:

```bash
python -m unittest discover -s . -p test_foundation_math.py -v
python translation_checks.py
python roberta_checks.py
python course_recovery_checks.py
python course_session_checks.py
```

They cover independent arithmetic/finite differences, attention parity, visibility/alignment, generation, weighted losses, Unicode offsets, corruption/heads/transfer/spans, fresh-process continuation and persistence/session failures. Complete mechanics demonstrations are `python translation_train.py --steps 400` and `python roberta_train.py --task both --steps 120`. Use common session commands for actual resumable work; demos and weight-only exports do not replace that state.

[Reference runs](project/reference-runs.md) states the corpus, corruption, runtime and outcomes of author runs. Tiny held-out translation combinations, two fixed-corruption MLM documents and eight seen QA questions establish different bounded checks. They do not reproduce WMT/RoBERTa/SQuAD scale or general language quality. A fresh local replay proves a declared compatible continuation, while Drive survival requires a successful **Disconnect and delete runtime** drill with recorded restored/advanced updates. Keep hosted evidence pending until performed.

### Carry specific evidence into the next project

The [implementation record](project/implementation.md) and [readiness](project/readiness.md) distinguish inspected source, your implementation, numerical invariants, measured training, task evaluation, your explanation and local/hosted recovery. For each unresolved item, write an exact file/component, fixture and observable acceptance check. Do not fill learner results from author reference checks.

Use [my study notes](project/study-notes.md) for your derivations; [explanation practice](practice/explain-your-model.md) and [changed-case questions](practice/questions.md) support reviewing a component. [The outline](project/course-outline.md), [translation record](project/translation.md), [RoBERTa record](project/roberta.md), [source map](project/source-map.md) and [research](research.md) retain the source/profile choices without creating another mandatory theory route. Actual learner P00-M1 still resumes `linear()` in the existing foundations child until its own evidence changes.

### Choose the decoder-only changes explicitly

Course 2 develops Small GPT with a next-token objective, one text tokenizer/corpus, causal self-attention, learned positions, pre-norm blocks, GELU FFN and a tied vocabulary head. Removing the translation encoder removes source memory and cross-attention; it does not automatically select all those remaining contracts. RoBERTa's bidirectional visibility would expose future next-token targets if copied unchanged.

Transfer the tested habits: exact tokenizer/split/source identity, shape/axis checks, legal causal information, token-weighted accumulation, optimizer/schedule/generator state, bounded stopping and verified recovery. Recheck them under the new task's position numbering, vocabulary, data and generation behavior. Mutable sibling imports make the new experiment depend on hidden changes; reuse explicit versioned source/artifacts instead.

The own-format contextual encoder and translation weights are not a ready-made official GPT checkpoint. Architecture similarity also does not create official-tokenizer compatibility or a conversational instruction-following system. Instruction data, alignment, serving and their evaluation are later tasks. The useful handoff is a tested model/data/training discipline with clearly named remaining work.

[[practice:ready-for-gpt-practice-1]]

[[practice:ready-for-gpt-transfer]]



### Keep editable source and attribution available

Model/helper/trainer source is separate from notebooks and personal notes:

- [foundation_math.py](examples/foundation_math.py), [foundation_math.cpp](examples/foundation_math.cpp), [test_foundation_math.py](examples/test_foundation_math.py) and [first_retrieval.py](examples/first_retrieval.py)
- [translation_attention.py](examples/translation_attention.py), [translation_data.py](examples/translation_data.py), [translation_model.py](examples/translation_model.py), [translation_metrics.py](examples/translation_metrics.py), [translation_train.py](examples/translation_train.py) and [translation_checks.py](examples/translation_checks.py)
- [roberta_data.py](examples/roberta_data.py), [roberta_model.py](examples/roberta_model.py), [roberta_train.py](examples/roberta_train.py) and [roberta_checks.py](examples/roberta_checks.py)
- [course_checkpoint.py](examples/course_checkpoint.py), [course_recovery_checks.py](examples/course_recovery_checks.py), [course_session.py](examples/course_session.py), [course_session_checks.py](examples/course_session_checks.py) and [course_source_export.py](examples/course_source_export.py)

Retain [the MIT license](examples/LICENSE.txt), [source acquisition provenance](source-provenance.json) and [selected reference verification](reference-verification.json). New source/study exports need their own digests and must preserve existing personal notes. Local authoring/checks remain separate from commits or external publication.

In [my study notes](project/study-notes.md), draw all three full paths, compare their loss denominators, and identify one unresolved implementation with its independent fixture. State what changes for Small GPT and what your present evidence does not establish. Use that concrete next action to continue rather than a generic completion label.

### Study and source tools

Use [the study notebook generator](examples/course_study_notebook.py), [source export generator](examples/course_source_export.py), [applied numerical probes](examples/model_learning_probes.py) and [independent probe checks](examples/test_model_learning_probes.py). The export includes a cleared study notebook and editable topic notes beside all reference modules. [Build record](project/implementation.md) and [author reference results](project/reference-runs.md) state their scope.

[Expanded topic verification](topic-verification.json) and [study/export checks](examples/test_course_study.py) record the current artifact checks.
