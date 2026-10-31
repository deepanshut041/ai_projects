# Build and explain the complete models

Follow the [32-topic outline](project/course-outline.md) and [start with the model specifications](lessons/01-start.md). Use [my study notes](project/study-notes.md) for derivations, diagrams, changed predictions and observations. A runnable reference supports comparison; it does not complete my own implementation or explanation.

| Topics | Result and important contracts | Source owners and checks |
|---|---|---|
| 1–6 | Reproducible task, tensor, loss, tokenizer and corpus specifications | source manifest, model configs, translation_data.py; axes, exact IDs, fit/split ownership and token counts |
| 7–14 | Embeddings, positions, learned attention/heads/masks, batching, FFN and post-norm residuals | translation_attention.py and translation_model.py; arithmetic/backward probes, layout, allowed-key parity and shifted labels |
| 15–17 | Complete encoder, decoder and conditional vocabulary model | translation_model.py; unique layer ownership, rectangular retrieval, source/future isolation and gradients into every required path |
| 18–21 | Counted training, generation/evaluation and complete-state continuation | translation_train.py, translation_metrics.py, course_checkpoint.py/session.py; unequal accumulation, EOS, source controls and exact fresh-process comparison |
| 22–26 | Bidirectional RoBERTa-style MLM with own-format byte-BPE and tied head | roberta_data.py/model.py/train.py; position identity, eligible/selected/replaced targets, shared parameters and fixed corruption measurement |
| 27–30 | Compatible backbone transfer into windowed extractive QA | same RoBERTa modules; fresh task state, original character offsets, packed endpoint losses and valid joint spans/EM/F1 |
| 31–32 | Falsifiable diagnosis, measured resource profile and Small GPT handoff | applied probes, explanation/readiness records and both full task traces |

## Preserve actual learning work

The current foundations implementation is P00-M1 setup complete, with `linear()` next in the relocated `projects/transformer-labs/transformer-foundations-lab`. Follow that project's checkpoint; authoring Tutorial 1 does not advance it.

The existing `attention-is-all-you-need` companion has checked attention and its public projection names. Continue its data/model work from that evidence when requested. Its code remains distinct from this course's complete reference; copying a reference over learner files is not an implementation exercise.

## Use the reference and editable study export

From the exported source folder, select the recorded Torch/NumPy runtime, then run `python translation_checks.py`, `python roberta_checks.py`, `python -m unittest test_model_learning_probes`, `python course_recovery_checks.py --project all` and `python course_session_checks.py`. Bounded training is included in the relevant source checks. The exporter includes every model/helper/trainer, blank study notes and an output-cleared study notebook; the separate Colab notebook imports the same session trainer.

Before extending a component, predict a changed numerical case, implement/inspect its exact file and run an assertion that can expose the expected failure. Record source/config/tokenizer/split identity, supervised counts, objective and schedule, seed, runtime, observations and unresolved errors. Keep generated runs and weights outside Git.

[Reference results](project/reference-runs.md) state bounded author evidence. [Readiness](project/readiness.md) requires my implementation/explanation, meaningful held-out task evaluation, measured budgets and persistent deleted-runtime recovery separately. [Source map](project/source-map.md) records paper/source/miniature differences.
