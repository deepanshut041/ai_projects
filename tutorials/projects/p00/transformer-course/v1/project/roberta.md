# RoBERTa build workbook

Use this record to return to the second build at a specific file and check. The teaching route is [Lesson 22 — BERT-to-RoBERTa design and bidirectional objectives](lessons/16-bert-to-roberta.md) through [Lesson 30 — constrained span decoding and QA metrics](lessons/23-span-decoding-and-metrics.md); this workbook does not replace those explanations. The complete editable programs live in `examples/`. No learner RoBERTa repository was created by writing this course.

The intended result is a complete miniature masked-language model, a compatible encoder transfer, and an extractive QA model that can train and return constrained answer spans. Useful broad language understanding, held-out QA quality, learner explanation and hosted recovery require their own evidence.

## Establish one reproducible profile

Begin with the offline CPU profile. From the course `examples/` directory prepared in [Lesson 2 — Reproducible source and implementation workspace](lessons/02-workspace.md), `python` means the frozen Python interpreter selected there.

| Setting | Offline reference | What a changed profile must record |
| --- | --- | --- |
| Text | Eight original training documents, two validation documents, eight training QA records | Corpus version, licence, duplicate policy and document split IDs |
| Tokenizer | Train-only byte BPE, 256 base bytes, 80 merges, five specials; 341 IDs for the fixture | Complete tokenizer files/state, normalization, special IDs and digest |
| Encoder | Width 32, two blocks, four heads, FFN ratio 4 | Configuration digest and parameter count |
| Context | At most 64 input tokens including specials | Position capacity, sequence construction and overflow policy |
| MLM corruption | Probability 0.35 for the quick run; 80/10/10 branches | Selection probability, replacement pool, generator state and evaluation seed |
| Training | Seed 7, 120 updates per task, AdamW, warmup then linear decay | Optimizer/scheduler settings, accumulation, update count and runtime |
| Execution | PyTorch 2.14.1 CPU author reference | Exact installed versions, dtype/device and deterministic execution settings |

The general corruption function defaults to probability 0.15. The quick trainer deliberately raises it to 0.35 to supply enough selected targets for short runs. Neither setting makes the fixture a reproduction of the paper's large training recipe.

Run the reference checks first:

```bash
python roberta_checks.py
```

Keep the actual command, code revision/digests, result and runtime in your implementation record. A reference pass says the provided program passed; it does not say you have implemented or explained its mechanisms independently.

## Build and inspect the data contract

Create or complete `roberta_data.py` in your RoBERTa working copy following [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md). Preserve its ordered byte merges and special-token meanings. Add padding and corruption from [Lesson 24 — dynamic MLM corruption and selected-target supervision](lessons/18-dynamic-mlm-corruption.md).

Before a model enters the loop, verify these concrete invariants:

- Decoding the IDs for `café`, leading spaces and previously unseen Unicode recovers the original text.
- Character offsets for the two UTF-8 bytes of `é` identify the same character interval.
- A shorter padded row has `False` at its padding key positions.
- Beginning, separator, padding, unknown and mask IDs are excluded from MLM target selection.
- Mask, random and unchanged branches all retain the original selected token as their label.
- Corruption preserves the caller's original input matrix; unselected labels use `-100`.

The padding mask and selected-target mask have different jobs. An input `<mask>` remains a real, allowed key; an unchanged selected token still contributes loss. Do not count only literal mask IDs as supervised targets.

## Build the encoder and vocabulary head

Create `roberta_model.py` in the same working copy. [Lesson 22 — BERT-to-RoBERTa design and bidirectional objectives](lessons/16-bert-to-roberta.md) introduces the configuration, [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md) introduces embeddings, and [Lesson 25 — bidirectional encoder and tied MLM head](lessons/19-roberta-encoder-and-mlm-head.md) adds the complete blocks and tied MLM head.

| Boundary | Shape | Meaning to explain before continuing |
| --- | --- | --- |
| Inputs, key flags and MLM labels | `[B,T]` each | IDs, existing keys and selected targets are distinct |
| Token plus learned-position embeddings | `[B,T,32]` | Pad position 1; real positions start at 2 |
| Q/K/V inside each head | `[B,4,T,8]` | Four independent feature groups, one token axis |
| Allowed-key mask | `[B,1,1,T]` | Broadcast over heads and query rows; no causal triangle |
| Encoder outputs | `[B,T,32]` | Contextual vector at every input position |
| Tied output table multiplication | `[B,T,32] × [32,V]` | The same Parameter table serves input lookup and vocabulary scoring |
| Vocabulary logits | `[B,T,V]` | One score per possible token at each position |

Run one forward/loss/backward probe before introducing a trainer. Check finite gradients in the backbone and head, equality of the explicit attention calculation with SDPA, and identity of the tied weight object. In evaluation, changing a blocked trailing key must leave real-token logits unchanged; changing real right context may alter an earlier prediction because this encoder is bidirectional.

Use the loss sum and selected count together. Uniform logits over `V` tokens have mean selected-token loss `log(V)`; adding ignored positions leaves that mean unchanged. An all-ignored batch has count zero and must not produce an optimizer update.

## Train and inspect one-token completion

Complete `roberta_train.py` as taught in [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md). Keep sample selection and corruption generators explicit. Prepare all microbatches in one accumulation group, sum their selected counts, backpropagate each loss sum over that common denominator, then clip/update/schedule once.

```bash
python roberta_train.py --task mlm --steps 120
```

Inspect the before/after fixed validation corruption, its numerator/denominator and the printed one-token fill candidates. The candidates are vocabulary rows; a row can represent a byte fragment, subword or whole word. An explicit `<mask>` represents one token in this inference function, not an arbitrary-length missing phrase.

The recorded local author reference reduced fixed validation loss from approximately 5.85086 to 3.83518 and improved selected-token correctness from 0/8 to 2/8. The model had 39,957 parameters. These are small author observations on the declared fixture. Eight selected validation targets do not establish robust language-model quality.

For your own run, retain its original stopping/acceptance rule and failed measurements. Before increasing the budget or changing the corpus, define which held-out documents, corruption policy, metrics and task threshold will count as success. Do not lower a threshold after seeing a failed run.

## Separate transfer from interrupted-training recovery

For task transfer, export the trained encoder with its configuration and tokenizer:

```bash
python roberta_train.py --task mlm --steps 120 --export-backbone /tmp/course-roberta-backbone.pt
```

The exported file is a backbone transfer artifact. It deliberately lacks optimizer moments, scheduler state, RNG state and completed-update position. It cannot continue interrupted pretraining exactly.

For full continuation, use [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md) and [course_checkpoint.py](examples/course_checkpoint.py). Save after a completed update, keeping at least two verified generations. The identity includes tokenizer/corpus/configuration, schedule and optimizer protocol, source digests and runtime semantics. Save both `mask_generator` and `batch_generator`, global RNG states and the completed update count. Fresh initialization may consume randomness before loading; recovery must replace it afterward.

```bash
python course_recovery_checks.py --project roberta
```

The recovery program compares uninterrupted and fresh-process continuation for both MLM and QA. It also exercises corrupt/incomplete generations and compatibility rejection. Consult the course [reference-run record](project/reference-runs.md) for the actual author result. Your own recovery evidence must identify the checkpoint and exact next updates; a successful save or deserialization alone is insufficient.

Never force-load an incompatible generation to make a resume succeed. A changed schedule closure, tokenizer merge order, corpus or dtype/device can change the next update despite matching parameter shapes. Starting a changed experiment is a separate profile.

## Create QA labels before learning endpoints

Follow [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md) to extend `roberta_data.py`. Validate each answer's original character slice, then build `<s> question </s> </s> context </s>` windows with context-only offsets.

Check adjacent half-open character boundaries, a multibyte answer, a span beyond the first window and a span that cannot fit. Character intervals are `[start,end)`; token endpoint targets are inclusive. A missing answer in one window is ignored with two `-100` labels. It is not turned into a fake `(0,0)` answer or an unanswerable-question label.

Choose overlap smaller than the remaining context token budget. The question and four special tokens consume space before context is added. Require every training answer to be fully represented in at least one retained feature. Records with empty/invalid answers need an explicit task policy; this answerable-only baseline does not teach abstention.

## Transfer the compatible backbone and fine-tune QA

Add the QA head and strict loader from [Lesson 29 — QA endpoint head, loss and fine-tuning](lessons/22-qa-head-and-fine-tuning.md). The backbone output `[B,T,D]` is multiplied by a `[D,2]` head to produce start/end scores. Cross-entropy classifies positions separately; the total is the mean of the two endpoint losses.

```bash
python roberta_train.py --task qa --steps 120 --backbone /tmp/course-roberta-backbone.pt
```

The loader requires the same own-model configuration and tokenizer identity. Same vocabulary size does not imply same token-row meaning. The source model's simple positions and untied MLM head, our changed tiny implementation, and official Hugging Face RoBERTa are distinct checkpoint contracts. There is no tested official-to-tiny weight conversion. Use an official tokenizer/backbone together in a separately declared profile if you explore that route.

The recorded combined author run reduced training-fixture QA loss from approximately 4.18688 to 0.01451 and returned all eight training answers exactly. The independent check requires loss below 0.1 and eight exact fixture answers. This is an overfit/mechanics gate on seen data, not a held-out QA quality gate.

## Decode and evaluate original records

Implement [Lesson 30 — constrained span decoding and QA metrics](lessons/23-span-decoding-and-metrics.md): consider context-only spans, enforce `start≤end` and a documented maximum answer length, rank each valid pair, then slice the original context's characters. Across overflow windows retain their original record ownership. A raw-logit window ranking is a heuristic, not calibrated confidence.

Inspect these failures explicitly:

- The largest endpoint scores select a question word or reversed pair.
- An answer occurs only after the first window.
- A gold answer exceeds the maximum permitted span length.
- Accent or punctuation boundaries expose offset mistakes.
- A paragraph contains no answer, but an answerable-only model still returns a span.

Report normalized EM and multiset-token F1 per original question, taking the best over acceptable gold strings before averaging. Report their units, decoder settings and representative original prediction/reference strings. Keep validation/test contexts and questions separate from QA training.

## Keep evidence states separate

| Gate | Evidence required | Current preparation state |
| --- | --- | --- |
| Data/model correctness | Own code plus independent branch, mask, position, gradient and span checks | Authored references checked locally; learner implementation pending |
| Complete small training | MLM and QA updates with declared input/config/runtime and counts | Authored CPU runs completed; learner runs pending |
| Small mechanics check | Fixed MLM observation and QA seen-fixture sanity result | Recorded above; limited to the fixture |
| Held-out task quality | Predeclared split/protocol/threshold, actual EM/F1/errors and MLM measurements | Larger learner evaluation pending |
| Explanation | Trace the tensors, masks, objectives and failures without relying on function-name memorization | Material prepared; human explanation unassessed |
| Full persistent recovery | Same next updates after fresh runtime; failure/fallback and retained-generation proof | Shared reference check available; learner/hosted proof pending |
| Colab Free feasibility | Thin source runner, measured pilot, durable checkpoint and fresh hosted runtime | Procedure prepared; hosted execution unverified |

Optional GPU execution uses the same source functions with completed CPU batches moved to the model's parameter device. The CPU path is verified; GPU behavior and cross-device replay are not established by this course's author run.

Before the GPT course, complete the relevant rows in [readiness](project/readiness.md), rehearse the [model explanation](practice/explain-your-model.md), and try the [changed-case questions](practice/questions.md). Return to the first missing implementation or evidence gate rather than marking completion from reading alone.
