# Explain both models on a board

This is a reference-design and code-trace study for the two authored miniature builds. The attention implementation in the existing translation project has its own verified history; reading these explanations does not complete either learner model. Use the board to show information movement, tensor meanings, training targets and state owners. Function-name memorization is unnecessary.

Start from [Lesson 1 — Model specifications and task contracts](lessons/01-start.md). For source and proof boundaries, consult [sources and choices](research.md) and [reference runs](project/reference-runs.md). When you want a live rehearsal, answer one question before opening its worked explanation; these documents record no human mastery result.

## Tell the translation story in two minutes

Draw an English source above a French target prefix. Start with a concrete target sequence `[BOS, je, vois, EOS]`. During teacher-forced training the decoder input is `[BOS,je,vois]` and the labels are `[je,vois,EOS]`. The vocabulary prediction at each position must use only the target prefix available there.

Draw source IDs → source embeddings plus sinusoidal positions → encoder blocks → source memory. Label memory `[B,S,D]`. Each encoder attention query can use every real source key; padding columns are blocked. Draw target IDs → target embeddings plus positions → decoder self-attention → cross-attention → FFN → target vocabulary head. The decoder's first attention is causal; cross-attention queries come from the target and keys/values from source memory.

For a board with `B=2,S=4,T=3,D=8,H=2`, write head width `d=4`. Self-attention Q/K/V use `[2,2,T,4]`; source cross-attention K/V use `[2,2,4,4]`. Cross scores have shape `[2,2,3,4]`, which makes the query/key distinction visible. Multiply softmax weights `[T,S]` by source values `[S,d]` to obtain context `[T,d]` per head. Merge heads and project back to width 8.

Finally draw shifted labels → token cross-entropy → backward → optimizer update. Ignore padded labels with `-100`; average over actual supervised target tokens. At inference there are no gold target tokens to feed: begin with BOS, select the next token, append it, and stop at EOS or the declared limit. The current reference recomputes target-prefix attention; it does not maintain a KV cache. Finished rows are padded while unfinished rows continue.

The source implementation and tiny fixture use GELU/post-norm choices described in the lessons. These choices differ from the original paper's full training recipe. Do not infer paper-scale quality from the tiny architecture being complete.

## Tell the RoBERTa story in two minutes

Draw uncorrupted text → byte-BPE IDs → sampled corruption beside a protected copy of the original labels. Explain the three separate flags: padding keys, eligible ordinary tokens and selected MLM targets. A selected unchanged input still has a label. An inserted mask remains an allowed key. Special tokens are excluded from target selection.

Draw token embeddings plus learned position rows → bidirectional encoder blocks → contextual vectors `[B,T,D]`. For pad ID 1, write `[0,7,2,1] → positions [2,3,4,1]`. Every real query may use right and left context; there is no causal triangle. Use the same Q/K/V and per-head arithmetic as the translation encoder.

Draw dense → GELU → LayerNorm → tied vocabulary table. The table is one shared Parameter: input lookup uses rows of `[V,D]`; output scoring uses its transpose `[D,V]`. The logits are `[B,T,V]`. The MLM loss averages only original labels at selected positions. A batch with zero selected targets must not advance the optimizer/scheduler.

Then replace the vocabulary head with a fresh two-column QA head while transferring the compatible encoder. Draw `<s> question </s> </s> context </s>`, original character offsets, overflow windows and gold inclusive token endpoints. Hidden states `[B,T,D] × [D,2]` produce start/end scores `[B,T]` each. Train two endpoint classifiers; at inference search context-only, ordered, bounded spans and slice the original context characters.

The own tiny tokenizer and checkpoint cannot interchange with official RoBERTa weights. Same shapes do not establish row meanings or position conventions. Seen-fixture QA success is an overfit check; it is not held-out SQuAD evidence or a demonstration that the model can abstain on unanswerable questions.

## Put each source file on the execution path

| Component | Translation reference | RoBERTa reference | Boundary to annotate |
| --- | --- | --- | --- |
| Tokens and batches | [translation_data.py](examples/translation_data.py) | [roberta_data.py](examples/roberta_data.py) | IDs, specials, masks, shifted targets or selected/span labels |
| Attention | [translation_attention.py](examples/translation_attention.py) | [roberta_model.py](examples/roberta_model.py) | Q/K/V head axes, allowed keys, scale and output projection |
| Complete model | [translation_model.py](examples/translation_model.py) | [roberta_model.py](examples/roberta_model.py) | Encoder/decoder memory versus bidirectional encoder and heads |
| Training | [translation_train.py](examples/translation_train.py) | [roberta_train.py](examples/roberta_train.py) | Loss sums/counts, clipping, optimizer/scheduler and device transfer |
| Metrics | [translation_metrics.py](examples/translation_metrics.py) | `answer_metrics` in [roberta_data.py](examples/roberta_data.py) | Metric normalization, denominator and output units |
| Recovery | [course_checkpoint.py](examples/course_checkpoint.py) | Same shared helper | Identity, full state, generation completion and fallback |

Use a specific configured batch rather than saying “attention happens.” At each multiplication, name both axes being contracted and the resulting axes. At each mask, name which information is excluded. At each loss, identify the original supervised target and reduction denominator.

## Trace a failure all the way to the update

### Translation: the target was never shifted

Write decoder input `[BOS,je,vois,EOS]` and identical labels. A causal mask allows a token to read itself, so the model can learn the current visible token rather than the next one. Training loss can fall while generation fails. Repair alignment to input `target[:-1]` and labels `target[1:]`, retain pad ignoring, then rerun the independent shift and future-token checks. Fixing only the causal mask does not fix this leak.

### MLM: labels were copied after corruption

Original token `cat` is replaced with a random input `dog`. If the label is copied after replacement, the target becomes `dog`. The forward pass and cross-entropy still produce finite numbers, but they train the wrong task. Copy original labels before mutation and test all three replacement branches with controlled draws. A generic “loss decreases” check would not catch this error.

### QA: an absent window answer became token zero

An answer appears after the first retained context window. Setting endpoints `(0,0)` trains the beginning token because zero is a valid class index. It does not ignore the window. Use explicit ignored labels for absent windows and check that some overflow feature contains the full annotated answer. A separate no-answer task requires its own trained convention; it cannot be inferred from ignored windows.

### Recovery: only weights survived

Two runs start from equal model weights, but one has fresh Adam moments, scheduler position and corruption generators. Its next gradient/update need not match the uninterrupted run. Restore the full compatible generation and compare actual next batches/updates in a fresh process. The checkpoint completion marker is published last so an interrupted write cannot replace the last usable generation.

## Explain the memory and state lifetimes

| State | Lifetime and owner | What happens at an interruption |
| --- | --- | --- |
| Model parameters and nontrainable buffers | Persist across updates; model owns them | Full checkpoint restores them |
| Adam moments and scheduler position | Persist across updates; optimizer/scheduler own them | Reinitializing changes future updates |
| Activations and current gradients | Live during a forward/backward/update group | Reference saves only after completed updates; redo lost work |
| Sample/corruption RNG | Determines future data/MLM corruption | Restore explicit and global generators |
| Translation source memory | Produced for one inference request | Recompute from that request's source after restart |
| Generated target prefix | Grows within one decoding request | Not a model training checkpoint field |
| QA feature offsets/windows | Belong to one original question/context | Rebuild using the same tokenizer/policy |

For one FP32 attention score tensor `[B,H,T,S]`, storage is `4BHTS` bytes. With `B=2,H=2,T=3,S=4`, it contains 48 scalars or 192 bytes, excluding Q/K/V, probabilities, gradients and allocator overhead. Doubling both sequence axes multiplies this score storage by four. Efficient SDPA can avoid materializing the same intermediates in some backends; this arithmetic does not prove the CPU run used a particular fused kernel or establishes GPU speed.

## Defend choices with conditions

| Choice | Present purpose | Consequence and a reason to revisit |
| --- | --- | --- |
| Separate source/target vocabularies | Each translation language has its own ID contract | IDs and pad values must not be swapped; a joint multilingual tokenizer would require a new recorded profile |
| Post-norm blocks | Follow the inspected implementation family | Normalization occurs after residual addition; another order changes execution and training behavior |
| Tied MLM table | Reuse token rows for lookup and output scoring | Shared gradients and fewer unique parameters; compatibility must be checked when transferring weights |
| Dynamic MLM corruption | Redraw selected targets on presentation | Save corruption RNG for exact continuation; fix evaluation corruption for comparable measurements |
| Overlapping QA windows | Recover answers beyond the first context budget | More features and duplicate candidates; long answers can still be excluded |
| Bounded greedy translation/span search | Small inspectable inference path | No beam-search or calibration guarantee; inspect task errors before introducing a more complex decoder |
| CPU miniature profile | Run complete mechanics locally and prepare Colab Free recovery | It does not establish paper-scale quality or hosted/device performance |

Distinguish the source's documented behavior, the reference's tested implementation, and a suggested alternative. Saying why a choice is useful today is not evidence of the original author's intention.

## Finish with evidence rather than confidence

Name the exact source/profile, one independent check that could fail for a plausible bug, the actual observed task result, and the next unproved claim. You should be able to explain why each model is complete as a miniature reference while its learner implementation, held-out quality and hosted gates remain separate.

Try the [changed-case questions](practice/questions.md), then fill the relevant evidence row in [readiness](project/readiness.md). A prepared drawing is study material; a human explanation becomes evidence only after the learner attempts it.
