# My Tutorial 0 foundations notes

I will save my own copy and use one section per topic. The full explanations and native exercises are in the course; these are my working notes, not prefilled answers or a completion record.

For each topic I will define the symbols/axes, derive a step, calculate an example, predict a changed condition and connect it to the Transformer. Optional cleared study probes import the same Python helpers.

## Lesson 1 — Linear algebra and neural maps

A legal multiply can still join the wrong axes. Using equal dimensions hides mistakes, so debug with unequal values for N, D and F. An elementwise product is not a matrix multiply. Bias belongs to output features; a bias per example would depend on batch organization.

In my notes: write the indexed affine formula, calculate every cell of one result row, and label which numbers are data and which are shared parameters. Explain how the same map will produce Q, K and V in attention. Stop when you can derive the output and parameter count without running the reference.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 2 — PyTorch tensors, storage and module state

Write the coordinate, dtype/device and storage contract for an ID tensor, its embeddings and a head-split tensor. Predict the transpose result before running it. Explain why a same-shaped tensor can still have the wrong axes. The runnable tensor checks are test_prerequisite_tensors.py; use the configured Torch interpreter.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 3 — Derivatives, gradients and local descent

Derive both partials from intermediate quantities, label what is held fixed and calculate the two different SGD updates. Explain how Jᵀg becomes a shared parameter's gradient. Use prerequisite_math.py for central differences, not as a substitute for the derivation.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 4 — Probability, statistics and information

Write the sequence factorization and state where an independence assumption is used. Compute an expectation, a variance and all three information quantities for the same distribution. Explain why an attention row and a vocabulary distribution can both be probabilities yet index different things.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 5 — Feedforward networks and nonlinear activations

Draw the affine–activation–affine graph and annotate every shape. Derive the affine-collapse identity, reproduce the controlled hidden/logit values and explain the quadrant contradiction. For the existing lab, attempt selected numerical units rather than copying the entire reference network into your learner file.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 6 — Initialization, symmetry and signal scale

Derive the variance calculation and list every independence/centering assumption. Calculate scales for a 128→512 layer. Inspect activation standard deviations and gradient norms across layers instead of assuming an initializer proves healthy signals. Explain both identical-unit symmetry and attention's sqrt(d) denominator.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 7 — Backpropagation and automatic differentiation

Derive all four parameter gradients and the input gradient, annotate upstream versus local derivatives, and explain the zero first row of ∇W₂. Compare one coordinate with a central difference before inspecting Torch. Explain how repeated embedding IDs will use the same “sum all paths” rule.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 8 — Loss functions, logits and stable arithmetic

Derive p−one_hot from log-sum-exp, compute both target changes and identify the exact denominator for a token batch. Compare BCE and categorical target contracts. The stable references are prerequisite_tensors.py and deep_learning_math.py, with extreme-value and gradient checks.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 9 — SGD, momentum, Adam and weight decay

Calculate the two moment updates, explain why the zero-gradient step moves and derive the SGD equivalence before explaining AdamW's difference. List which optimizer state a checkpoint needs.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 10 — Training updates, schedules and gradient scale

In my notes: draw the update boundary, derive the unequal-micro-batch denominator, calculate all three schedule values and compare global with coordinate clipping. The actual loop is prerequisite_mlp.py, with checks in prerequisite_mlp_checks.py.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 11 — Datasets, batching and supervision contracts

In my notes: draw two ragged records, their padded IDs, valid-key mask and ignored labels. Calculate both record and token means and say which objective the training code chooses. List the state needed to reproduce the next batch.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 12 — Text encoding, tokenization and vocabulary identity

Draw `text→normalization→bytes/pieces→IDs` and label every coordinate. Hand-run both BPE merges, state the document-boundary rule and explain why the frozen artifact must travel with a checkpoint. Compute the correct byte and code-point intervals for a changed sentence.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 13 — Embeddings, lookup gradients and weight tying

Write the one-hot equivalence, calculate the repeated-ID gradients and draw both gradient paths into a tied table. Count tied and untied parameters. Explain the distinction between token ID, lookup row, position-enriched input and contextual representation.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 14 — Sequence order and positional encoding

Derive the permutation relation and pairwise shift identity, calculate one table row and draw addition versus concatenation. Explain why “fixed” does not imply proven length extrapolation.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 15 — Queries, keys, values and scaled attention

Write all shapes, compute the changed-value output and derive the row-softmax backward rule. Explain which input supplies Q versus K/V in cross-attention. A large attention weight is a model computation, not by itself a causal explanation of a prediction.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 16 — Padding masks, causal masks and empty rows

Draw allowed matrices for encoder, decoder and cross-attention for S=4,T=3 with padding. Calculate one renormalized row and explain the empty-row policy. Predict which invariance check exposes an inverted mask.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 17 — Multi-head projections, splitting and concatenation

Draw the split/transpose/merge path with unequal T,S. Count parameters at D=8,H=2 and compare H=4. Explain why Wo is still needed after concatenation and why a reshape before the inverse transpose scrambles positions and heads.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 18 — The position-wise feedforward sublayer

In my notes: work both positions by hand, show parameter sharing, calculate the count and compare the leading attention/FFN operations.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 19 — Residual connections, depth and normalization order

Draw both branches, derive I+J_F and produce a cancellation counterexample. Work the zero-sublayer pre/post-norm case and name the residual stream in cross-attention.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 20 — Layer normalization and its gradients

Calculate both rows, label every reduced axis and derive the two correction terms in ∇x. Compare `LayerNorm(D)` with a whole-sequence normalization and explain train/eval behavior. Reference arithmetic is in prerequisite_tensors.py.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 21 — regularization and generalization controls

In my study notes, derive dropout variance for $a=-3,p=0.2$, then explain which residual path remains when the sublayer is dropped. Write both smoothing conventions for a three-class target, and predict whether a `no_grad()` call alone makes a training-mode Transformer deterministic. Connect each proposed regularizer to the validation evidence that would justify it.

Continue with encoder–decoder learning and inference, where these controls must operate on precisely shifted sequence streams.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 22 — encoder–decoder learning and inference

In my study notes, shift the target `[9,8]`, count its supervised positions and draw its legal decoder self-attention matrix. Annotate one cross-attention row when the source has five real tokens plus two PAD positions. Explain why teacher-forced token accuracy and free-running sequence accuracy can diverge: generated mistakes change the later prefixes the model sees.

Continue with evaluation and debugging, applying the same discipline to a complete small training experiment before a full translation build.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 23 — evaluation, debugging and a complete small training experiment

In my study notes, write the task/domain/split before recording results, calculate the constant-class and uniform-loss baselines, and retain one failed or deliberately corrupted-case diagnosis. Explain what a completed tiny training run proves and what remains unmeasured for translation. Keep author checks, your implementation, your explanation and hosted capability as separate evidence.

Continue with checkpointing and reproducible recovery, preserving the exact state that produced the next update.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 24 — checkpointing and reproducible recovery

In my study notes, list the state needed for the **next** batch and update, then draw the generation's write/verify/complete/rotate order. Predict the restored update after a failed newer save, distinguish warm start from resume, and record which fresh-process comparison you actually performed. Hosted persistence remains pending until Lesson 25's deleted-runtime drill succeeds.

Continue with paper assembly and the implementation handoff.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## Lesson 25 — read the paper and assemble the foundations

The source exporter packages the same classifier/checkpoint sources and numerical/attention probes used locally; it excludes environments, data, weights and generated runs. From `examples/`, choose fresh destination filenames:

The report prints archive/member hashes and identifies the actual working-tree source snapshot. The cleared study notebook follows the topic lessons and imports source probes; its explicit export option refuses to overwrite an existing personal notes notebook. Keep your edited copy. The separate thin Colab notebook performs source verification, setup/checks and trainer calls rather than duplicating implementation code.

### My explanation and notation

- Main idea:
- Equation, indices and axis meanings:
- Assumptions and limitations:

### My calculation and changed-condition prediction

- Hand calculation:
- Prediction before running or revealing:
- Observed result and what I corrected:

### My connection to Attention Is All You Need

- Where this is used:
- Mistake I can now diagnose:
- Question I still need to resolve:

## My return-to-work record

- Topic and exercise I stopped at:
- My source/environment identity:
- My next calculation or check:
- Evidence still missing:
