# Detailed course contract and second pass

Lead: workspace developer-learning-tutorials; planning/AI project workflow owns scope/evidence and feature-whiteboard-defense supplies explanation practice. The deliverable is a tracked tutorial/reference and reader entry, not creation of a learner model repository.

## Ordered teaching contract

| Previous result | Next problem and lesson | Implementation/check |
| --- | --- | --- |
| P00 source-conditioned model | Score a text prefix; Lesson 1 | Shift/conditional probability/NLL calculation |
| One clear prediction task | Give it a reproducible home; Lesson 2 | Config/head invariant, runtime and handoff |
| Model/data ownership | Fix which text can be learned; Lesson 3 | Pinned Alice chapters and frozen tokenizer |
| Frozen documents | Produce bounded predictions; Lesson 4 | Unique targets, PAD/EOS and sampler state |
| Correct inputs/labels | Enforce allowed information; Lesson 5 | QKV/head layout, explicit attention and independent gradient reference |
| Causal update | Assemble full GPT; Lesson 6 | Learned positions, pre-norm/GELU/tied logits and model tests |
| Untrained logits | Learn from errors; Lesson 7 | Stable NLL, token weighting, AdamW/clipping/schedule and two-string fit |
| A fitting model | Assess unseen behavior; Lesson 8 | Token-weighted evaluation, unigram and bounded seeded generation |
| Useful update state | Survive a failed save; Lesson 9 | Verified immutable generations, older fallback and retention |
| Persistent store | Run one experiment across sessions; Lesson 10 | Common trainer, explicit new/resume, all-state parity |
| Reproducible local run | Survive a deleted hosted runtime; Lesson 11 | Safe allowlisted source export, thin notebook and actual hosted gate |
| Working pipeline | Explain resources and transfer; Lesson 12 | Parameter/storage derivation, scoped profile and P02 exit |

## Maths at first use

Conditional probability/log products/perplexity appear in Lesson 1; indexing/rectangular storage in 2–4; head contractions/variance scaling/masked softmax/finite differences in 5; mean/variance, residual derivative paths, learned rows and Gaussian/tanh GELU in 6; log-sum-exp/p-minus-target, weighted gradients, Adam moments, norms and cosine LR in 7; denominator/tokenization/sampling odds in 8; parameter/storage powers and measurement scope in 12. Each row has a tiny calculation and a prediction/practice item. Recovery's state assumptions are spelled out before implementation.

## Exact implementation boundary

Reference: 120832 parameters, full two-layer pre-norm GPT, byte tokenizer baseline; no pretrained compatibility, cache, AMP, workers or cross-device parity promise. Store the source under the tutorial's examples/, expose passive downloads, and embed complete files in the actual lesson sequence. Hash existing tutorial packages before editing; never modify their source for this new course. Source/weights/corpus run outputs stay ignored; compact selected proof is tracked.

## Second-pass findings and repairs

- Corrected the workspace command to actual `./ai new project ...` syntax after inspecting CLI help.
- Repaired snippet boundaries that initially stopped inside attention/evaluation definitions; every intermediate file now parses and staged checks run before future files exist.
- Deferred the sampling CLI's checkpoint import until invocation, so the pre-checkpoint lesson can demonstrate generation.
- Split Trainer into configuration/initialization/update/validation-and-state units with explanation and checks around them.
- The staged notebook export initially required a profiling file taught in the next lesson. Removed it from the training-source allowlist; export now succeeds at its teaching point.
- Expanded corruption handling to reject unloadable state and incompatible manifest schema, with a regression fixture and fresh reference verification.
- Visual inspection found overlapping nonadjacent arrows. Reordered the pre-norm board as a four-corner grid so the direct and sublayer paths remain visible. Simplified the data board's arrows while retaining the train-only fit and separate-partition encoding explanation.
- Kept all remaining abrupt-process/hosted/learner gates explicit; local success does not replace these.

The disposable verification reconstructs all embedded code in order, copies only available download dependencies, checks each stage, runs local new/resume, exports/compiles the notebook, and compares syntax trees with the reference files. Reader and visual evidence are recorded in 06-review.md.
