# P01 author reference runs

These are local author checks of the downloadable examples. They are not learner milestone completion, hosted Colab proof, a GPT-2 reproduction or a claim of useful general language ability. Full compact identities/results: [reference-results.json](project/reference-results.json).

## Runtime and protocol

Python 3.12.13, Torch 2.14.1+cpu, AMD Ryzen 9 9955HX, one Torch thread, FP32. Main model: L2/D64/H4/F256/T64/V260, dropout 0.1, 120832 parameters. Train chapters I–IX, validate X, test XI–XII not evaluated. AdamW peak LR .001/floor .0001, 20 warmup/300 total updates, batch 8, seed 17. Two sessions of 150 updates use the same fixed schedule and persistent run identity.

## Executed checks

- Twenty-one reference unit tests pass: data/artifact/shift boundaries, independent attention values/gradients and finite differences, causality/padding/model bounds, tied parameters/loss gradients, unequal accumulation, recovery/RNG/retention/failure guards, evaluation and generation.
- Fresh CPU processes: uninterrupted 8 updates versus 4 + exit + 4 restore with dropout 0.1; complete saved state compares exactly, including model/Adam/RNG/sampler/schedule/progress/monitoring.
- Two-string fitting with dropout 0, constant LR .003, clip 1 and 300 updates: NLL 5.596348 → 0.082698; declared target <0.1 passed. This is memorization, not held-out evidence.
- Source export creates an allowlisted ZIP with identity manifest and an output-cleared thin notebook; all code cells compile locally. No hosted execution has been performed.

## Main text demonstration

| Measurement | Actual result |
| --- | --- |
| Initial validation NLL / perplexity | 5.585223 / 266.459634 |
| Final validation NLL / perplexity | 2.415295 / 11.193075 |
| Training-unigram validation NLL | 3.347120 |
| Declared improvement target | At least 0.05 nats below initial; observed 3.169928 |
| Valid labels consumed | 153189 |
| Full state size at update 300 | 1494315 bytes |
| Session boundary | Explicit resume restored 150 and advanced to 300 |

The likelihood checks pass on this frozen chapter protocol. Samples still contain malformed words and repetition. These two observations are compatible: improving next-byte probabilities is easier than learning long coherent prose on this tiny run.

A temperature-0.8 sample with prompt `Alice ` and seed 29 began `weld tthan. arsasatre...`; greedy decoding repeated `the` throughout the 80-token budget. The compact results preserve three sampled outputs plus greedy and their raw IDs. None reached EOS within that budget. Do not select a prettier output and omit these failures.

## Scoped resource measurements

Ten one-thread CPU forward/backward repetitions after three warmups gave medians of about 3.174 ms at T32 and 6.947 ms at T64 on the author host. These exclude optimizer, data acquisition and checkpoint persistence and are not a Free/GPU throughput claim. T32 has fewer learned position parameters. Remeasure in your own child runtime.

Weights at default size occupy 483328 FP32 bytes. Approximate weights+gradients+two Adam moments are 1933312 bytes, excluding activations/temporary allocations. One B8/H4/T64 FP32 score matrix is 524288 bytes. Peak host/VRAM and Drive-save latency were not measured here.

## Limits and next evidence

The reference tests include injected partial-copy/storage failures and corrupt-newest fallback, but not killing a hosted writer or deleting a Colab runtime. Actual child abrupt-process/failure-injection, runtime resource limits, notebook execution and learner explanations remain required in [readiness](project/readiness.md). Test chapters remain untouched for a final protocol. Optional cache, AMP, workers and cross-device continuation are absent, not silently supported.
