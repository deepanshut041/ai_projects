# Topic rewrite — Lessons 21–25 author review

Date: 2026-10-12. Scope: the last five topic lessons in the saved 25-lesson outline. Authored in order, after the earlier foundation lessons. Parent owns metadata, the source/probe integration, notes/export updates and complete reader verification. Learner P00-M1 remains Lesson 2 setup complete with `linear()` next; hosted recovery remains pending.

## Authored files

- `lessons/21-regularization.md`: dropout expectation/variance and gradient scaling; explicit residual/input/attention placements; smoothing conventions and PAD-output policy; coupled/decoupled decay; early-stopping selection; `eval()` versus `no_grad()` probe.
- `lessons/22-encoder-decoder.md`: conditional autoregressive factorization, source/decoder streams, three attention roles, exact shifted targets, teacher forcing, EOS/PAD/mask roles, counted loss probe, greedy counterexample, beam scoring/stopping limits. Includes native `seq2seq-contract` diagram marker.
- `lessons/09-training.md`: Lesson 23, completely rewritten around evaluation/debugging and the actual shared classifier; baselines, split/domain limits, NLL/counts, perplexity, tiny-overfit and full training probes, confidence/calibration limits, translation BLEU reporting and targeted diagnoses.
- `lessons/10-recovery.md`: Lesson 24, completely rewritten around state transitions, Adam continuation, update boundaries, complete state, RNG/sampler position, verified generations/fallback/identity, fresh-object corruption probe and separate-process replay suite.
- `lessons/11-colab-and-handoff.md`: Lesson 25, completely rewritten around paper equation/shape mapping, original choices, full miniature batch, parameter/storage/compute calculations, paper section 4, paper inference/checkpoint averaging choices, shared portable source, deleted-runtime procedure and concrete readiness.

Each lesson defines its topic goal and prerequisites, gives original worked numbers, links actual reference code or provides a complete probe, identifies changed-condition failures, uses its assigned native practice ID and asks specific derivation/connection notes. Existing filenames for Lessons 23–25 preserve compatibility identities. No implementation child, later course, manifest, website/service or shared source file was edited by this agent.

## Numerical and source-contract checks executed

Frozen interpreter: the attention child's existing `.venv/bin/python`, Python 3.12.13 / PyTorch 2.14.1+cpu / NumPy 2.5.4, FP32 CPU, one thread for training. This uses the existing environment; it does not change the child.

- Lesson 25's complete Python block passes: MHA 288 and FFN 552 entries; miniature specified total 2,138 parameters; explicit attention score arrays 148 scalars / 592 FP32 bytes; paper-style schedule values 0.0625, 0.25, 0.125 at steps 1, 4, 16 for D4/warmup4.
- The four-corner diagnostic described in Lesson 23 fits 4/4 after 1,000 fixed-batch updates. Initial loss 0.7676748037, final loss 0.0001392846461; 4,000 record presentations. This is a deliberate diagnostic run, not a resumed sampled experiment.
- The actual trainer CLI's complete new run reaches update 1,000 and validation 36/36 with NLL 0.0001299955442. It reports 42 FP32 parameters / 168 parameter bytes and a verified update-1,000 generation. Measured CLI process peak RSS 313,257,984 bytes; trainer elapsed 0.4446269680 seconds in this local run. Neither number is Colab/GPU evidence.
- Two separate actual CLI sessions with a fixed goal 1,000 and 25-update slices restore update 25 and finish at 50.
- The smoothing calculation independently evaluates to 0.6485614662970294. The initial probe rounding was corrected to a compatible asserted value/tolerance.
- Local package-relative links for all five lessons resolve; paired display-math fences, absence of control characters/tabs and scoped Git whitespace checks pass. Aligned equation row separators and Bash continuation backslashes were checked/corrected separately.

The independent coverage-review agent reviewed all five lessons against the source. It confirmed the regularization, target/mask, numerical/evaluation, recovery and paper comparison contracts, requested correct aligned rows and a clearer Course 2 sentence, and requested the original inference/checkpoint-averaging note. Those corrections are applied. It independently ran each complete standalone Python block for Lessons 21–25 in a fresh process; all five pass. Its outputs/digests are saved in `11-inline-checks.json` alongside the cumulative 26/26 all-course snippet result:

- Lesson 21: smoothed loss 0.6485614662970294; expectation/variance and module/gradient-mode assertions pass.
- Lesson 22: six valid target labels; uniform mean NLL 2.0794413884480796; masks/shift/ignored-gradient checks pass.
- Lesson 23: validation loss 0.7687182426452637 and 18/36 before training, then 0.00012999554423408376 and 36/36 at 1,000 updates / 16,000 record presentations.
- Lesson 24: identical continued update 4 and corrupt-newest fallback to update 3, comparing full recorded state and update report.
- Lesson 25: 2,138 parameters, 592 explicit score bytes and rates [0.0625, 0.25, 0.125].

## Source inspection and integration IDs

All URLs below were inspected on 2026-10-12 through web browsing, not inferred from a current search snippet alone unless stated. No source prose, examples or diagrams were copied.

| Source ID | Inspected primary URL and relevant coverage |
| --- | --- |
| `transformer-paper` | https://arxiv.org/html/1706.03762v7 — sections 3–5 architecture/attention/positions/training; section 4 complexity/dependency comparison; section 6.1 inference, checkpoint averaging and beam settings. |
| `dropout` | https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html — Bernoulli/inverted scaling, evaluation identity, shape. |
| `cross-entropy` | https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html — logits/target/reduction contract, uniform-mixture smoothing and ignore-index scope. |
| `torch-nograd` | https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html — gradient-recording mode, distinct from module evaluation behavior. |
| `torch-modules` | https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html — module evaluation/mode contract. |
| `checkpoint-docs` | https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html — model versus general training checkpoint and optimizer state; tutorial currently reports PyTorch 2.14. |
| `randomness` | https://docs.pytorch.org/docs/2.14/notes/randomness.html — same-environment randomness controls and limits across releases/platforms/CPU-GPU. |
| `d2l-seq2seq` | https://d2l.ai/chapter_recurrent-modern/seq2seq.html — D2L 1.0.3 conditional sequence learning/teacher forcing. |
| `d2l-beam-search` | https://d2l.ai/chapter_recurrent-modern/beam-search.html — D2L 1.0.3 bounded beam search/greedy comparison context. |
| `bleu-reporting` | https://arxiv.org/abs/1804.08771 — version 2 abstract inspected; reference tokenization/normalization affects score comparability. |
| `sacrebleu` | https://github.com/mjpost/sacrebleu — official README, corpus/metric signatures, detokenized inputs, reference/metric configuration reporting. |
| `colab-faq` | https://research.google.com/colaboratory/faq.html — variable/temporary resources, runtime deletion and Drive behavior. |

Unversioned PyTorch `stable` Dropout/Module links returned only redirect pages; explicit 2.14 URLs were read. The Google Research page for Rethinking the Inception Architecture was also opened, but the lesson's smoothing convention is grounded in the inspected CrossEntropyLoss API and its original authored calculation, so no extra manifest ID is needed.

## Four-pass review and remaining ownership

- Topic/continuity: the lessons own regularization, sequence contracts, measured training, recovery and paper assembly; no stretched prediction story or extra mandatory route.
- Maths: defined axes/distributions/state; dropout variance, smoothing gradient, conditional factorization, NLL/perplexity, Adam history, parameter counts, score storage and compute units have conditions and worked calculations.
- Implementation: commands/probes use existing exact API names, `examples/` working directory and frozen interpreter; precision/schedule/source profiles are explicit. The trainer uses warmup then constant rate, not the paper schedule. Tiny grids/probes are not asserted as translation implementations.
- Readiness: source/author checks, learner implementation/explanation, task/domain evidence, fresh local continuation and fresh hosted storage survival stay distinct. Same-width alternative designs, averaged inference weights and exact training state are not conflated.

Parent still owns all-course metadata/source IDs, frozen rendering, notebook/export integration, root/prefix/browser/visual review and final preservation checks. No hosted execution, commits or external publication was performed.
