# Project 1 starting plan — tiny GPT-1

Status: proposed; no model implementation or run.
Updated: 2026-10-09
Lead: AI project workflow; supporting AI experiment planning for proposed pilots. This umbrella package owns the roadmap review. When Project 1 begins, create/resume its child-owned package before implementation/runs.

Colab amendment (2026-10-09): local correctness remains first; P01 now also requires the [Colab Free training and recovery contract](../../../guide/COLAB.md). This extends the earlier CPU-only starting scope. Shared requirement/evidence ownership is in the [Colab package](../colab-free-training-and-recovery-requirements/00-resume.md); actual notebooks, trainer and recovery proof will be child-owned. Pro is deferred until requested.

## Ownership and proposed files

Proposed independent repository: `ai_projects/projects/gpt-lab/`. It does not exist yet. Use the workspace's `ai new project` command, inspect the child's AGENTS.md and configure manager tasks as argument arrays through its registered path.

Proposed layout:

```text
gpt-lab/
  src/gpt_lab/{config,tokenizer,data,attention,model,losses,train,generate,checkpoint}.py
  configs/{correctness,cpu,colab-free,published-reference}.json
  notebooks/colab.ipynb  # thin source/setup/train/resume/evaluate wrapper
  tests/                 # independent numerical and behavioral checks
  reports/               # authored interpretation and small summaries
  docs/implementation/   # child-owned plan and progress
  pyproject.toml
  uv.lock
```

Datasets, environments, checkpoints, profiler traces and generated runs stay ignored/local. Use original synthetic fixtures for first checks; record generator revision, seed and hashes. Any later real-text corpus needs license/provenance and document-level split rules.

## Source and fidelity contract

Authority: [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), sections 3–4, plus [original source revision](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/train.py). Compare GPT-2 only as a separate later configuration.

Preserve causal attention, learned positions, residual-then-normalization ordering, tanh-approximate GELU, classification-token pooling and tied token/output weights. No GPT-2-style extra final LayerNorm in the GPT-1 variant. Represent token and position tables separately and limit logits to the token vocabulary: the old TensorFlow source combines those rows in a single table. Record that output-domain difference, the byte tokenizer, tiny dimensions, synthetic data and modern training machinery as deliberate departures. This is not historical checkpoint or training-result reproduction.

Published-reference metadata: 12 layers, width 768, 12 heads, FFN 3072, context 512; historical BPE uses 40,000 merges. Merge count is not interchangeable with exact vocabulary size. Never instantiate the published configuration implicitly.

## Model and data contract

| Item | Proposed contract |
| --- | --- |
| Config validation | Positive dimensions; width divisible by heads; context at least 2; explicit special IDs and dropout; reject unsupported settings. |
| Tokenizer | Reversible UTF-8 bytes; 256 ordinary IDs + BOS/EOS/PAD/CLS. Reserve/exclude specials deliberately in text sampling; do not silently treat PAD as text. |
| Inputs and hidden states | IDs [B,T], valid-token mask [B,T], hidden [B,T,D]; FP32 on CPU initially. |
| Attention | Q/K/V [B,H,T,D/H]; score scale by square root of head width; allow only non-padding keys at positions ≤ query position. Right padding; at least one valid token in each sample. |
| Block | Learned embeddings → attention → residual/post-norm → FFN → residual/post-norm. Train/eval dropout behavior explicit. |
| LM output | [B,T,V], no output bias; tied token table; no extra final GPT-2 normalization. |
| LM loss | Logits at position t predict IDs at t+1. Mask PAD targets and invalid pairs; EOS may be a target; classification-format masking is explicit. Reject a loss batch with no valid targets. |
| Classifier | Append exactly one CLS token at the end of content; gather that explicit position, not last padded position. Binary linear head. Missing/duplicate CLS is invalid. |
| Fine-tuning | Classification cross-entropy; optional auxiliary LM coefficient configurable. For this format mask the CLS prediction target. Log both terms; changing the coefficient must change the combined loss as specified. |
| Generation | Uncached autoregressive recomputation initially; explicit BOS for empty input, EOS stop and max-new-token limit. Reject context overflow initially; no silent learned-position rollover. |
| Checkpoint | Versioned config/tokenizer/data/source identity, model/optimizer/scheduler/scaler when used, step/tokens, RNG and next-data/sampler state. Reject incompatible identity. Round-trip eval logits, compare resumed updates, verify persistent immutable generations and fresh-runtime recovery per COLAB.md. |
| Sample boundaries | BOS/EOS delimit independent examples; do not accidentally train across documents or pad tokens. Split before tokenization/windowing for later corpora. |

Proposed configs: correctness = L2/D64/H4/FF256/T64; CPU = L4/D128/H4/FF512/T128, optionally L6 after timing. Both use V260. Exact counts will be taken from instantiated parameters, deduplicating tied weights. Estimate for a binary classifier with biases and affine LayerNorm:
`V*D + T*D + L*(12*D^2 + 13*D) + 2*D + 2`.
Estimates are 120,834 / 843,010 / 1,239,554 for the three designs.

## Hands-on learning contract

The learner needs to implement, train, inspect and debug, not simply read a finished solution. Work in small buildable steps. The assistant can prepare scaffolding, numerical references and review; preserve selected component/training tasks for the learner.

At each milestone:
1. Draw shapes and predict the result before running.
2. Implement a meaningful piece, including attention/masking, post-norm blocks, loss alignment or the training loop.
3. Run checks and a bounded training exercise; inspect actual activations, gradients and loss.
4. Explain a failure case and the evidence behind the fix.
5. Record the lesson and how it constrains an inference optimization.

Use a two-token scalar attention example, a three-token shifted-loss example and selected finite differences to connect equations to execution. PyTorch autograd for the full model is appropriate; manually deriving every network gradient is not required. Learner understanding is assessed separately from automated tests. Repeat a step if the implementation works but its behavior remains unclear.

## Milestones and evidence

| Stage | Deliverable | Acceptance |
| --- | --- | --- |
| M1 | Source-to-component map, configs/tokenizer/fixtures and Colab plan | Source revision and simplifications recorded; Unicode encode/decode and special-ID behavior defined; one hand-worked shifted target example; Free profile, source ZIP route, dependency setup and persistent-storage budgets declared. |
| M2 | Complete forward/reference attention | Finite shapes; future/padding changes leave valid prior logits unchanged; scalar/per-head independent reference agreement; selected FP64 gradient checks. |
| M3 | LM training, generation, checkpoint and Colab notebook | Small fixture overfit; finite nonzero expected gradients; EOS/context rules; output round trip and resumed updates agree. Verify interrupted writes and resume the same run from Drive after deleting the Colab Free runtime. |
| M4 | Downstream adaptation | CLS pooling independent of padding; binary classifier fits a tiny fixture; combined-loss arithmetic and backbone/head gradients verified. |
| M5 | CPU profile, memory ledger and learner explanation | Record weights/activations/workspace, parameter count, thread settings, step/uncached-generation latency and lessons; distinguish fitting from generalization. |
| M6 | Early KV-cache extension after base-model understanding | Cached versus uncached logits/positions agree; quantify cache bytes and prefill/decode separately; compare concatenation/preallocation under a bounded protocol. |

Initial numerical targets: reference FP64 components `atol=1e-8, rtol=1e-6`; FP32 parity `atol=1e-5, rtol=1e-4`. These are proposed starting tolerances, to be justified by observed operation sizes/error before acceptance. Compare deterministic eval with dropout disabled. Save max absolute/relative error; investigate rather than silently weakening thresholds.

A same-process CPU checkpoint round trip should produce identical eval logits. Also test loading in a new process and compare uninterrupted N updates with K + reload + N−K under identical RNG/optimizer/data/scheduler state and documented deterministic settings. Portability across devices/environments is a separate tolerance check. For Colab, publish verified full-state generations to Drive, reject partial/corrupt candidates, and demonstrate recovery in a fresh hosted runtime. Use the complete COLAB.md acceptance matrix, including rollback inside an accumulation window and retention/storage failures.

Learning pilot proposal: eight fixed short LM examples and a separately defined classification fixture; correctness config; seed 0; maximum 300 updates and 120 seconds per pilot, whichever comes first. Stop immediately on nonfinite loss/gradients. Target LM training loss ≤0.2 nats/token and full fixture classification accuracy; if the bound is hit first, record an inconclusive learning check and inspect alignment/optimizer/model before extending. These targets check fitting only, not held-out language capability. Use disjoint generated examples for an optional held-out behavior check.

Start timing with a small explicit thread count and record it; compare 1/4/8 threads only if overhead warrants it. Model/optimizer bytes and peak resident memory are separate from parameter count. No throughput estimate or CPU-default training duration is established yet.

The CPU fixture pilot above stays unchanged. Separately declare the Colab Free profile and run goal during M1; start its smoke check with the correctness config. Proposed hosted session limits are 60 minutes or 1,000 additional optimizer updates, with persistent saves every 5 minutes or 100 completed updates and on normal end. Finalize them from measured step/save times. Keep the experiment goal and LR schedule stable across session slices. The thin notebook imports the same trainer, with explicit new/resume modes and resource/identity checks; it must support uploaded exact child source while remotes remain local.

## Deferred work and recovery

BPE and historical tokenizer/checkpoint parity can follow the first complete reference model. Add KV caching as the early M6 systems extension after the base model is understood; it does not replace small-model training. Ordinary generation and checkpointing remain part of the base build. SSD streaming begins only after cache/state semantics are verified. Do not require a speedup to finish a correctly conducted experiment. See [the connected research track](07-inference-research-track.md).

Preserve failed run records and optimizer/config inputs. Reduce batch/sequence/width for slow pilots; do not change post-norm semantics to hide instability. Add a model variant only with its own config and coverage checklist. Paid compute and remote publication remain separately instructed actions.
