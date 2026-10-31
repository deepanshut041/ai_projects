# Roadmap review — learning models deeply for constrained inference research

Reviewed: 2026-10-09. Status: proposed plan; no model implementation or run.

## Goal and overall judgment

The research goal is to run larger models on machines with limited accelerator memory by studying weight placement, SSD/RAM offloading, KV-cache management, quantization and scheduling. Reaching it requires hands-on architecture learning: implement and train small complete models, inspect their behavior, then optimize mechanisms that are understood.

The existing roadmap is a strong architecture-learning foundation for that goal. Keep its breadth and complete-model projects. It needs a more explicit connection between each architectural mechanism and the eventual inference research, plus learner-owned implementation/training milestones. Reading code and executing AI-generated solutions alone do not demonstrate understanding.

Use a recurring cycle:

**Understand the equations → implement → train a small model → inspect/debug → explain → profile → optimize → verify.**

Architecture learning and systems research should progress together. Do not require all 19 models to be finished before the first systems investigation, and do not bypass the small-model builds to jump directly into a large inference engine.

## Recommended improvements

| Priority | Improvement | Why it matters |
| --- | --- | --- |
| Must | Keep complete small implementations and bounded training exercises | The learner needs to build, observe and debug to understand mechanisms in depth. |
| Must | Add a mechanism-to-memory/state/access map to every project | Optimization choices depend on the model's actual dependencies and reuse. |
| Must | Add explicit learner checkpoints | Explain tensor shapes, calculate a small loss/attention example, inspect gradients, and predict an ablation before running it. |
| Must | Add an inference investigation beside each architecture | Connect the learning work to larger-model capacity, cache behavior and offload scheduling. |
| Must | Make prerequisites and the real study order explicit | Numbering differs from the suggested order; capstones combine several earlier mechanisms. |
| Should | Reuse the train/data/checkpoint/profile harness while keeping each model/version fully runnable | Reduces repeated plumbing without replacing model implementation practice. |
| Should | Give each close model relative its own runnable config and coverage/difference table | Qwen, DeepSeek and Kimi variants share components but have consequential differences. |
| Should | Define correctness, learning, resource-measurement and optimization gates separately | Correctness and understanding are required; a speedup is a result to investigate, not a guaranteed outcome. |
| Should | Add bounded CPU pilots and empirical hardware calibration | Parameter budgets alone do not determine memory or runtime. |
| Should | Add a systems literature/baseline register | Compare with existing offloading/cache/quantization systems before claiming a new contribution. |
| Should | Adapt section 9 to independent repositories and choose one canonical roadmap owner | Fits the workspace and avoids drift between duplicate copies. |

The roadmap already has good causal/state/checkpoint checks, pinned-source expectations and cautious interpretations. Retain those; add the missing links to the stated research goal.

## How the architectures serve the research

These are proposed investigation directions, not measured findings.

| Study | Learn by implementing/training | Subsequent inference question |
| --- | --- | --- |
| GPT-1/GPT-2 | Causal MHA, post/pre-norm, positions, shifted loss, optimization and generation | What is recomputed without a cache, and how do weight/cache bytes grow? |
| Llama/Qwen dense | RoPE, GQA, RMSNorm, gated FFN, Q/K normalization | How does KV-head structure change cache storage and physical reads? |
| Switch/Qwen MoE | Router/expert gradients, dispatch, balance and token capacity | Which experts must be loaded, and how predictable is reuse? |
| DeepSeek MLA/sparse/compressed attention | Latent representations, index selection, causal compression | Which state must be resident, reconstructed or fetched? |
| Hybrid DeltaNet/Kimi/Mamba/RWKV | Recurrence, chunking and state reset | How do bounded recurrent states coexist with growing attention caches? |
| Gemma/PaliGemma/unified multimodal | Cache-sharing ownership, per-layer lookups, visual/audio sequences | Which tensors can be shared or offloaded, and how much state does multimodal prefill add? |
| Diffusion | Masked training, repeated denoising and conditional generation | How does repeated refinement change weight/state access compared with autoregressive decoding? |

Retain the roadmap's dependency-led order: foundations 1–5; then 10/11 and 6–9; later 12–15; branches 16–19 when prerequisites are understood. Add systems exercises progressively, starting at Project 1. The sequence may adapt to observed learning and research questions; it is not a reason to remove architectural depth.

## Project 1 remains GPT-1

Build and train a complete small GPT-1 with learned positions, causal multi-head attention, post-norm blocks, tied output weights, explicit classification pooling, generation and checkpoint/resume. Tiny training verifies the path and teaches behavior; it does not recreate GPT-1's historical language capability.

Use a complete 2-layer, width-64 CPU config first, then scale toward approximately 1M parameters after timing. Byte tokenization is a declared simplification; BPE can follow without blocking the first learning cycle.

The learner should implement selected core pieces, work through small numerical examples, run/debug the training loop and explain results. The assistant can prepare structure, reference checks and feedback, while preserving meaningful exercises for the learner.

Project 1 ends with a memory ledger and uncached-generation baseline. Add a verified KV-cache extension once the base model is understood; compare exact outputs and memory before trying offloading. Its classifier is a short downstream-learning milestone, not the central research question.

See [the concrete starting plan](02-implementation-plan.md) and [the connected inference research track](07-inference-research-track.md).

## Delivered state

Both original roadmap copies remain unchanged. This review, a hands-on GPT-1 starting plan and a linked systems research plan are saved. No repository, model code, training, benchmark or publication has been produced.
