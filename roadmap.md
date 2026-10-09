# Projects roadmap

Updated: 2026-10-09
Canonical plan for implementing, training and understanding known AI models.

Start each work session with [resume.md](resume.md), then [progress.md](progress.md). This file defines direction and completion gates; progress.md owns current status/evidence; resume.md owns the immediate next action; [decisions.md](decisions.md) records durable choices. Update all affected records before ending a substantial work session.

## Purpose and ownership

Learn model mechanisms deeply by building and training complete small implementations. The longer-term motivation is larger-model inference on constrained machines, but a Projects exercise implements or studies established techniques. Novel optimization questions, focused reproductions and improvement experiments belong in [Research](../../ai_research/roadmap.md).

Keep the full architecture path. Study models through **understand → implement → train → inspect/debug → explain → profile → verify**. A known optimization can be an educational extension; a claimed improvement needs a separately scoped Research protocol. Do not replace hands-on learning with reading or an already-finished generated solution.

Implementations live in independent repositories under ../projects/, registered as submodules. The workspace CLI lives separately in ../../cli/. All shared Projects documentation lives here under docs/. Use C++ and Python for implementation and tutorial examples. Large datasets, weights, environments, checkpoints and generated runs remain ignored, with provenance/acquisition instructions committed.

## Learning path and prerequisites

IDs are stable even when the study order changes. Dependencies mean learning prerequisites, not source-code coupling. Closely related builds share plumbing but remain separately runnable with a coverage/difference checklist.

| ID | Complete small-model project | Prerequisites | Mechanisms to implement and learn |
| --- | --- | --- | --- |
| P00 | Tokenization and Transformer Foundations | None; bounded tensor/autograd primer included | Character/byte/BPE tokenization, embeddings, attention, masks, post-norm encoder–decoder, conditional training and recovery. |
| P01 | Small GPT Language Model | P00 correctness, training/recovery and explanation gates | Decoder-only causal MHA, learned positions, pre-norm/GELU, next-token loss, text generation and full training state. Historical GPT-1 is an optional reference extension. |
| P02 | GPT-2 | P01 | GPT-2-specific config/initialization/weight mapping, tokenizer compatibility and verified KV cache; strengthen the existing pipeline. |
| P03 | Llama 2 → Llama 3 | P02 | RMSNorm, RoPE, SwiGLU, GQA and position/cache semantics. |
| P04 | Qwen3 Dense | P03 | Q/K normalization, model-specific configuration and reference parity. |
| P05 | Switch-style routing → Qwen3 MoE | P04 | Top-1/top-k routing, expert dispatch/combine, gradients and balance. Label decoder adaptations explicitly. |
| P06 | Qwen3-Next | P05 | Gated DeltaNet, gated attention, MoE and hybrid state. |
| P07 | PaliGemma | P03 | Vision encoder/projector, patch/token assembly, masks and image-dependent training. |
| P08 | Qwen3.5 | P06, P07 for vision | Complete hybrid text model, then model-specific vision integration. |
| P09 | Qwen3.8 dense VLM | P08 | Version-specific hybrid/vision paths and MTP where specification supports it. |
| P10 | Gemma 4 E2B text model | P03 | Local/global attention, per-layer embeddings and cache-sharing ownership. |
| P11 | DeepSeek V3 → V3.2 | P05 | MLA, shared/routed experts, MTP and causal sparse indexing. |
| P12 | DeepSeek V4-Flash-style model | P11 | Compressed attention and constrained residual-stream mixing. |
| P13 | Qwen3.8-Flash-Next-style model | P06, P09, P11 | Sparse selection, n-gram lookup, gated residuals and integrated hybrid MoE. |
| P14 | Kimi Linear → Kimi K3-style model | P06, P11; P07 for vision | KDA/gated MLA, Attention Residuals, latent experts and later vision path. |
| P15 | DeepSeek V4.1-Flash-style model | P12 | Causal encoder/decoder, shared compressed caches, replay and conditional memory. |
| P16 | Mamba 2 → Mamba 3 | P02; recurrence primer | State-space recurrence, sequential/chunked equivalence and richer state variants. |
| P17 | RWKV-7 | P02; recurrence primer | Time/channel mixing, recurrent updates and persistent-state behavior. |
| P18 | LLaDA → iLLaDA; optional editing extension | P02 | Bidirectional masked prediction, corruption/loss and iterative conditional generation. |
| P19 | Gemma 4 unified multimodal | P07, P10 | Direct modality projection, mixed sequences and end-to-end training checks. |

Default order: **P00 → P01–P05 → P10/P11 → P06–P09 → P12–P15**. P16–P19 are distinct branches entered when their prerequisites are understood. Complete the separately named incremental builds within a project; record partial coverage honestly. Source readiness is checked at each kickoff, not assumed from the model name.

The opening route is **tensors/gradients → tokenizer → attention → complete tiny encoder–decoder → small GPT text generator**. P00 is the foundation project; P01 is the first decoder-only language model. P01 means a small GPT-style model, not an exact GPT-1 reproduction. Keep P01–P19 IDs stable and retain older GPT-1 plans/tutorials as labeled reference material.

Detailed model plans, source links, sizing and CPU verification notes are consolidated below in [this roadmap](#detailed-project-plans-and-retained-source-notes). Current status lives in progress.md; setup/report conventions live in SETUP.md. Published reference sizes are metadata, not instructions to allocate/train large models.

## Training compute and recovery

Develop locally and use Colab for bounded training. **P00–P04 must target Colab Free first**; P05 onward gets an evidence-based compute choice at kickoff, with Pro later when requested. Keep complete miniature models and local CPU correctness; published dimensions are reference metadata. This is a requirement, not a verified hosted capability.

Follow [COLAB.md](COLAB.md) for child-owned notebooks, dependency/source delivery, profiles, full training state, verified Drive checkpoint generations, retention and fresh-runtime recovery. A thin notebook calls the same trainer as local runs. Proposed session and save budgets are measured before expansion. Checkpoints include optimizer/scheduler/precision/RNG/data-position state; weights alone do not resume a training run. Model completion for the early group includes evidence after **Disconnect and delete runtime** and resuming the same run.

## Required cycle for each project

1. Read primary paper/source and pin revisions. Draw the full model and annotate tensor/state shapes.
2. Declare local correctness/published-reference configurations, tokenizer/data/objective contracts, simplifications and compute bounds; P00–P04 also require a Colab Free profile under the [Colab contract](COLAB.md).
3. Implement meaningful components and assemble a complete model. The learner predicts small numerical examples before running them; the assistant supports scaffolding, checks and review.
4. Train a small model/task; inspect actual loss, activations and gradients; diagnose at least one failure or controlled change.
5. Verify forward/backward, causal or conditional behavior, generation/state reset, complete training checkpoint/resume and relevant independent numerical references. For P00–P04 also verify persistent saves, interrupted-write recovery and resume after deleting the Colab runtime.
6. Explain model-specific mechanisms from the implementation and observations. Passing tests and learner understanding are separate gates.
7. Profile weights, cache/state, intermediates, workspace and tensor lifetimes/access. Measure prefill/decode separately where relevant.
8. Implement one established execution technique when it reinforces the mechanism being studied. Verify semantics and errors before interpreting timings.
9. Create or extend the corresponding tutorial under tutorials/ with editable diagrams, C++/Python worked examples, practice and whiteboard preparation; record tutorial availability separately from understanding.
10. Save coverage, results/limits and the next action. Update progress.md, resume.md and any new decision.

A negative or inconclusive performance result can complete an educational investigation. A speedup is not required, but evidence and explanation are.

## Project 0 Tokenization and Transformer Foundations

**Recommended first project:** build the text-to-tensor pipeline, then a complete miniature encoder–decoder Transformer based on [Attention Is All You Need](https://arxiv.org/html/1706.03762v7). Proposed independent repository: ../projects/transformer-foundations-lab/; no repository/model/training exists yet. [START_HERE.md](START_HERE.md) owns kickoff; the [current planning package](implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/02-implementation-plan.md) records source evidence, contracts and documentation verification.

This is one bounded build, not a new curriculum area. Use Python/PyTorch for complete training and selected C++/Python numerical traces. Implement the mechanisms directly using tensor operations/autograd; use high-level Transformer modules only as independent checks, not as the implementation. A new tensor engine, CUDA kernels, full WMT translation and full C++ training port are optional later work.

| Milestone | Build and learn | Evidence before advancing |
| --- | --- | --- |
| P00-M1 | Tensors, broadcasting, matrix products, stable softmax/cross-entropy, chain rule/autograd and optimizer updates; fit one tiny MLP on synthetic labels. Pin sources/runtime and record CPU/Free budgets. | Hand-computed shapes/loss, an analytic or finite-difference gradient check, finite gradients and a explained loss decrease. |
| P00-M2 | Compare character and UTF-8 byte baselines; implement byte-BPE training, deterministic merges, encode/decode and artifact save/load. | Unicode/empty/repeated-input round trips, merge-order example, deterministic tie handling, split-before-tokenizer-training and artifact/hash checks. |
| P00-M3 | Embeddings, sinusoidal positions, scaled dot-product/single-head then multi-head attention, padding/causal masks, LayerNorm, residuals and ReLU FFN. | Hand-worked attention, independent numerical/gradient checks, shape/mask invariants and future-target perturbation checks with dropout off. |
| P00-M4 | Assemble encoder self-attention + decoder masked self-attention + cross-attention with post-residual LayerNorm; teacher forcing and padding-aware target loss. | Full forward/loss/backward and autoregressive conditional decode; prove source conditioning, target causality, PAD exclusion, EOS/bounds and fresh-request reset. |
| P00-M5 | Train on generated copy sequences; use reversal as one controlled extension. Add common trainer, thin Free notebook and full persistent recovery. | Separate fitting and disjoint held-out generation metrics; train/validation split checks; interrupted-save fallback, deterministic local resume and fresh Free runtime recovery under COLAB.md. |
| P00-M6 | Inspect gradients/attention/errors, count parameters and map memory; explain the encoder–decoder-to-GPT transition and save a tokenizer/trainer handoff. | Evidence-backed explanation, one diagnosed failure/change, bounded local profile, source/departure checklist and a versioned handoff. Learner mastery is assessed separately. |

### Tokenizer and data contract

Begin with a character exercise and a lossless byte baseline, then implement a deliberately simple byte-BPE. [Sennrich et al. §3.2](https://arxiv.org/html/1508.07909v5) motivates merge learning; [minbpe](https://github.com/karpathy/minbpe) is a readable byte implementation reference. This educational tokenizer is not claimed compatible with GPT-1/GPT-2 token IDs or preprocessing.

Proposed IDs: bytes 0–255, PAD/BOS/EOS/SEP 256–259, merge IDs from 260. Train merges on training records only, never across document/special-token boundaries. Break equal-frequency ties lexicographically by token-ID pair; stop at the vocabulary cap or no repeated eligible pair. Encoding applies frozen merge ranks, not new test-input frequencies. Treat literal special-token spellings as ordinary text unless explicitly supplied structurally. Preserve exact UTF-8 text without implicit normalization; distinguish byte round trips from rendering arbitrary generated bytes, whose UTF-8 error policy is recorded.

Persist schema, byte vocabulary/merge ranks, special-ID map, boundary/normalization policy, training split/source hashes and tokenizer hash. Freeze that identity in each run/checkpoint; a changed tokenizer requires an explicit new run. Report tokens per byte/record and vocabulary size on the same held-out text, without claiming that compression proves model quality. A character baseline may reject unseen characters explicitly; the byte path must encode valid unseen Unicode.

The mandatory model task starts with synthetic digit/text copy pairs using the byte baseline; a second bounded run can use BPE. Split/deduplicate source sequences before fitting the tokenizer or creating overlapping batches. Declare seeds, generator version, length distribution and split hashes; shared alphabet is acceptable, duplicate examples across splits are not. Train with decoder input `[BOS, y0, ...]` and labels `[y0, ..., EOS]`; ignore PAD in loss and metric denominators. Evaluate free generation without teacher forcing. Reversal adds a distinct transformation with the same pipeline; it does not replace the required copy baseline.

### Small architecture and paper fidelity

| Profile | Proposed complete model | Purpose |
| --- | --- | --- |
| Local correctness | 1 encoder + 1 decoder layer, width 32, 4 heads, FFN 128, source/target limit 32, byte vocabulary 260, FP32/dropout 0 | Fast CPU shape/gradient/mask and overfit checks. |
| Colab Free learning | Up to 2 encoder + 2 decoder layers, width 64, 4 heads, FFN 256, source/target limit 64, vocabulary at most 512 | Bounded task training/recovery; choose only after a measured smoke run. |
| Architecture reference | Paper dimensions/training choices recorded as metadata | Explicit reduced-size/task/training departures; never allocate implicitly. |

Keep bidirectional encoder attention, causal target self-attention and decoder queries attending to encoder keys/values. Encoder and cross-attention hide source padding; decoder self-attention also hides target padding. Valid query rows must have a visible key; reject empty/all-padding source records and define safe padded-query behavior. Do not impose GPT causality on the encoder. Use paper post-norm, sinusoidal positions, ReLU and embedding scaling. Record weight tying, dropout, label smoothing and scheduler choices; initial unsmoothed loss/plain bounded optimizer choices may differ from the paper. The [Annotated Transformer source](https://github.com/harvardnlp/annotated-transformer) is a component/copy-task guide whose convenience pre-norm code differs from the paper; inspect before reusing.

Finalize task pass thresholds/budgets before training. A proposed gate is at least 90% exact generated copy matches on 100+ disjoint held-out examples in the trained length range, plus reported token accuracy and source-shuffle/control degradation. Small-set overfit is a separate gate. Diagnose failures before expanding; record new protocols without relabeling a failed run as success. Synthetic success demonstrates the chosen sequence task, not translation or general language competence.

### Handoff to the first language model

P00 supplies a frozen tokenizer artifact/API, proven mask/block contracts, data/split/provenance conventions and trainer/checkpoint patterns. P01 remains independently runnable: use an explicit versioned package or source snapshot with license/revision, not live imports through workspace-relative paths. Reuse verified plumbing selectively; do not transfer encoder–decoder weights as a GPT checkpoint. Copy the [COLAB.md](COLAB.md) notebook/full-state/persistent-recovery gates into the child package and verify them again for the language-model task.

## Project 1: Small GPT Language Model

Proposed repository: ../projects/gpt-lab/ (not created yet), entered after P00 gates. Build a complete small decoder-only text model using causal multi-head attention, learned positions, pre-norm/final LayerNorm and GELU. [build-nanoGPT](https://github.com/karpathy/build-nanogpt) is a code-reading reference; use bounded local/Free settings instead of its large/distributed defaults. This is an educational GPT-style model, with original GPT-1 details retained as optional historical study.

| Milestone | Outcome |
| --- | --- |
| P01-M1 | P00 handoff/revision checklist; decoder-only architecture map; tiny config, frozen byte/BPE tokenizer, licensed text split/shifted-target and Free setup/storage/budget contracts. |
| P01-M2 | Complete pre-norm decoder; independent attention/gradient and causality checks. Explain removal of encoder/cross-attention and changes from P00's positions, activation and normalization. |
| P01-M3 | Train/backpropagate a small text LM, measure held-out loss, sample text, implement common-trainer notebook/full persistent resume and verify interrupted saves/fresh Free recovery. Inspect/explain behavior. |
| P01-M4 | Optional historical GPT-1 post-norm/classification adaptation with explicit pooling and combined objective. Keep this separate from the required text LM. |
| P01-M5 | Required parameter/memory ledger, reproducible CPU profile, diagnosed failure/change and learner explanation. |
| P01-M6 | Optional early KV-cache extension; required GPT-2 cache coverage follows in P02. Match uncached/cached positions and outputs before comparing storage strategies. |

P01 completion requires M1–M3 and M5 plus the shared learning/recovery gates; M4/M6 are optional extensions. Initial correctness config remains 2 layers, width 64, 4 heads, context 64, FFN 256 and byte vocabulary 260; use the actual frozen BPE vocabulary in the BPE profile. Increase toward 1–3M parameters only after measured fit. The [older GPT-1 plan](implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/02-implementation-plan.md) and [tutorial](tutorials/projects/p01/gpt-1-foundations/v1/lesson.md) retain historical source/architecture material; their mandatory post-norm/classifier sequence is superseded by this route.

Begin at **P00-M1**; create only the selected child's repository when implementation starts. No foundation/model milestone is complete from updating this plan.

## Resuming and advancing

Before editing: read resume/progress, inspect the owner AGENTS.md and Git state, then resume the child's existing implementation package. Create a child-owned package before deep implementation. Use task argv arrays and registered paths.

Advance only with linked implementation, correctness, training/inspection, learner-explanation and resource evidence appropriate to the model. Do not mark a whole family done after one variant. State implementation, verification, learning, research and external publication separately.

For a new Research question, write a short handoff identifying the mechanism, observed limitation, baseline and question. Put the authoritative protocol and results in Research. The learning roadmap continues independently.

## Change rules

Keep IDs stable; append decisions rather than erasing them. Record a changed study order/prerequisite and its reason. Extend source cutoff dates explicitly. Historical review/setup packages remain evidence; their old checkpoints do not override resume.md.


## Detailed project plans and retained source notes

The model plans, sizing guidance, exercises and resource links below were consolidated from the original October 9, 2026 planning document. They are reference notes within this single roadmap. The current P00-first order and P00/P01 gates above govern execution; the retained GPT-1 description is an optional historical branch. Later inherited model/release claims and resource availability were not freshly verified by this opening-route revision; recheck primary sources and pin revisions at each project kickoff. Proposed million-parameter budgets are optional later configurations, not allocations required for the first tiny model.

Original scope: from GPT-1 through dense, sparse, recurrent, diffusion and multimodal models. Prepared and release/resource cutoff: October 9, 2026.

### Objective and working assumptions

Return to LLM research by writing complete miniature model implementations, understanding their mathematics, verifying their behavior, and optimizing their execution.

The learner wants ground-up implementation before a small language model; current understanding is unassessed. P00 supplies bounded prerequisites, then the main work is implementation and experimentation. Blogs such as Sebastian Raschka's are examples of the desired teaching format; the resources below also include Karpathy, Umar Jamil, LabML, Stanford, model authors, and other open implementations.

The available PC does not have a suitable GPU for substantial training. Therefore:

- Use tiny, configurable models for local forward passes, backward passes, generation, state management, and profiling.
- Use short learning checks on synthetic or very small datasets when practical.
- Train early complete miniature models on Colab Free through the same code, with persistent checkpoints and fresh-runtime recovery under [the compute contract](COLAB.md). Local checks still come first; choose Pro later from measured needs.
- Keep full pretraining and large-scale post-training as later experiments.
- Write readable PyTorch reference implementations before adding optimized execution paths.
- Make each project end in a complete model, not just an isolated architectural modification.

The published model names identify the architectures being studied. The small parameter budgets identify the educational replicas. These replicas are not official checkpoints and are not expected to reproduce the original models' capabilities.

This is a selected roadmap of distinct, publicly documented architectures available at the cutoff date. It is not an exhaustive release catalogue or a benchmark ranking.


### How to size the implementations

The following are proposed starting budgets. Count the exact parameters in each implementation; embedding tables, expert sizes, and projection choices can change the totals substantially.

| Level | Starting configuration | Use |
|---|---|---|
| Component test | Width 32–64; sequence length 8–32; batch size 1 | Verify an equation, mask, gradient, routing decision, or recurrence. |
| Complete tiny dense model | 4–6 layers; width 128; 4 attention heads where appropriate; vocabulary 1,024–2,048; context 64–256 | Default GPT, Llama, Qwen, and dense-model projects. Target roughly 1–3M parameters. |
| Complete tiny sparse/hybrid model | 4–8 layers; width 128–192; 4 routed experts; top-1 or top-2 routing; small latent ranks | MoE, hybrid attention, and advanced cache projects. Target roughly 3–8M total parameters. |
| Tiny multimodal model | Small images with few patches; a shallow vision component; short audio clips where applicable | Verify image/audio token construction, modality fusion, and gradients. |
| Larger configuration | Increase dimensions through the same configuration interface | Resource estimates and future experiments when compute is available. |

Use FP32 for the initial CPU implementation. Use small FP64 inputs for selected numerical gradient checks. Preserve valid dimensional relationships, head divisibility, rotary dimensions, and state shapes when reducing a model.

For MoE and lookup-memory models, report total parameters, selected experts, state/cache memory, and lookup-table parameters separately. Sparse activation does not remove the storage cost of inactive experts or tables.

Keep explicit configuration profiles per model:

1. **Local correctness:** runs tiny CPU numerical checks and short experiments.
2. **Colab Free (required for P00–P04):** bounds the small-model training data/token, memory and session budgets; uses the same trainer and tested persistent resume. Later projects decide their tier at kickoff.
3. **Architecture reference:** records published dimensions and deliberate differences; never allocates implicitly.
4. **Colab Pro (later, optional):** expands a measured budget when requested without replacing the checkpoint/recovery workflow.


### Roadmap at a glance

The order follows learning dependencies rather than the historical genealogy of every model. Projects with arrows contain incremental builds, each of which should remain runnable.

| Project | Complete model implementation | Suggested local budget | Main implementation objective |
|---|---|---|---|
| 0 | Tokenization and Transformer Foundations | Width 32–64; count at kickoff | Tokenizer, attention and complete tiny conditional encoder–decoder. |
| 1 | Small GPT Language Model | Tiny correctness first; 1–3M later | Causal pre-norm decoder, learned positions and complete text LM; GPT-1 history optional. |
| 2 | GPT-2 | 1–3M | Pre-normalization decoder and complete language-model pipeline. |
| 3 | Llama 2 → Llama 3 | 1–3M | RMSNorm, RoPE, SwiGLU, GQA, and KV caching. |
| 4 | Qwen3 Dense | 1–3M | Qwen-specific attention normalization and model configuration. |
| 5 | Switch Transformer → Qwen3 MoE | 2–5M | Routing, dispatch, expert combination, and load balancing. |
| 6 | Qwen3-Next | 3–6M | Gated DeltaNet, gated full attention, hybrid state, and MoE. |
| 7 | PaliGemma | 2–4M | Vision encoder, projector, multimodal sequence, and decoder. |
| 8 | Qwen3.5 | 3–6M | Complete hybrid text model, followed by its vision path. |
| 9 | Qwen3.8-27B-style model | 3–8M | Integrated dense hybrid vision-language model and MTP. |
| 10 | Gemma 4 E2B-style model | 1–3M | Local/global attention, cross-layer KV sharing, and per-layer embeddings. |
| 11 | DeepSeek V3 → V3.2 | 2–5M | MLA, shared/routed experts, and learned sparse indexing. |
| 12 | DeepSeek V4-Flash-style model | 3–6M | Compressed attention and constrained residual-stream mixing. |
| 13 | Qwen3.8-Flash-Next-style model | 3–6M | QSA, n-gram embeddings, gated residuals, hybrid MoE, and MTP. |
| 14 | Kimi Linear → Kimi K3-style model | 3–8M | KDA, gated MLA, Attention Residuals, and latent-space experts. |
| 15 | DeepSeek V4.1-Flash-style model | 4–8M | Causal encoder–decoder, shared compressed caches, and conditional memory. |
| 16 | Mamba 2 → Mamba 3 | 1–3M | State-space recurrence, richer state tracking, and MIMO. |
| 17 | RWKV-7 | 1–3M | Recurrent time/channel mixing and generalized delta-rule updates. |
| 18 | LLaDA → iLLaDA; optional LLaDA2.1 | 1–3M | Masked diffusion, iterative generation, and token editing. |
| 19 | Gemma 4 12B-style unified model | 2–4M | Direct image/audio projection into a shared language backbone. |

#### Suggested study order

- **Foundations:** project 0, then projects 1–5.
- **Modern dense and sparse implementations:** projects 10–11, then 6–9.
- **Advanced research capstones:** projects 12–15.
- **Distinct research branches:** projects 16–19. These can be explored independently once their prerequisites are understood.

Gemma 4 fits naturally after Llama. DeepSeek V3 fits after the first MoE implementation. Kimi K3 depends on understanding both hybrid recurrence and MLA. DeepSeek V4.1 benefits from completing the earlier DeepSeek projects.


### Detailed implementation projects

#### 1. GPT-1 historical reference (optional P01 extension)

The required P01 is the Small GPT Language Model defined above. This retained historical architecture/task-head exercise is optional; it does not override the P00-first sequence or require exact GPT-1 replication.

**Build:** A small decoder-only language model with causal multi-head attention, learned positional embeddings, GELU, residual connections, and GPT-1's post-residual LayerNorm arrangement. Add a classification head as a separate downstream use.

**Resources:**

- [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer): annotated component implementation and training examples.
- [Original GPT-1 implementation](https://github.com/openai/finetune-transformer-lm/blob/master/train.py): historical architecture authority.

The annotated Transformer is a component-learning resource, not a faithful GPT-1 implementation. Use the original GPT-1 source to resolve model-specific details; write your own modern PyTorch version of its older TensorFlow code.

**Local exercise:** Overfit a short sequence, then check that the task head receives gradients and can fit a tiny classification dataset.

**Completion:** Complete model, causal-mask checks, correct next-token target shift, and working task-head code.

#### 2. GPT-2

**Build:** A complete GPT-2-style model and pipeline, including pre-normalization blocks, final LayerNorm, token embeddings, positional embeddings, optimizer integration, checkpointing, and generation.

**Resources:**

- [Karpathy: build-nanoGPT](https://github.com/karpathy/build-nanogpt): step-by-step code history and linked code-along lecture.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): visual explanation.
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html): GPT and tokenizer lessons.
- [Raschka: LLMs From Scratch](https://github.com/rasbt/LLMs-from-scratch): complementary implementation, pretraining, and fine-tuning material.

**Local exercise:** Reuse P00's byte-BPE understanding, then document GPT-2 tokenizer/preprocessing differences and any compatibility claims. Save/reload complete checkpoints and compare generation with/without a KV cache. Extend P01's pipeline with GPT-2-specific coverage rather than repeating the foundation build.

**Completion:** Reproducible forward/loss/backward execution, checkpoint round trip, and cache parity.

**Optional GPT-3-style research exercise:** Sweep several small model configurations and token budgets. Measure learning curves and resource use. Treat this as a scaling study, not an exact GPT-3 reproduction. Stanford's [CS336 scaling material](https://cs336.stanford.edu/) provides the research framework; use local experiments if its assignment API is unavailable.

#### 3. Llama 2 → Llama 3

**Build:** A complete Llama-style model. Implement RMSNorm, RoPE, SwiGLU, bias choices, grouped-query attention, and KV caching. Keep the Llama 2 and Llama 3 configurations separately identifiable.

**Resources:**

- [Umar Jamil: Llama from scratch](https://github.com/hkproj/pytorch-llama): code and linked explanation.
- [Raschka: converting GPT to Llama](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama): incremental Llama notebooks.

**Local exercise:** Compare normalization, positional encoding, and feed-forward alternatives one at a time. Check cached generation positions carefully.

**Completion:** Complete model, correct GQA head mapping, rotary-position tests, and cache parity.

#### 4. Qwen3 Dense

**Build:** Your own complete Qwen3 implementation, including its query/key normalization, rotary attention, grouped-query attention, feed-forward blocks, and configuration details.

**Resources:**

- [Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch).
- [Public Qwen3 dense and MoE notebooks](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3).

The article is subscriber-only; the companion code is public. Use the tiny configuration for local work. Matching a released checkpoint is an optional later verification exercise.

**Local exercise:** Copy identical small weights into your implementation and a trusted reference configured equivalently; compare logits and selected gradients.

**Completion:** Complete model, normalization/head-shape checks, and reference agreement within a documented numerical tolerance.

#### 5. Switch Transformer → Qwen3 MoE

**Build:** First a complete Switch-style language model with top-1 routing, then a Qwen3-style MoE model with multiple selected experts and weighted output combination.

**Resources:**

- [LabML: annotated miniature Switch Transformer](https://nn.labml.ai/transformers/switch/index.html).
- [LabML: training walkthrough](https://nn.labml.ai/transformers/switch/experiment.html).
- [Qwen3 MoE notebooks](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3).

**Local exercise:** Use a slow token-by-token expert dispatcher as the correctness reference. Implement a batched dispatcher and compare outputs and gradients. Log expert selection and, where used, capacity overflow and dropped tokens.

**Completion:** Routing and combination are correct; selected experts and router weights receive the expected gradients. An expert need not receive gradients in a batch where no tokens select it.

#### 6. Qwen3-Next

**Build:** A complete hybrid MoE model with Gated DeltaNet and periodic gated full-attention blocks. Implement recurrence, convolutional state where required, and attention-cache management.

**Resources:**

- [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms): hybrid architecture and recurrence explanation.
- [Gated DeltaNet implementation material](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet).

**Local exercise:** Start with a readable sequential recurrence. Compare full-sequence and step-wise execution before implementing chunked execution. Measure recurrent-state storage separately from the growing full-attention cache.

**Completion:** Complete hybrid model, correct state reset, and agreement between execution modes under matching semantics.

#### 7. PaliGemma

**Build:** A complete small vision-language model: vision encoder, modality projector, multimodal token assembly, attention masks, and language decoder.

**Resource:** [Umar Jamil: PaliGemma from scratch](https://github.com/hkproj/pytorch-paligemma), with a linked explanatory video and component code.

**Local exercise:** Use generated images containing simple shapes. Verify patch ordering, image-token insertion, and gradients through the vision path. Add a small training loop; the linked implementation primarily demonstrates model construction and inference.

**Completion:** End-to-end multimodal forward/backward execution. A tiny learning check should establish that image information can influence the task result; changing random inputs alone is insufficient evidence of useful visual understanding.

#### 8. Qwen3.5

**Build:** The complete dense hybrid text model first, then its vision-language path. Preserve the model's own normalization, gating, position handling, and state requirements rather than assuming the preceding hybrid implementation is identical.

**Resources:**

- [Raschka: Qwen3.5 implementation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/16_qwen3.5).
- [Tiny Qwen](https://github.com/Emericen/tiny-qwen): readable implementation covering the Qwen3.5/3.6/3.8 generation.

Raschka's example reuses some Hugging Face linear-attention helpers. Reimplement those helpers as an explicit exercise if the goal is to understand every component.

**Local exercise:** Trace text-only prefill, multimodal prefill, and incremental decoding. Check that a fresh request resets every relevant state.

**Completion:** Complete model with documented text/vision scope and working state management.

#### 9. Qwen3.8-27B-style model

**Build:** A complete miniature dense hybrid vision-language model following the released architecture, including its multi-token prediction component.

**Resources:**

- [Official model card](https://huggingface.co/Qwen/Qwen3.8-27B).
- [Released configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
- [Tiny Qwen reference implementation](https://github.com/Emericen/tiny-qwen).

**Local exercise:** Check ordinary next-token prediction, the additional MTP training path, vision integration, and cached generation independently before combining them. Implement a minimal tool-call parser/harness as a separate system exercise.

**Completion:** Complete model and verified integrated behavior. Record which released mechanisms are implemented and which remain deferred. A random or lightly trained replica is not evidence of frontier reasoning or agent capability.

#### 10. Gemma 4 E2B-style model

**Build:** A complete text model with local/global attention, cross-layer KV sharing, and per-layer embeddings. Use a small auxiliary embedding dimension.

**Resources:**

- [Per-layer embeddings explanation](https://sebastianraschka.com/llm-architecture-gallery/per-layer-embeddings/).
- [Raschka: standalone Gemma 4 notebook](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/17_gemma4).
- [Official E2B model](https://huggingface.co/google/gemma-4-E2B).

**Local exercise:** Compare cache ownership and storage with sharing enabled and disabled. Check the layer-specific embedding slice and gate for each block. The linked notebook is text-only; multimodal work is a separate extension.

**Completion:** Correct local/global masks, cache-sharing relationships, and per-layer embedding paths.

#### 11. DeepSeek V3 → V3.2

**Build:** A complete MLA/MoE language model with shared and routed experts. Include V3's load-balancing strategy and MTP as independent components, then add V3.2-style sparse indexing.

**Resources:**

- [Official DeepSeek V3 implementation](https://github.com/deepseek-ai/DeepSeek-V3).
- [MLA walkthrough/code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla).
- [DeepSeek Sparse Attention walkthrough/code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa).

**Local exercise:** Verify latent-cache reconstruction and position handling. For sparse attention, compare against a simple gather-based reference using the same selected indices. Test that the indexer cannot select future positions.

**Completion:** Runnable V3-style and V3.2-style configurations, correct cache behavior, and indexer checks.

#### 12. DeepSeek V4-Flash-style model

**Build:** A complete model with Compressed Sparse Attention, Heavily Compressed Attention, and manifold-constrained hyper-connections. Prototype residual mixing with two streams; complete the exercise with the released design's four-stream arrangement at small width.

**Resources:**

- [Official V4-Flash architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).
- [KV sharing, mHC, and compressed-attention walkthrough](https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures).
- [CSA/HCA explanation](https://sebastianraschka.com/llm-architecture-gallery/csa-hca/).
- [mHC explanation](https://sebastianraschka.com/llm-architecture-gallery/mhc/).

**Local exercise:** Check compression boundaries and causal visibility. Verify non-negativity and approximate row/column sums for the constrained residual mixing matrix. Test residual-stream shapes and gradients.

**Completion:** Complete generating model with correct compressed state and constrained mixing. Muon is a separate optimizer exercise after model correctness.

#### 13. Qwen3.8-Flash-Next-style model

**Build:** A complete hybrid MoE model with Qwen Sparse Attention, small n-gram embedding tables, gated residual machinery, and MTP.

**Resources:**

- [Official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- [Qwen Sparse Attention explanation and pseudocode](https://sebastianraschka.com/llm-architecture-gallery/qwen-sparse-attention/).

**Local exercise:** Implement selection over complete four-token micro-blocks and correct handling of the visible incomplete tail. Use tiny selection budgets and hash tables. Check n-gram construction at sequence boundaries and verify lookup gradients.

**Completion:** Complete model with correct block selection, lookup memory, residual behavior, and hybrid state. Report lookup-table storage separately.

#### 14. Kimi Linear → Kimi K3-style model

**Build:** A complete Kimi Linear-style model with Kimi Delta Attention and periodic MLA. Extend it toward K3 with gated MLA, Attention Residuals, latent-space experts, and its specified activation and other model-specific components. Add the vision path after text correctness.

**Resources:**

- [Kimi Linear material](https://github.com/MoonshotAI/Kimi-Linear).
- [Attention Residuals explanation and code](https://github.com/MoonshotAI/Attention-Residuals).
- [Kimi K3 specification and report](https://github.com/MoonshotAI/Kimi-K3).
- [Released K3 model card](https://huggingface.co/moonshotai/Kimi-K3).

**Local exercise:** Verify channel-wise recurrence gating, attention over earlier depth representations, and projection into/out of the expert latent space. Keep individual feature switches for numerical checks and later ablations.

**Completion:** Runnable configurations with an explicit coverage checklist against the published K3 specification. Label partial coverage accurately.

#### 15. DeepSeek V4.1-Flash-style model

**Build:** A complete small causal encoder–decoder. Implement encoder-derived shared KV state, CSA2 cache/index sharing, bounded replay, and small Engram conditional-memory tables. Begin with four encoder and four decoder layers.

**Resources:**

- [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
- [Official reference model code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).
- [Inference instructions](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md).
- [Engram implementation](https://github.com/deepseek-ai/Engram).

**Local exercise:** Trace cache ownership through encoder and decoder. Check causal masks on every path, including decoder access to encoder states. Compare cached execution and bounded replay against a simple reference. Add cache quantization only after float-state correctness; treat speculative decoding as a subsequent exercise.

**Completion:** Complete encoder–decoder model with verified cache ownership, causal behavior, and a documented coverage checklist.

Projects 14 and 15 are paper-driven capstones. Public components and reference implementations are available, but no complete, verified training code-along comparable to the early GPT tutorials was established during source review.


### Distinct research branches

#### 16. Mamba 2 → Mamba 3

**Build:** A complete state-space language model. Begin with the simpler Mamba 2/SSD reference, then implement Mamba 3's recurrence, complex-valued state tracking, and SISO/MIMO variants.

**Resources:**

- [Official Mamba repository](https://github.com/state-spaces/mamba): includes simplified Mamba 2 and minimal SSD components.
- [Tri Dao: Mamba-3 methodological deep dive](https://tridao.me/blog/2026/mamba3-part2/).
- [Community pure-PyTorch Mamba-3 implementation](https://github.com/VikramLex/mamba3-minimal).

**Target:** 1–3M parameters. Use a small state dimension and MIMO rank.

**Local exercise:** Compare sequential and chunked execution, state resets, and SISO/MIMO shapes. Use synthetic state-tracking tasks for a short learning check. Verify community code against the official mathematics/reference before using it as an authority.

#### 17. RWKV-7

**Build:** A complete recurrent language model with time/channel mixing and generalized delta-rule state updates. Implement the recurrence in ordinary PyTorch first.

**Resources:**

- [Official architecture documentation](https://github.com/RWKV/RWKV-wiki/blob/main/docs/basic/architecture.md).
- [RWKV-LM code](https://github.com/BlinkDL/RWKV-LM).
- [Pretraining tutorial](https://wiki.rwkv.com/advance/pretrain.html).

**Target:** 1–3M parameters.

**Local exercise:** Check step-wise generation against equivalent sequence execution. Measure persistent-state size and compare it with attention-cache growth. CUDA-dependent training examples are references for later execution; your initial implementation must have a CPU path.

#### 18. LLaDA → iLLaDA; optional LLaDA2.1

**Build:** A complete masked diffusion language model with bidirectional attention, corruption/masking, the appropriate masked prediction objective, and iterative generation. Add token editing as a separate LLaDA2.1 exercise.

**Resources:**

- [Official LLaDA/iLLaDA material](https://github.com/ML-GSAI/LLaDA): model/inference code and training pseudocode.
- [dLLM](https://github.com/ZHZisZZ/dllm): runnable adaptation, training, inference, and evaluation recipes.
- [LLaDA2.1 paper](https://arxiv.org/abs/2602.08676).

**Target:** 1–3M parameters, short sequences, and few denoising steps.

**Local exercise:** Check mask sampling, loss selection/weighting, immutable prompt positions during conditional generation, and token-editing behavior. Compare convergence on tiny tasks with an autoregressive baseline. Raw training losses across different objectives are not directly comparable.

The original LLaDA repository does not release the original complete training framework and data. Its pseudocode and the public dLLM recipes support an educational implementation, not an exact training reproduction.

#### 19. Gemma 4 12B-style unified model

**Build:** A complete miniature multimodal model that projects image patches and audio frames directly into the language backbone, without separate deep modality encoders.

**Resource:** [Google's Gemma 4 12B architecture guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/).

**Target:** 2–4M parameters with small images and short clips.

**Local exercise:** Verify patch/frame ordering, modality positions, mixed-sequence masks, and gradients through the projection layers. Compare interface and resource costs with your encoder-plus-projector model from project 7. Use a tiny learning check for modality-dependent behavior.

This is an independent multimodal implementation project guided by the published architecture, not an established full from-scratch training tutorial.


### Required CPU verification protocol

Every model should support the following checks at a tiny configuration.

#### Forward and backward correctness

- Assert shapes at embeddings, projections, heads, expert dispatch, residual streams, and output logits.
- Check that outputs, losses, and expected gradients are finite.
- Compare selected components with a simple reference using identical weights and inputs.
- Use FP64 finite-difference/gradient checks on very small differentiable inputs where appropriate.
- Record the numerical tolerance and maximum error. Choose tolerances based on dtype, operation, and sequence length rather than a universal threshold.

#### Causal behavior

For causal models, alter future tokens while retaining the prefix. Earlier outputs should remain unchanged within tolerance. Disable dropout and use identical evaluation settings.

Apply this check to sparse indexers, compression, convolution, n-gram construction, and encoder–decoder access as well as ordinary attention. Diffusion models intentionally use bidirectional attention; test their conditional-generation restrictions instead.

#### State and cache behavior

- Compare full-sequence, chunked, and token-by-token execution where the formulations should be equivalent.
- Test empty/new requests, state reset, sequence boundaries, and multiple batch elements.
- Match masks, position policies, and sliding windows between reference and cached versions.
- Remember that dropping history from a full-attention cache changes the available context; compare against a reference with the same semantics.

#### Short learning check

When feasible, run a small number of optimization steps to overfit a handful of examples or solve a synthetic task. Record the loss curve and held-out/task behavior where meaningful.

This checks that the model and optimization path work. It does not establish general language capability, useful reasoning, or reproduction of the original model's performance.

#### Checkpoint and configuration checks

Save/reload the state and configuration, then compare outputs. Count parameters from the actual model. Record every simplification relative to the published architecture.


### Optimization track

Keep both a clear reference implementation and an optimized implementation. Every optimization needs numerical checks before performance interpretation.

| Exercise | What to measure |
|---|---|
| Preallocated KV cache versus repeated concatenation | Allocation count, memory, prefill/decode time. |
| Sequential versus chunked recurrence | State/output/gradient agreement and runtime. |
| Token-wise versus batched expert dispatch | Routing agreement, selected-expert gradients, and runtime. |
| Separate versus combined projections | Numerical error, allocations, and runtime. |
| Tensor layout and unnecessary copies | Copy/contiguous operations and memory use. |
| Shared versus independently stored caches | Ownership correctness and bytes per token. |
| Dense versus sparse selection references | Correct selected positions, masking, and resource cost. |
| Float versus simulated quantized state | Reconstruction error and model-output changes. |

Use the [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). Record thread count, dtype, dimensions, warmup, and repeated timings. Measure prefill and incremental decoding separately.

CPU profiling establishes results for the local CPU implementation. GPU kernel performance and distributed scaling require their own later measurements. Quantized storage alone does not demonstrate faster execution.


### Training, adaptation, and reasoning projects

These complement the model implementations. Develop their algorithms and tiny local checks now; substantial runs can wait for suitable compute.

| Project | Implementation work | Resources |
|---|---|---|
| Tokenization and data | Core work now belongs to P00: byte/BPE training, encode/decode, filtering, deduplication and reproducible splits. Later models add tokenizer-specific compatibility. | [minbpe](https://github.com/karpathy/minbpe), [BPE paper](https://arxiv.org/html/1508.07909v5), [Stanford CS336](https://cs336.stanford.edu/); retained supplementary [BPE walkthrough](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html). |
| Pretraining systems | Data loading, optimizer state, gradient accumulation, checkpoints, resource accounting, and profiling. | [build-nanoGPT](https://github.com/karpathy/build-nanogpt), [LLMs From Scratch](https://github.com/rasbt/LLMs-from-scratch), CS336 assignments 1–2. |
| Instruction tuning | Chat formatting, assistant-token loss masking, and an SFT loop. | [Raschka chapter 7 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07), [Hugging Face Smol Course](https://github.com/huggingface/smol-course). |
| LoRA and DoRA | Low-rank updates, frozen/base parameters, adapter merging, and gradient checks. | [LoRA/DoRA walkthrough](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch). |
| Preference optimization | Preference pairs, policy/reference log probabilities, and DPO loss. | [DPO from scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb). |
| Reasoning and RL | Verifiers, rollout handling, GRPO, distillation, and inference-time selection. | [Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch), CS336 assignment 5. |

For a tiny untrained model, test loss equations and use simple synthetic rewards. Meaningful reasoning experiments usually need a competent pretrained starting policy. Implementing a reasoning objective does not by itself create a reasoning model.


### Recent releases: implementation priority

The strongest final architecture projects in this roadmap are the miniature Kimi K3 and DeepSeek V4.1 builds. Their selection is a learning recommendation, not a claim that they are the newest releases in every category.

- **GLM-5.3:** Its [official card](https://huggingface.co/zai-org/GLM-5.3) states that it uses the same base model as GLM-5.2 and attributes gains to post-training. Use it as inspiration for a controlled post-training study.
- **Reflection Beam:** The [October 5 release](https://reflection.ai/blog/introducing-beam) describes interleaved local/global attention and fine-grained MoE. Use its training and balancing choices as a comparison once those mechanisms are implemented.
- **Mistral Large 4:** At the cutoff date, [official documentation](https://docs.mistral.ai/resources/changelogs) lists a public preview and says open weights are coming soon. Add an implementation project after sufficient public artifacts become available.

Public code, open weights, and complete training reproducibility are different properties. Some projects provide public model/inference code without the original data or complete training pipeline. Resource descriptions above identify the relevant gaps. Model and code licenses also vary; Kimi K3, for example, releases weights under its own license.


### Source-review and verification status

The linked resources and selected release claims were researched during preparation of this roadmap on October 9, 2026. Resource availability, model cards, and public repository material were inspected. Some long-form articles are paywalled, and their public companion code is provided where available.

No model implementations, training runs, numerical checks, or performance benchmarks were executed as part of producing this document. The parameter budgets, study ordering, proposed exercises, and completion gates are recommendations for the upcoming work.

For durable research, record the exact repository commit or model revision you use when starting each project. Update the release cutoff explicitly when extending this roadmap later.
