# Research — LLM roadmap review and GPT-1 starting plan

Reviewed: 2026-10-09. Scope: roadmap assessment and Project 1 plan, without implementation/training. Lead: AI project workflow through the planning coordinator; supporting AI experiment planning after the user clarified the research objective.

## Verified workspace context

Read root and Projects AGENTS.md/README.md, the planning coordinator, shared task workflow and AI project workflow. Independent model repositories must be registered as submodules; the Projects umbrella owns this cross-project review.

- `cmp` returned 0: Projects and Research copies of `llm-implementation-research-roadmap.md` are byte-identical. Both remain untouched.
- Projects already contains modified/untracked setup, CLI and planning work. It was preserved.
- `.gitmodules` lists project_docs, linear-model-lab, retrieval-lab and attention-lab. GPT-1 has no registered repository.
- Attention Lab's README explicitly limits it to single-head forward attention. It does not provide learned projections, autograd/training, tokenization or a full Transformer.
- Local `lscpu -J`: AMD Ryzen 9 9955HX, 16 cores/32 logical CPUs. `/proc/meminfo`: approximately 36.9 GiB total RAM, 21.3 GiB available at inspection. No training throughput was measured. Subsequent read-only device inventory appears below. CPU-first learning remains practical while accelerator capability is unresolved.

## Roadmap assessment

Strengths: complete small models; explicit architectural simplifications; readable reference before optimization; causal/state/checkpoint checks; careful interpretation of training and profiling; primary resources mixed with practical teaching material.

Execution gaps: 19 projects repeat several model families; numbering differs from study order; section 10 requires optimization for every model; per-project GPT-1 completion omits global generation/checkpoint requirements; section 9's single `llm-research/models/` tree conflicts with independent implementation ownership if used literally. Parameter ranges alone do not specify step time, activation memory or experiment duration.

These are review judgments, not measured educational outcomes. The roadmap is a strong architecture-learning catalogue. After the user's clarifications, retain its breadth and hands-on training: reuse plumbing and attach systems investigations rather than replacing deep model learning with a short inference-only path.

## Primary-source evidence

Pages inspected on 2026-10-09. This checks selected claims and availability, not every linked tutorial, release history, training implementation or license.

| Source | Finding | Implication and limit |
| --- | --- | --- |
| [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), sections 3–4 | LM pretraining followed by supervised adaptation with optional auxiliary LM objective; learned positions and tied embeddings; published dimensions and BPE | Add both stages explicitly. Tiny/byte-tokenized work is educational rather than historical training reproduction. |
| [GPT-1 pinned source](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/train.py), block/model/objective | Post-residual LayerNorm, tanh GELU, classification-token pooling, shifted/masked LM loss and classification-plus-LM training loss | Preserve the specific GPT-1 contract. Combined embedding-table layout differs from the proposed modern token-only output domain. |
| [GPT-2 source](https://github.com/openai/gpt-2/blob/master/src/model.py), block/model | Normalization before sublayers and an additional final normalization | Good next project: compare the architectural delta with the same harness. |
| [Switch paper](https://arxiv.org/abs/2101.03961) | T5-based sparse-model experiments and simplified routing | A decoder-only top-1 model should be called a Switch-routing adaptation; a faithful original model needs the original backbone/objective. |
| [Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) and [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B) | Similar hybrid backbone; later card lists MTP and vision | Keep version-specific runnable configurations; compare mechanisms without duplicating the harness. Card claims do not establish that every training path is in released code. |
| [Gemma 4 E2B card](https://huggingface.co/google/gemma-4-E2B) | Per-layer embeddings; distinguishes encoder-free 12B Unified model | Keep text-core versus multimodal scope explicit. Cache-sharing implementation details were not independently audited. |
| [DeepSeek V4-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | CSA/HCA and constrained residual connections described | Integration capstone; no local correctness/performance established. |
| [Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Sparse attention, n-gram tables, gated residuals and hybrid MoE | Requires component coverage and storage accounting. |
| [Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3) | KDA/gated MLA, Attention Residuals, latent MoE and vision | Capstone built on several prerequisites. |
| [DeepSeek V4.1-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Causal encoder/decoder, projected shared KV, CSA2, replay and conditional memory | Causal visibility and cache ownership are separate contracts. |

`git ls-remote` resolved GPT-1 master to `a69b5c43b0452462890bca8ff92fb75dee9290cf`; the pinned train.py was opened. GPT-1 text_utils.py was also opened as tokenizer context. Other primary pages above are mutable; pin their revisions when their project begins.

An exploratory probe of an assumed DeepSeek-V4 GitHub URL failed; it was not a roadmap link and is not evidence against the roadmap. Its actual V4 model card was subsequently inspected.

## Assumptions and remaining questions

- Treat “then we're gonna start” as sequencing: review now, implementation next.
- Proposed byte tokenizer avoids making BPE an entry prerequisite; it must be labeled a deliberate departure.
- Research direction is now explicit: memory-constrained larger-model inference, including SSD use, weight placement and KV-cache optimization. Weekly time availability is unknown; use milestones rather than calendar estimates.
- Published model dimensions are reference metadata, not a request to allocate/train those configurations.
- Tiny parameter estimates in the plan are arithmetic for a specified proposed design; actual model counts and runtime remain unmeasured.

## Clarified goal and systems evidence

The user explicitly requires learning by implementing and training smaller models before optimizing larger ones. Architecture depth and the SSD/KV research goal are complementary. An initial suggestion to make five projects the complete required path is superseded: keep the full architecture study and interleave relevant systems work.

Read the AI experiment planning skill to predeclare protocols and metrics; no runs were authorized/executed by this review.

- [GQA paper](https://arxiv.org/abs/2305.13245), [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437), and [Mamba-2/SSD paper](https://arxiv.org/abs/2405.21060) were opened as primary mechanism references. Here SSD in the Mamba paper means structured state-space duality, distinct from solid-state-drive offloading.
- [FlexGen](https://arxiv.org/abs/2303.06865) establishes GPU/CPU/disk placement as prior work focused on throughput-oriented workloads; its [implementation](https://github.com/FMInference/FlexLLMGen) is archived. No local performance inference follows.
- [PagedAttention](https://arxiv.org/abs/2309.06180) concerns allocation/fragmentation/sharing. [KIVI](https://arxiv.org/abs/2402.02750) concerns KV quantization; those are distinct from active disk paging.
- [Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling), [llama.cpp](https://github.com/ggml-org/llama.cpp) and [LMCache disk storage](https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html) supply candidate baselines. LMCache's example compares repeated-prefix TTFT; do not treat it as proof of active-context capacity.
- Read-only hardware inventory found Samsung 980 PRO 1TB and Micron 3400 512GB NVMe devices. The Projects mount is ext4 on /dev/nvme1n1p1. PCI display devices are AMD and Intel; NVIDIA management failed to connect to a driver. Exact usable accelerator/backend/VRAM is unresolved. Disk free space and bandwidth were not measured.

See [the connected research track](07-inference-research-track.md) for the proposed protocols, prior-work limits and ownership.

Research is sufficient for the review and first-project plan. Starting an experiment requires child-owned planning, locked dependencies, fixtures and a bounded compute protocol.
