# P01 sources, choices and proof boundaries

Prepared 2026-10-09–10. Required scope follows the current Projects roadmap: complete small pre-norm GPT text LM; M1–M3/M5 required; GPT-1/classification and early cache optional. This tutorial is original teaching material with a checked educational reference, not a claim of exact pretrained GPT-2 compatibility.

## Inspected primary material

| Source | Pin / inspected part | Role and limit |
| --- | --- | --- |
| [build-nanogpt](https://github.com/karpathy/build-nanogpt/blob/6104ab1b53920f6e2159749676073ff7d815c1fa/train_gpt2.py) | 6104ab1b53920f6e2159749676073ff7d815c1fa; CausalSelfAttention, MLP, Block, GPT | Pre-norm/learned-position/tied-head/activation reference; no large or distributed defaults executed |
| [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | 2.14 Boolean allowed-mask and dropout contract | Independent attention value/gradient comparison in local tests |
| [PyTorch GELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html) | Exact and tanh variants | Declare the activation instead of silently mixing variants |
| [Checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | General model/optimizer state | Workspace adds full RNG/data/identity/persistent-generation protocol |
| [Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) | Version/platform/RNG limits | Same-environment checks do not assert cross-device identity |
| [Project Gutenberg Alice](https://www.gutenberg.org/ebooks/11) | UTF-8 ebook 11; 174311 raw bytes; raw SHA in gpt_data.py | Catalog labels public domain in USA; preserve terms/provenance; one-book task only |
| [Colab FAQ](https://research.google.com/colaboratory/faq.html) | Free resources/runtime lifetime/Drive operations | Hosted availability and deleted-runtime recovery remain unmeasured |

The P00 ByteBPE source is copied byte-for-byte into gpt_tokenizer.py, SHA256 `5adcd5044d52d541dc729554a3f6d184ff0d53d2704122d0b37f4dd51e6db190`. Original workspace path: `ai_projects/docs/tutorials/projects/p00/transformer-foundations/v1/examples/byte_bpe.py`. Its explicit artifact/API contract is preserved without a live sibling import. A learner's own tokenizer may replace it only after compatibility checks and a new recorded source identity. No external tokenizer code was vendored.

## Declared design departures

| Choice | This tutorial |
| --- | --- |
| Model scale | 2 layers, D64/H4/F256/T64/V260; 120832 unique parameters |
| Inputs/task | One-book byte-level text LM, chapter split; no encoder/cross-attention or pretrained weights |
| Blocks | Pre-norm, learned positions, final LayerNorm, tanh GELU, tied bias-free vocabulary projection |
| Attention | Explicit causal/PAD reference; optional independent SDPA comparison; no KV cache |
| Training | FP32, one process/thread for CPU checks, AdamW with matrix decay, valid-token accumulation, fixed warmup/cosine schedule |
| Batching | Document-bounded non-overlapping target windows, reset context, random window sampling with replacement |
| Evaluation | Token-weighted validation on chapter X; add-one training-unigram baseline; test XI–XII untouched |
| Generation | Cropped context with reset positions, structural output bans except EOS, temperature/top-k, UTF-8 replacement display |
| Recovery | Complete state at completed-update boundaries, verified immutable generations, fallback and retention; strict same identity |

Each simplification is a declared protocol choice. The math traces are authored arithmetic; source/API statements are documented evidence; the [reference results](project/reference-results.json) record executed local behavior; learner understanding and hosted survival remain unassessed. Raw corpus, weights and generated runs remain outside Git.

## Reproduction and remaining work

Follow lessons in order and use the [reference-run record](project/reference-runs.md). Source digests in the compact results identify executed files, including dirty edits; after a source change rerun the affected checks and record a new identity. The complete manuscript is not a substitute for the independent child's source/runtime lock and evidence.

The current canonical Colab contract lives at `ai_projects/docs/guide/COLAB.md` in the workspace. Copy its common-trainer, persistent-state, retention/failure and deleted-runtime gates into the child package at kickoff. The [readiness record](project/readiness.md) makes the unresolved tasks explicit.
