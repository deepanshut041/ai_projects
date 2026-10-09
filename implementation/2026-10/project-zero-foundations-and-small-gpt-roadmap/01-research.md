# Research — Project Zero foundations and small GPT roadmap

Updated: 2026-10-09

## Request, ownership and actual state

The user asks for a ground-up Project Zero before a small GPT-style model and updates to the roadmap and related records. Lead: AI project workflow; supporting workflow: source review. This is a Projects learning plan and documentation change. No model implementation, training, hosted session or external publication is requested in this turn.

Canonical resume/progress currently start at P01 GPT-1 / P01-M1 not-started. No gpt-lab exists. Existing attention-lab is a single-head forward seed, not a trained Transformer. Existing P01 tutorial and earlier plans are useful historical references. Local documentation, umbrella and CLI edits already exist; preserve them. New learning IDs must not renumber P01–P19.

## Primary-source evidence inspected

| Source | Inspected material | Finding and implication |
| --- | --- | --- |
| [Attention Is All You Need, v7](https://arxiv.org/html/1706.03762v7) | Figure 1; §§3.1–3.5, §5 | Encoder–decoder architecture; masked target self-attention and source cross-attention. Use a complete miniature conditional model in P00, with post-norm, sinusoidal positions and ReLU. Reduced dimensions, synthetic tasks and training defaults are educational departures, not WMT reproduction. |
| [Sennrich et al., v5](https://arxiv.org/html/1508.07909v5) | §3.2, Algorithm 1 | Frequency-based symbol merges learned on training data. Its word/character segmentation differs from our proposed UTF-8 byte baseline. |
| [minbpe](https://github.com/karpathy/minbpe) | README and basic tokenizer explanation | Readable byte-BPE train/encode/decode/save/load reference. Our reserved-ID/tie-break/boundary policies are explicit project contracts, not claimed minbpe compatibility. |
| [Harvard Annotated Transformer](https://github.com/harvardnlp/annotated-transformer) | SublayerConnection, Encoder/DecoderLayer, data_gen, greedy_decode | Useful component/copy-task teaching reference. Its code moves LayerNorm before the sublayer for simplicity; do not copy that and claim paper post-norm parity. The hosted HTML failed to load; pinned repository source was inspected successfully. |
| [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | Computational graph, backward, gradient accumulation | Supports a bounded tensor/gradient/optimizer primer. Building an entire tensor/autograd engine is unnecessary for this learning objective. |
| [build-nanoGPT](https://github.com/karpathy/build-nanogpt) | Block, GPTConfig, GPT | Pre-norm decoder, learned positions and GELU are a suitable P01 reference. Its large/distributed training settings are not our run defaults. |
| [Stanford CS336 assignment 1](https://github.com/stanford-cs336/assignment1-basics) | README, setup/test/data sections | Optional exercise/test reference for language-model foundations. Its complete assignment and bulk corpus downloads are not prerequisites for P00. |

Exact inspected repository revisions and byte hashes are in [source-provenance.json](source-provenance.json). Papers are versioned separately. No upstream code was copied into a model implementation or executed.

## Recommendation and alternatives

Choose **P00 Tokenization and Transformer Foundations**, proposed independent `transformer-foundations-lab`, with six bounded milestones: tensor/autograd primer; character/byte/BPE tokenizer; attention/block primitives; full encoder–decoder; synthetic task training with persistent recovery; inspection and transfer to P01. The end product is a trained conditional sequence model plus reusable tokenizer contracts, not a folder of disconnected snippets.

A tokenizer-only P00 would leave gradients, masks and cross-attention untested in training. A decoder-only P00 would repeat P01 and omit the user's requested original Transformer route. Full WMT translation or a new autodiff framework would expand data/compute and distract from fundamentals. Keep real translation, beam search and production tokenizers optional.

P01 becomes **Small GPT Language Model**, using a simple pre-norm decoder and the P00 tokenizer/data/recovery lessons. Exact GPT-1, historical classification adaptation and benchmark reproduction become optional reference extensions. P02 strengthens GPT-2-specific coverage, tokenizer compatibility, checkpoint mapping and cache behavior rather than rebuilding BPE from scratch.

## Design assumptions and limits

- Python/PyTorch is the first complete training path; selected C++/Python component traces build understanding. No mandatory duplicate C++ full-model port.
- P00 joins the existing P01–P04 Colab Free and full-state recovery requirement; no hosted capability is established by docs checks.
- Initial CPU/Free sizes and task thresholds are proposed. Measure and finalize them in the child before training. Failed gates stay pending; no forced success claim.
- Synthetic held-out copy accuracy establishes sequence-task behavior only. BPE token counts are compression observations, not evidence of better LM quality. Learner explanation remains unassessed.
- Preserve historical plans/tutorial assets and later model notes; canonical live docs supersede the GPT-1-first sequence.

Created: 2026-10-09T23:21:45.966765+00:00
Task: 2026-10/project-zero-foundations-and-small-gpt-roadmap
Selected skills: ai-project-workflow, ai-paper-review
State: scaffold; sources and conclusions have not been assessed.

## Request and execution scope

Recommend a ground-up Project Zero covering tokenization and Attention Is All You Need implementation before a small GPT-style language model, then update the roadmap and related state docs.

Record whether the actual user instruction authorizes planning only or also execution. Scaffolding does not grant permission.

## Verified context and sources

| Source/path and version | Finding | Impact on the task | Limitation |
| --- | --- | --- | --- |

## Requirements, constraints, and non-goals

Not assessed yet. Include destination, available inputs, budget, and applicable repository instructions.

## Assumptions and open questions

Separate evidence-backed facts from assumptions; identify which dependent action each material question affects.

## Research readiness

Not assessed. Record enough evidence to choose the next approach; record missing access or evidence explicitly.
