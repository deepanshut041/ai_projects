# Translation Transformer — implementation workbook

This build turns English into French with a source encoder, a causal target decoder and cross-attention. Use the [Lesson 1 — Model specifications and task contracts](lessons/01-start.md) for the explanation and calculations; return here to choose the next file, run a check and record what actually works.

The existing `attention-is-all-you-need` implementation project already has its attention component checked and committed at `475c634`. Tokenizer, data, model, trainer and translation entry points still need their learner implementation. The course's complete miniature references are separate executable examples; their successful runs do not complete those project files or assess understanding.

## Keep the existing file names

The project is under `ai_projects/projects/transformer-labs/attention-is-all-you-need/` from workspace home. Its top-level launch files stay `attention.py`, `tokenizer.py`, `prepare_data.py`, `train.py`, `inference.py`, `compute_bleu.py`. Shared implementation stays in `src/attention_is_all_you_need/`. Continue those files rather than creating a second translation repository or replacing the checked attention component with a downloaded tutorial file.

| Complete course reference | Existing project destination | Responsibility |
| --- | --- | --- |
| [translation_attention.py](examples/translation_attention.py) | `src/attention_is_all_you_need/attention.py`; top-level `attention.py` | Inspect Q/K/V, packing, masks and SDPA; keep the checked project implementation and final interface |
| [translation_data.py](examples/translation_data.py): vocabulary and WordPiece preparation | `src/attention_is_all_you_need/tokenizer.py`; top-level `tokenizer.py` | Train/load tokenizer artifacts; expose special IDs, vocabulary size, encode and decode |
| [translation_data.py](examples/translation_data.py): pairing, encoding and collation | `src/attention_is_all_you_need/data.py`; top-level `prepare_data.py` | Validate parallel records, preserve splits and construct the five batch tensors |
| [translation_model.py](examples/translation_model.py) | `src/attention_is_all_you_need/config.py` and `model.py` | Declare compatible config, embeddings, positions, FFN, encoder/decoder blocks and prediction head |
| [translation_train.py](examples/translation_train.py) | `src/attention_is_all_you_need/trainer.py`; top-level `train.py` | Own update count, optimizer/scheduler, token-weighted loss, validation and run identity |
| [course_checkpoint.py](examples/course_checkpoint.py) | `src/attention_is_all_you_need/checkpoint.py` | Save verified full-state generations, reject incompatible continuation and retain rollback |
| `Transformer.inference` in [translation_model.py](examples/translation_model.py) | `src/attention_is_all_you_need/model.py`; top-level `inference.py` | Encode once, decode prefixes, finish batch rows independently and print decoded French |
| [translation_metrics.py](examples/translation_metrics.py) | Top-level `compute_bleu.py` | Evaluate a declared held-out corpus and retain metric settings, examples and error notes |

Inside the package, import attention with `from .attention import Attention`. Keep `Attention(embedding_dimension=...,num_heads=...,attn_dropout=...)`, `forward(src,tgt=None,attention_mask=None,causal=False)`, and `q_proj/k_proj/v_proj/out_proj`. A config-to-attention adapter supplies those values; changing the attention constructor to accept a config directly would break the existing callers.

Python files own model/data/training logic. The concept notebook calls the existing attention functions; drawings, widget construction and auxiliary display calculations stay in `attention_visuals.py`. A Colab notebook should likewise import and call the trainer rather than contain a second implementation.

## Establish the current baseline

From workspace home, `./ai run attention-is-all-you-need` runs the existing attention demonstration. From the implementation project's directory, `uv run --frozen pytest -q` checks the attention source; `uv run --frozen --extra notebook pytest -q` also checks the numerical matrix helpers. These are the existing component checks, not complete translation training.

For the complete course reference, follow [Lesson 2 — Reproducible source and implementation workspace](lessons/02-workspace.md) to select the frozen interpreter and enter the course `examples/` directory. The `python` commands below mean that selected interpreter. The reference files are already available; the implementation destinations above remain your build work.

```bash
python translation_attention.py
python translation_data.py
python translation_model.py
python translation_checks.py
```

The attention command prints explicit projection/score/context matrices. The data command prints an uneven batch with source padding ID zero, target padding ID one and ignored target labels. The model command prints logits `(6,5,12)`, `30796` parameters and finite loss/nonzero head gradient. The complete reference checks cover behavior and independent arithmetic, not only tensor shapes.

## Build and verify one stage at a time

| Stage | What to implement or inspect | Check before continuing | Evidence to save |
| --- | --- | --- | --- |
| 1. Learned retrieval | Keep checked `attention.py`; explain `XW_Q`, `XW_K`, `XW_V`, score rows/key columns and `A@V` | [Lesson 9 — Learned Q/K/V and scaled attention](lessons/07-qkv.md) known-answer dot-product/value test; predict a changed payload without changing weights | One worked matrix calculation and the corresponding printed tensors |
| 2. Head and visibility contracts | Retain packed heads and true-means-allowed masks | [Lesson 10 — Multi-head attention implementation](lessons/08-heads.md): head identity/round-trip, explicit/SDPA parity, future-token and padded-key perturbation tests | Shapes plus one changed-input behavioral result |
| 3. Tokenizer artifacts | Fill `tokenizer.py`; use separate English/French ID spaces and lookup actual special IDs | [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md): NFC equivalence, BOS/EOS exactly once, encode/decode round-trip, saved-artifact reload | Tokenizer JSON/config/version/hash, vocabulary count and special-ID map |
| 4. Paired dataset | Fill `prepare_data.py`/package `data.py`; split and filter pairs together | Unequal parallel files raise; a blank side drops the entire pair; no independently truncated translations | Acquisition/terms/checksum, line and pair counts, filtering reasons and split identity |
| 5. Collator | Implement source padding, shifted target input, input masks and `-100` labels | Exact uneven batch below; labels do not alias/mutate input tensors | Input/mask/label printout and supervised-token count |
| 6. Embeddings and blocks | Fill config/model; sinusoidal positions, GELU FFN, residual/post-norm encoder and decoder | [Lesson 14 — Residual and normalization blocks](lessons/11-transformer-blocks.md) known position vectors; show LayerNorm on `[1,3]`; preserve dimensions | Config and small numerical probes |
| 7. Complete model | Connect source encoder, target decoder, cross-attention and vocabulary head | [Lesson 17 — Assemble the complete translation model](lessons/12-translation-model.md): logits, finite loss and gradients reaching source embeddings/encoder/cross-attention/head; whole-model future isolation | Source/config identity and loss/gradient observations |
| 8. Bounded training | Fill trainer/top-level train; sum/count reductions, one optimizer step per accumulation group, scheduler after update | [Lesson 18 — Translation optimization and complete training updates](lessons/13-translation-training.md): unequal microbatches match a full-batch update; tiny fit meets declared thresholds | Initial/final loss, counts, update count, configuration and generated examples |
| 9. Free translation | Add `Transformer.inference` and top-level inference; preserve all source masks and per-row completion | [Lesson 19 — Autoregressive generation, completion and decoding scores](lessons/14-translation-generation.md): controlled batch EOS test, budget includes BOS, meaningful trained output | IDs and text, EOS/truncation status, source/reference/prediction comparisons |
| 10. Recovery | Fill package checkpoint and wire completed-update saves into trainer | [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md): fresh-process continuation, corrupt/incomplete fallback, compatibility rejection and two-good-generation retention | Selected generation, verified hashes, restored update and comparison result |

Do not move on from a wrong shape/label/gradient by increasing model size. Reproduce the smallest failing input, inspect the named axis or state owner, and keep the failing case as a check.

## Use the uneven batch as the collator contract

For source rows `[2,4,3]`, `[2,5,6,3]` and target rows `[2,8,9,3]`, `[2,7,3]`, with source pad zero and target pad one, require:

```text
src_input_ids [[2,4,3,0], [2,5,6,3]]
src_pad_mask  [[1,1,1,0], [1,1,1,1]]
tgt_input_ids [[2,8,9],   [2,7,3]]
tgt_pad_mask  [[1,1,1],   [1,1,1]]
tgt_outputs   [[8,9,3],   [7,3,-100]]
```

There are five supervised labels. The second target's last input is EOS even though its next label is ignored. The causal/input mask controls information access; the label mask controls which errors count. Setting the input EOS to padding to “match” the ignored label would change the model's context incorrectly.

## Declare a small run before scaling up

Start with the reference's six training pairs and two held-out combinations. The profile is CPU/FP32, width32, four heads, one encoder and decoder layer, FFN ratio four, source/target capacities32, dropout zero and 400 updates. Its target vocabulary has twelve entries; each of the six training rows supplies five labels. AdamW's fixture learning rate `.01` and short cosine warmup are diagnostic choices, not the large-run settings.

From course `examples/`:

```bash
python translation_train.py --steps 400 --seed 7
```

The recorded reference result has initial loss `2.5181`, final loss approximately `0.000138`, training token accuracy one, and exact generated training sentences. Both held-out combinations also decoded correctly. The training check requires loss below one tenth of the initial loss, token accuracy one and the exact six training outputs. These are fixed diagnostic thresholds. A failed learner run keeps its result and threshold; investigate before changing the target.

The held-out combinations test composition within the same tiny vocabulary. They do not estimate real-world translation quality or reproduce the paper's WMT score. A larger run needs a fixed test corpus, artifact identities, a declared decoding policy, SacreBLEU signature and inspected error samples. The source's random corpus holdout is validation, not automatically an official WMT test set. [Source/departure map](project/source-map.md).

The existing `configs/local.json` and `configs/colab-free.json` are proposed profiles. Their names differ from `TransformerConfig`: map `embedding_dim→embedding_dimension`, `attention_heads→num_attention_heads`, `encoder_layers→encoder_depth`, `decoder_layers→decoder_depth`, and token limits to `max_src_len/max_tgt_len`. `feed_forward_dim=128` at width32 gives ratio four. Derive null vocabulary sizes from the saved tokenizers. Implement and validate this loader; a proposed JSON file is not proof that the CLI consumes it correctly.

## Profile the small model before choosing a larger one

Measure the exact profile you will run: token lengths, real-token count, width/depth/heads, dtype, device, batch/accumulation, package versions and attention path. Report time per **optimizer update**, not an ambiguous “step” that sometimes means one microbatch. Warm up first so import/setup and first-call overhead do not dominate the timed interval.

This CPU probe runs from course `examples/` and leaves no weights or trace files:

```python
import time
import torch
from translation_train import make_state, update

def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0

torch.set_num_threads(1)
state = make_state(seed=7, steps=40, device="cpu")
for _ in range(3):
    update(state)
start = time.perf_counter()
for _ in range(20):
    update(state)
elapsed = time.perf_counter() - start
print("seconds_per_update", elapsed / 20)
print("model_state_tensor_bytes", tensor_bytes(state.model.state_dict()))
print("optimizer_state_tensor_bytes", tensor_bytes(state.optimizer.state_dict()))
print("real_labels_per_update", int((state.train_batch["tgt_outputs"] != -100).sum()))
```

The final count must be thirty. State-byte counts describe only the tensors retained in those state dictionaries. They exclude Python overhead, temporary activations, gradients, kernel workspaces and dataset storage, so they are not peak RAM or VRAM. Record peak process RAM for the CPU/Colab pilot as well; on a GPU use measured peak allocation and synchronized timing rather than reusing this unsynchronized CPU timer.

The conceptual attention score count is `B×H×T×S`; decoder self-attention uses `S=T`. Doubling self-attention length quadruples that count. Increasing vocabulary enlarges both its embedding and prediction head. Increasing FFN width adds parameter and activation memory. Measure those changes one at a time. A successful CPU SDPA call does not establish a GPU FlashAttention kernel or speedup.

## Recover a complete run, then test persistent storage

The trainer must distinguish a new run from a resume. Save at a completed optimizer/scheduler update boundary, with no partial microbatch accumulation pending. Record model/config/tokenizer/corpus identity, model buffers, optimizer moments/groups, scheduler, completed update, RNG and data order/cursor. The fixed reference batch needs only completed-update progress; an actual shuffled corpus needs its sampler/permutation/cursor or a tested reconstruction rule.

Run the local reference recovery drill:

```bash
python course_recovery_checks.py --project translation
```

It compares uninterrupted training with a new-process continuation and checks incomplete/corrupt newest generation fallback, incompatible identity rejection and retention of two verified generations. A copied weight file alone cannot pass that contract. The [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md) shows the exact `save_generation` and `load_latest` calls to map into package `checkpoint.py`.

Keep generated tokenizers, datasets, weights, checkpoints and reports under the implementation project's ignored `data/`, `artifacts/`, `weights/`, `checkpoints/` and `runs/` areas. Export only source/configuration and reproducible acquisition instructions. A source ZIP must include its file hash manifest and exclude those generated artifacts. Notebook cells launch the same trainer and save functions used locally.

Colab Free readiness requires an actual bounded pilot, persistent full-state save and fresh-runtime recovery: verify the saved bytes, use **Disconnect and delete runtime**, rebuild the same source/environment, remount persistent storage, load the newest compatible verified generation, complete further updates and save again. Show rollback from a deliberately incomplete/corrupt newest generation. A local subprocess test or a mounted Drive folder does not establish that hosted drill.

## Keep the evidence states separate

| Evidence | Current state |
| --- | --- |
| Existing project's attention implementation | Implemented, locally checked and committed at `475c634` |
| Complete miniature translation reference | Forward/loss/backward, bounded training, generation and independent CPU checks executed |
| Tokenizer/subword arithmetic | Offline vocabulary checks and optional real WordPiece save/reload probe executed |
| Reference full-state recovery | Exact new-process translation continuation, corrupt/incomplete fallback, incompatible identity rejection and two-generation retention checked locally |
| Later implementation-project tokenizer/data/model/trainer/inference files | Learner build pending; the tutorial did not fill them |
| Learner explanation of tensors, masks, shift, objective and recovery | Unassessed; use the lesson prediction/reveal questions and [readiness record](project/readiness.md) |
| Real parallel-corpus quality, resource fit and distributed behavior | Pending measured runs under declared artifacts/settings |
| Colab Free deleted-runtime continuation | Pending hosted evidence |

The next incomplete translation implementation task is tokenizer and paired-data preparation. Keep the independent foundations checkpoint unchanged; completing authored references is not the same as completing its learner gates. [Reference runs](project/reference-runs.md) records executed author results; [implementation overview](project/implementation.md) connects both projects; [readiness](project/readiness.md) names the evidence needed before Course 2.
