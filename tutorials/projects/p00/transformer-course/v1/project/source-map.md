# Sources, topic map and implementation choices

The course uses five source videos for two builds: an English-to-French encoder–decoder and a RoBERTa encoder trained with masked language modeling, then adapted to extractive question answering. The lessons follow the dependencies between concepts. One video can inform several lessons, and a lesson can combine explanations from several sources.

The first six topics specify tasks, tensor interfaces, losses, tokenizers and corpus ownership after Tutorial 0. The expanded applied topics develop embedding/output gradients, sinusoidal indexing, attention backward, exact residual/FFN ordering, complete model stacks and task-specific evaluation. Lessons 7–21 build translation and recovery; Lessons 22–30 build RoBERTa/QA; [Lesson 32 — explain the complete models and prepare the Small GPT handoff](lessons/24-ready-for-gpt.md) compares their objectives and prepares the transition to Course 2. The videos remain a source index, not five separate tutorials.

## The source revision

The implementation reference is Priyam Mazumdar's MIT-licensed **PyTorch-Adventures**, pinned to commit `2f18a5470c4ef0a3d413dd7b18d07941f45779e9`. The attention/translation sources and retained captions were inspected alongside the two RoBERTa sources on 2026-10-12. The inspected code, rather than a moving `main` URL, defines the source behavior described below.

| Pinned source area | Exact repository path | Used files |
| --- | --- | --- |
| [Attention](source:attention-code) | `PyTorch for Transformers/Attention Mechanisms/Attention/` | `attention_mechanism.ipynb` and `README.md` |
| [Translation](source:translation-code) | `PyTorch for NLP/Seq2Seq for Neural Machine Translation/` | `tokenizer.py`, `prepare_data.py`, `data.py`, `model.py`, `train.py`, `inference.ipynb`, `compute_bleu.py`, `README.md` |
| [RoBERTa and QA](source:roberta-code) | `PyTorch for NLP/RoBERTa for Masked Language Models/` | `utils.py`, `prepare_data.py`, `model.py`, `pretrain_roberta.py`, `finetune_roberta_qa.py`, `inference.py`, `evaluate_squad_score.py`, `README.md` and launch scripts |

The downloadable examples include the applicable [license notice](examples/LICENSE.txt). The arithmetic, diagrams, explanations and bounded fixtures in this course were written for this route. Reference code and architecture influence are recorded here; original captions, datasets, model weights and copied source images are not course teaching assets.

## Attention source → several connected lessons

[Attention is All You Need: Ditching Recurrence for Good!](source:video-attention), video `JXY5CmiK3LI`, develops attention from a weighted retrieval to the full configurable class.

| Source chapter/time window | Concept carried into the course | Lesson destination |
| --- | --- | --- |
| `00:03:19–00:25:00` | Query/key matches, normalized weights and value mixtures | [Lesson 1 — Model specifications and task contracts](lessons/01-start.md), [Lesson 3 — Model tensor interfaces and resource budgets](lessons/03-tensors.md), [Lesson 9 — Learned Q/K/V and scaled attention](lessons/07-qkv.md) |
| `00:25:00–00:34:20` | Attention directly on input vectors without learned projections | [Lesson 9 — Learned Q/K/V and scaled attention](lessons/07-qkv.md) |
| `00:34:20–00:44:50` | Multidimensional linear maps and learned single-head attention | [Lesson 3 — Model tensor interfaces and resource budgets](lessons/03-tensors.md), [Lesson 9 — Learned Q/K/V and scaled attention](lessons/07-qkv.md) |
| `00:44:50–01:26:30` | Separate heads, packed projections, reshape/transpose and batched multiplication | [Lesson 10 — Multi-head attention implementation](lessons/08-heads.md) |
| `01:26:30–02:17:30` | Padding, causality and their intersection | [Lesson 11 — Padding, causal and cross-attention masks](lessons/09-masks.md), [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md) |
| `02:17:30–02:41:10` | French queries retrieving English keys/values; rectangular masking | [Lesson 11 — Padding, causal and cross-attention masks](lessons/09-masks.md), [Lesson 17 — Assemble the complete translation model](lessons/12-translation-model.md) |
| `02:41:10–03:01:00` | Efficient SDPA and final `Attention` interface | [Lesson 10 — Multi-head attention implementation](lessons/08-heads.md), [Lesson 11 — Padding, causal and cross-attention masks](lessons/09-masks.md) |

The course keeps `q_proj`, `k_proj`, `v_proj`, `out_proj` and the final `src/tgt/attention_mask/causal` interface. It adds explicit numbered matrix products and independent numerical/behavioral checks. The existing implementation project already contains the locally verified attention component; its later translation files remain separate work.

## Translation data source → tokenizer, split and batch contracts

[Prepping Data for Neural Machine Translation](source:video-translation-data), video `UkBggwQsbF8`, shows how raw parallel text becomes two tokenized ID sequences and one training batch.

| Source chapter/time window | Concept carried into the course | Lesson destination |
| --- | --- | --- |
| `00:02:00–00:30:00` | French WordPiece, NFC/lowercase, special IDs, templates, encode/decode | [Lesson 5 — Source and target tokenizers](lessons/05-tokenization.md), [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md) |
| `00:30:00–00:41:35` | Parallel corpora, joining aligned columns and splitting | [Lesson 6 — Corpus splits and task visibility](lessons/06-objectives.md), [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md) |
| `00:41:35–00:56:00` | English BERT tokenizer, French tokenizer, lengths and stored token IDs | [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md) |
| `00:56:00–01:08:35` | Target next-token format and the input/output offset | [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md), [Lesson 18 — Translation optimization and complete training updates](lessons/13-translation-training.md) |
| `01:08:35–01:31:50` | Padding, source/target masks, ignored target padding and collation | [Lesson 11 — Padding, causal and cross-attention masks](lessons/09-masks.md), [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md) |

The larger source path uses Common Crawl, Europarl, Giga French and UN parallel text from the [WMT14 translation task](source:wmt14). The inspected preparation script creates a random `.005` holdout; that split is not the official WMT test set. The course explains pair-level cleaning and train-only tokenizer fitting before claiming validation or test quality. Real corpus acquisition, terms and a named evaluation split remain implementation work.

## Translation model source → blocks, optimization and generation

[Build + Train the Transformer for Neural Machine Translation!](source:video-translation-model), video `DOqg_QxInd8`, assembles the pieces and trains the translation objective.

| Source chapter/time window | Concept carried into the course | Lesson destination |
| --- | --- | --- |
| `00:02:08–00:11:40` | Config and the missing position/order information | [Lesson 6 — Corpus splits and task visibility](lessons/06-objectives.md), [Lesson 14 — Residual and normalization blocks](lessons/11-transformer-blocks.md) |
| `00:11:40–00:30:00` | Sinusoidal positions, frequency columns and slice/add | [Lesson 8 — Sinusoidal position implementation](lessons/08-translation-positions.md) |
| `00:30:00–00:37:00` | Separate source/target token embeddings and positions | [Lesson 7 — Translation embeddings and vocabulary projection](lessons/07-translation-embeddings.md), [Lesson 8 — Sinusoidal position implementation](lessons/08-translation-positions.md) |
| `00:37:00–00:54:56` | Reusing the final attention calculation in self/cross roles | [Lesson 10 — Multi-head attention implementation](lessons/08-heads.md), [Lesson 17 — Assemble the complete translation model](lessons/12-translation-model.md) |
| `00:54:56–01:01:36` | Expanded FFN and GELU | [Lesson 14 — Residual and normalization blocks](lessons/11-transformer-blocks.md) |
| `01:01:36–01:21:10` | Residual connections, LayerNorm, post-norm encoder and decoder | [Lesson 14 — Residual and normalization blocks](lessons/11-transformer-blocks.md) |
| `01:21:10–01:35:23` | Embeddings, repeated blocks and target vocabulary projection | [Lesson 17 — Assemble the complete translation model](lessons/12-translation-model.md) |
| `01:35:23–01:54:38` | BOS prefix, last-position prediction, argmax, EOS and inference debugging | [Lesson 19 — Autoregressive generation, completion and decoding scores](lessons/14-translation-generation.md) |
| `01:54:38–02:43:51` | AdamW, schedule, accumulation, Accelerate, validation and checkpointing | [Lesson 4 — Translation likelihood and supervised losses](lessons/04-learning.md), [Lesson 18 — Translation optimization and complete training updates](lessons/13-translation-training.md), [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md), [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md) |
| `02:44:21–02:45:15` | Inspecting a trained translation | [Lesson 19 — Autoregressive generation, completion and decoding scores](lessons/14-translation-generation.md) |

`compute_bleu.py` in the pinned translation subtree supplies the separate metric path. [Lesson 19 — Autoregressive generation, completion and decoding scores](lessons/14-translation-generation.md) derives clipped corpus precision and brevity penalty, then supplies a declared [SacreBLEU](source:sacrebleu) adapter. The source's `evaluate` BLEU path and the course's explicit settings are not assumed numerically interchangeable.

## RoBERTa pretraining source → a second architecture and objective

[Lets Reproduce RoBERTa from Scratch!](source:video-roberta), video `Fzv2lSt0Ing`, changes the data/objective to masked-token prediction with a bidirectional encoder. It is a RoBERTa source, not another translation video.

| Source chapter/time window | Concept carried into the course | Lesson destination |
| --- | --- | --- |
| `00:01:45–00:25:00` | Wikipedia/BookCorpus preparation, tokenizer and sequence grouping | [Lesson 22 — BERT-to-RoBERTa design and bidirectional objectives](lessons/16-bert-to-roberta.md), [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md) |
| `00:25:00–00:35:00` | RoBERTa configuration and token/position capacities | [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md), [Lesson 25 — bidirectional encoder and tied MLM head](lessons/19-roberta-encoder-and-mlm-head.md) |
| `00:35:00–00:44:15` | Padding and collator masks | [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md), [Lesson 24 — dynamic MLM corruption and selected-target supervision](lessons/18-dynamic-mlm-corruption.md) |
| `00:44:15–01:12:25` | Dynamic masking, selected labels, 80/10/10 actions and library collator comparison | [Lesson 24 — dynamic MLM corruption and selected-target supervision](lessons/18-dynamic-mlm-corruption.md) |
| `01:12:25–02:08:14` | Token/learned-position embeddings, attention/FFN blocks and MLM head | [Lesson 25 — bidirectional encoder and tied MLM head](lessons/19-roberta-encoder-and-mlm-head.md) |
| `02:08:14–02:58:45` | MLM trainer, loss, accumulation and evaluation | [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md), [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md) |
| `02:58:45–03:07:30` | Testing training and inspecting masked-token results | [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md) |

The paper, [RoBERTa: A Robustly Optimized BERT Pretraining Approach](source:roberta-paper), gives the broader training recipe. The source prepares Wikipedia and BookCorpus; the paper's larger corpus includes CC-News, OpenWebText and Stories as well. The course uses original tiny documents and a small byte-BPE implementation for offline mechanics. Those scopes must stay separate.

## Extractive QA source → spans, transfer and answer decoding

[Extractive Question Answering: Getting RoBERTa To Answer Questions](source:video-qa), video `_FVwYFICYuE`, adds a span head to the pretrained encoder. Its task returns text from the supplied context; it is not open-ended chat generation.

| Source chapter/time window | Concept carried into the course | Lesson destination |
| --- | --- | --- |
| `00:01:00–00:03:20` | What an extractive answer is | [Lesson 22 — BERT-to-RoBERTa design and bidirectional objectives](lessons/16-bert-to-roberta.md), [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md) |
| `00:03:20–00:09:30` | SQuAD examples, answer text and character offsets | [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md) |
| `00:09:30–00:44:40` | Sequence IDs, offset mapping and truncated answer handling | [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md) |
| `00:44:40–00:54:50` | QA output head and pretrained backbone choices | [Lesson 29 — QA endpoint head, loss and fine-tuning](lessons/22-qa-head-and-fine-tuning.md) |
| `00:54:50–01:13:30` | Start/end logits and loss | [Lesson 29 — QA endpoint head, loss and fine-tuning](lessons/22-qa-head-and-fine-tuning.md) |
| `01:13:30–01:20:10` | Fine-tuning the task model | [Lesson 29 — QA endpoint head, loss and fine-tuning](lessons/22-qa-head-and-fine-tuning.md), [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md) |
| `01:20:10–01:28:10` | Decoding and displaying an answer | [Lesson 30 — constrained span decoding and QA metrics](lessons/23-span-decoding-and-metrics.md) |

The pinned `evaluate_squad_score.py` is the scoring companion. The course's QA fixtures check offset ownership, windows, ignored missing-window labels, compatible backbone transfer and valid span decoding. They do not establish held-out SQuAD quality or an unanswerable-question classifier.

## Corrections with an observable check

| Inspected risk or misleading description | Course behavior | Check/explanation |
| --- | --- | --- |
| Early attention examples depend on notebook-global dimensions or scale by total model width | Modules use their own dimensions; each head scales by `sqrt(head_dim)` | [Lesson 9 — Learned Q/K/V and scaled attention](lessons/07-qkv.md), known-answer/packed-head checks |
| Mask shape is explained as requiring physical repeats | `[B,1,1,S]` key masks broadcast; causal and padding rules are explicitly intersected | [Lesson 11 — Padding, causal and cross-attention masks](lessons/09-masks.md), mask intersection and illegal-input perturbation |
| Fully blocked probability rows or active evaluation dropout can produce wrong outputs | Explicit empty-row convention; SDPA receives dropout zero in evaluation | [Lesson 10 — Multi-head attention implementation](lessons/08-heads.md), empty-row and path-parity checks |
| SDPA is described as though every call guarantees FlashAttention | Kernel dispatch is conditional; CPU correctness does not prove GPU performance | [Lesson 10 — Multi-head attention implementation](lessons/08-heads.md) and the profiling workbook |
| Unpositioned attention is called permutation invariant | Its position-aligned outputs are permutation **equivariant** under the stated unrestricted conditions | [Lesson 14 — Residual and normalization blocks](lessons/11-transformer-blocks.md), row-shuffle derivation |
| Independent corpus cleaning/truncation or a random holdout can obscure pair/test identity | Validate paired lines, filter pairs together, fit tokenizers on training data and name validation/test roles | [Lesson 6 — Corpus splits and task visibility](lessons/06-objectives.md), [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md) |
| Averaging unequal microbatch means or local rank means changes token weighting | Sum losses/correct counts and divide by the true supervised count for the declared group | [Lesson 18 — Translation optimization and complete training updates](lessons/13-translation-training.md), [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md), unequal-count update checks |
| Single-sample inference leaves batch/EOS/budget boundaries implicit | Preserve batch axis; finish rows independently; capacity and budget include BOS; keep source masks | [Lesson 19 — Autoregressive generation, completion and decoding scores](lessons/14-translation-generation.md), controlled batched completion/budget checks |
| RoBERTa source constructs mask lengths after padding, producing all-true padded masks | Derive visibility from original lengths/actual pad ID | [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md), unequal-row masking checks |
| Source default pad ID and actual tokenizer pad ID can differ | Read special IDs from the exact artifact and keep config/data/positions consistent | [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md), artifact identity checks |
| Random MLM replacements use a no-replacement sample over the vocabulary | Independent ordinary-token draws with replacement; preserve original selected labels; exclude special targets | [Lesson 24 — dynamic MLM corruption and selected-target supervision](lessons/18-dynamic-mlm-corruption.md), explicit corruption/label checks |
| QA source treats token offset ends inclusively and uses `(0,0)` for unavailable labels | Character offsets are half-open; unavailable-window labels are `-100` and excluded consistently | [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md), exact boundary and ignored-target checks |
| Source has no overflowing context windows and selects endpoints independently | Overlapping windows; select a context-only ordered span with a length limit | [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md), [Lesson 30 — constrained span decoding and QA metrics](lessons/23-span-decoding-and-metrics.md), window/constraint checks |
| Loading weights is described as enough to continue training | Restore optimizer/scheduler, RNG/generators, data progress and compatible identities | [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md), fresh-process continuation and failed-save fallback |

These are concrete implementation/contract choices. A check establishes the behavior under its fixture; it does not establish that every external training run, device or dataset follows the same conditions.

## Deliberate departures are different from fixes

| Topic | Source/paper distinction | This course's declared choice |
| --- | --- | --- |
| Translation FFN | Paper uses ReLU; inspected model uses GELU | Keep source GELU and teach its Gaussian-CDF meaning |
| Translation embeddings | Paper applies `sqrt(D)` scaling; inspected code omits it | Keep the inspected convention and record it |
| Translation schedule/loss | Paper uses Adam/inverse-root warmup and label smoothing; source uses AdamW/cosine and plain cross-entropy | Source-shaped mechanisms, reduced config and diagnostic learning rate; no exact-paper-reproduction claim |
| Fixed sinusoidal state | Source stores a non-gradient `Parameter` | Register a fixed buffer, keeping device/state behavior without a trainable parameter |
| Translation tokenizer/data | Source uses English BERT plus French WordPiece over a large corpus | Offline word lookup/six training pairs; optional actual WordPiece exercise; no large-corpus quality claim |
| Translation metric/decoding | Source greedy/`evaluate` BLEU differs from the paper's benchmark decoding; scoring APIs/settings vary | Explicit greedy contract, inspectable arithmetic BLEU and an optional SacreBLEU result with its full signature |
| RoBERTa positions | Inspected source uses `arange(0,T)`; Hugging Face handles padding-aware positions | Teach padding-aware cumulative positions with capacities matched to reserved IDs |
| RoBERTa vocabulary head | Inspected MLM decoder is separate/untied | Tie decoder weight to token embeddings and test shared parameter identity |
| RoBERTa artifact compatibility | Official tokenizer/weights and the tiny implementation use different mappings, names and some embedding details | Strict own-format backbone transfer; no claim that official weight files load directly |
| Corpus/scale | RoBERTa paper's large varied corpus and training recipe exceed the source's Books/Wikipedia and our fixtures | Original tiny documents, bounded CPU pretraining/QA and explicit quality limits |
| QA missing windows | Ignoring a positive answer outside a window is not modeling an unanswerable question | Ignore unavailable spans; no SQuAD2-style no-answer classifier claim |

The [Transformer paper](source:attention-paper), [BERT paper](source:bert-paper), [RoBERTa paper](source:roberta-paper), [SDPA](source:torch-sdpa), [Linear](source:torch-linear), [Adam/AdamW](source:torch-adam), [Tokenizers](source:tokenizers), [RoBERTa documentation](source:hf-roberta), [QA preprocessing documentation](source:hf-qa) and [SacreBLEU](source:sacrebleu) provide the primary contracts used for these comparisons. Source architecture, deliberate miniature simplification and implementation correction are recorded separately so they can be revisited as experiments.

## Caption provenance and proof limits

English automatic captions were retained for reading alongside chapter metadata and the pinned code. Technical spellings in automatic captions can be wrong; code/API checks decide names, dimensions and executable contracts. Timestamps above are source chapter anchors, not a claim that every sentence of a lesson is spoken in that interval. The lessons are authored explanations and worked checks, not transcriptions.

| Video ID | Retained English caption SHA-256 |
| --- | --- |
| `JXY5CmiK3LI` | `f1c79f8a20e2b05d06e16600ce1bbd60fd1cbad9a53cb1d5ff42f21e7351b235` |
| `UkBggwQsbF8` | `50a85ef3c3eda18738179b3479064b6a689c4aac68ba29276180732163fd359e` |
| `DOqg_QxInd8` | `218f9c68e7110403d75c42e3c67a1a19badde4bfcd9633fbacc535910914b824` |
| `Fzv2lSt0Ing` | `f7788cc2c3acf4a291a5b781266ec569948b3a452faf6b58badb96c02def0a7f` |
| `_FVwYFICYuE` | `13d4012402fd71124d11e4710f3e5ae1390b3951b9578197aa4dd5771581d38d` |

The first three retained sets were acquired on 2026-10-10; the two RoBERTa sets were acquired on 2026-10-12. These hashes identify those saved copies; later caption downloads can change. Raw caption/source caches stay ignored and are excluded from source ZIPs. No recorded caption or paper result implies learner completion.

The [reference-run record](project/reference-runs.md) keeps commands, environments and actual miniature results. [Translation](project/translation.md), [RoBERTa](project/roberta.md) and [readiness](project/readiness.md) keep the remaining implementation, quality, explanation and hosted gates visible. Full-scale WMT/RoBERTa/SQuAD reproduction, distributed performance and Colab deleted-runtime recovery require their own measured evidence.

## Topic improvement source refresh — 2026-10-13

The expanded outline was written before authoring. Thirty-two applied topics retain the original 24 chapter IDs/paths and add eight missing component/evaluation/transfer/diagnosis topics. Every topic has worked mathematics, source contracts, changed-condition practice and personal notes; the study notebook uses self-contained checked sources.

Primary refreshes include official PyTorch 2.14 CrossEntropyLoss, Embedding, LayerNorm and AdamW, the original BLEU paper, full BERT/RoBERTa papers and the maintained Hugging Face SQuAD metric implementation. The old squad-metric GitHub URL returned 404 during this refresh; its original provenance is retained, while current metric claims link to the inspected maintained source.

Source review retains the zero-selected MLM skip (no forced selection), tiny selection probability .35, ordinary-only random replacements, PAD-dependent learned positions, output gradients into a tied PAD row, weight-only transfer versus full-state resume, packed-position QA loss versus context-constrained decoding, and raw cross-window score limitations. None is silently reattributed to an official pretrained model.
