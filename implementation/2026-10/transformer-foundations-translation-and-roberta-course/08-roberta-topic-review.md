# Tutorial 1 topic rewrite — RoBERTa, transfer and QA

Date: 2026-10-13. Author scope: displayed Lessons 22–32, in the saved outline's order; only their assigned Markdown, this review and roberta-topic-practice.json are edited by this agent. Model/trainer/tokenizer/checkpoint/session/export source stays unchanged. Parent owns metadata, notes/export integration, complete-course preservation and reader checks.

## Source inspection checkpoint

Read all of roberta_model.py, roberta_data.py, roberta_train.py, roberta_checks.py, course_checkpoint.py and course_session.py; read course_source_export.py and existing lesson/source-map coverage. Important distinctions to retain:

- corrupt_mlm permits zero targets and does not force one selection. training_step returns updated=False without optimizer/scheduler advancement in that case. Tiny masking rate is .35 rather than the common .15; random replacements select ordinary IDs only.
- The tiny tokenizer is trained on TRAIN_TEXTS plus QA training questions. It excludes VALIDATION_TEXTS but is an authored combined training fixture, not a strict independent pretraining-only vocabulary experiment.
- Padding-aware learned positions derive from token IDs, not the supplied key mask. Changed nonpad IDs in interior masked positions can change later real positions, so padding-invariance tests must preserve the actual position numbering.
- The tied MLM output path can update the PAD row despite padding_idx suppressing lookup gradients. Shared parameter identity is tested, not inferred from matching values.
- Backbone artifact transfer checks format/config/tokenizer and strict tensors. Direct model load_mlm_backbone only checks architecture and copies weights; it cannot verify tokenizer identity by itself.
- QA training loss normalizes endpoints across the full sequence. Context-only constraints are applied by best_span at inference, not by the current loss. Ignored windows are not trained as no-answer examples.
- predict_qa compares raw start+end logit scores across windows; fixed-window ranking is unchanged by softmax constants, but cross-window calibration is not guaranteed.
- Complete-stage resume restores full state and named mask/batch generators; it rejects a fresh backbone on resume. Weight-only transfer starts QA with a new optimizer/schedule/counters.

The initial inspection above preceded sequential drafting. The completed authoring and verification are recorded below.

## Completed sequential authoring

All displayed topics 22–32 were written in outline order, with advanced Python assumed. Nine old filenames/URLs remain compatibility identities; two new topic files fill explicit transfer and diagnosis gaps:

| Topic | Authored source | Main added depth and worked check |
| --- | --- | --- |
| 22 | lessons/16-bert-to-roberta.md | Contextual versus vocabulary vectors; selected conditional loss versus causal likelihood; BERT/RoBERTa recipe versus miniature/source choices; two-target NLL and actual MLM/QA shape/right-context probe. |
| 23 | lessons/17-roberta-tokens-and-positions.md | Ordered own-format byte merges; Unicode duplicate offsets; special/pair layout; cumulative positions and exact capacity; interior-pad perturbation confound; artifact round trip and separately labeled official-tokenizer profile. |
| 24 | lessons/18-dynamic-mlm-corruption.md | Eligibility/selection/visibility separation; conditional 80/10/10 expected counts; random-original coincidence; selected-logit versus contextual gradients; unforced zero-target probability/state; controlled branch and skip-update probe. |
| 25 | lessons/19-roberta-encoder-and-mlm-head.md | Full registered post-norm/GELU encoder; head transform and tied gradient paths; PAD output-gradient caveat; independent 1,326-parameter count and complete finite backward. |
| 26 | lessons/20-mlm-training-and-counting.md | Actual shared training-corpus and with-replacement sampling; optional packing/tail; selected-weighted accumulation; precise AdamW groups/indexed warmup-linear schedule; fixed evaluation/mode behavior and single-token candidate limits. |
| 27 | lessons/27-backbone-transfer.md | Checked own-format backbone schema/config/tokenizer; direct-model loader's narrower scope; discarded MLM head/new QA head; fresh optimizer versus exact resume; tensor equality and tokenizer-rejection probe. |
| 28 | lessons/21-qa-offsets-and-windows.md | Original half-open coordinates versus UTF8 bytes; first/last repeated-offset mapping; pair/context budget and derived overlap condition; ignored windows and positive-only scope; accented/later-window fixtures. |
| 29 | lessons/22-qa-head-and-fine-tuning.md | Two-column head, worked reversed endpoint example; both CE gradients/true window denominator; all-packed loss domain versus context-only decode; supervised/ignored row and backbone gradient probe. |
| 30 | lessons/23-span-decoding-and-metrics.md | Joint constrained pair derivation/enumeration; original-character extraction; record ownership and raw-window shift/calibration limitation; multiset EM/F1 and no-answer scope; independent arithmetic probe. |
| 31 | lessons/31-debugging-and-profiling.md | Task-specific falsifiable probes; mode/graph/trainability; actual finite-guard differences; overfit controls; parameter/activation/runtime budget distinctions and isolated invariant/gradient/profile probe. |
| 32 | lessons/24-ready-for-gpt.md | Three complete task traces and classification axes; equal scalar losses with different denominators; all-source interface probe; preserved full reference commands/assets/attribution; concrete learner/hosted evidence and decoder-only handoff. |

The lesson pages contain essential derivations and original intermediate numbers, not repeated full source listings. Complete unchanged source/API coverage remains linked beside the explanation and in the final source inventory. Original applicable diagram/scenario IDs remain placed; accepted new diagrams are `backbone-transfer` and `model-diagnosis`. The parent removed the redundant old `gpt-handoff` marker because its reasoning is covered by the two final-topic questions.

`roberta-topic-practice.json` supplies exactly 22 unique objects over 11 chapter IDs, with two changed-condition question/answer/rubric items per topic. Primary legacy IDs remain where applicable. New topic primary IDs are `backbone-transfer-transfer` and `diagnosis-transfer`. Questions include architecture/artifact compatibility, changed counts/masks/coordinates, optimization state and proof limits. Practice prose was spaced/read for clarity before the final handoff.

## Verification and preservation

The independent numerical agent executed all 39 Python-fenced blocks across the complete 32-topic course in fresh CPU subprocesses, using fresh ignored copies of source examples with no preloaded names; every block passed. It inspected these 11 topics' formulas/counts/current-source caveats and found no numerical/source correction needed. Exact block digests/stdout are saved in `topic-snippet-checks.json`. Its applied 29 source-contract checks and the original 71 checks also pass; the parent owns their final combined evidence.

Important verified results in this group include the 1326-parameter miniature tied MLM model, 39957-parameter default MLM model, exact compatible backbone equality/fresh QA optimization state, controlled original-label preservation, zero-target no-update with advanced corruption RNG, accented half-open extraction, QA endpoint gradients, legal joint span selection/multiset F1, raw-score shift counterexample and clean blocked trailing-key invariance with finite backward. Profile timing is explicitly a local observation without a fixed performance threshold.

All package-relative local links in these 11 lessons resolve. Display-math fences pair correctly; no unintended control characters/tabs are present. All 22 native practice IDs are unique and placed exactly once. Scoped Git whitespace checks pass. These checks are source/author behavior, not learner explanation or hosted recovery.

Compared against `topic-improvement-baseline.json`, the SHA-256 bytes of roberta_data.py, roberta_model.py, roberta_train.py, roberta_checks.py, course_checkpoint.py and course_session.py remain identical. This agent edited no source model/trainer/tokenizer/checkpoint/session/export, manifest, notes workbook, website service, Tutorial 0/Small GPT or implementation child. The parent owns notebook/source-export improvements and final whole-workspace preservation checks.

## Primary source inspection and access facts

Inspected on 2026-10-13:

- https://arxiv.org/html/1810.04805v2 — BERT bidirectional encoder/input/pretraining/fine-tuning contracts, MLM/NSP and corruption branches.
- https://arxiv.org/html/1907.11692v1 — RoBERTa recipe, dynamic corruption, full sequence construction, removed NSP, byte encoding and empirical corpus/batch/budget distinctions.
- https://huggingface.co/docs/transformers/model_doc/roberta — current official API/tokenizer/configuration context; moving configuration defaults must not substitute for an actual pretrained artifact's stored config.
- https://huggingface.co/docs/transformers/tasks/question_answering — context offsets, overflowing features, question/context ownership and task preprocessing.
- https://github.com/huggingface/transformers/blob/main/src/transformers/data/metrics/squad_metrics.py — maintained normalization, multiset overlap, maximum-over-gold EM/F1 and separate no-answer metric handling. Parent added source ID `hf-squad-metric`.
- https://huggingface.co/FacebookAI/roberta-base/resolve/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json and tokenizer.json at the same fixed revision — direct public HTTPS reads of JSON artifacts, with no model weights acquired. Confirmed vocabulary 50265, MASK 50264, PAD 1, position rows 514 and type-vocabulary 1; the tokenizer has BOS 0/EOS 2/UNK 3. These restore useful concrete official/course comparisons in topic 23.

Access limitations are explicit: the older `https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v1.1.py` returned 404 through browsing; raw/GitHubPages versions also failed. It is retained as historical provenance rather than claimed freshly read. The maintained Hugging Face source was read instead. Direct browser-tool access to fixed HF config URLs failed; public HTTPS JSON reads succeeded and inspected only source/tokenizer metadata. The optional `RobertaTokenizerFast.from_pretrained` workflow remains unexecuted and clearly labeled as a separate online profile; metadata reads are not that acquisition/training test.

Existing source-map/caption provenance and pinned instructional revision remain the retained evidence for the five videos' source coverage. The source-map corrections and current self-contained source, not a moving external main branch, define the miniature behavior. All prose/calculations/diagrams/practice wording in the rewrite are original; primary sources supply documented method/API contracts rather than copied lesson text.

## Four-pass review and final boundary

- Topic sequence: design → token/position identity → corruption → full encoder/head → pretraining/measurement → checked stage transfer → QA alignment → endpoint objective → constrained decoding/metrics → diagnosis/resources → complete handoff. Each topic identifies its inputs, operation and next unresolved contract.
- Maths: defined axes and counts precede consequential formulas; expectation/zero probability, GELU derivative, tied-table gradients, parameter counts, weighted accumulation, half-open/window geometry, endpoint derivatives, span scores/F1 and resource calculations include assumptions and changed conditions.
- Implementation: every Python-fenced block is complete from the named `examples/` working directory and imports actual source. Optional external acquisition is explicitly separated. No source-prose promise of forced MLM targets, context-masked QA CE, calibrated window scores, globally frozen PAD rows or implicit mode restoration survives.
- Readiness: own-format versus official compatibility, author versus learner checks, seen/fixed/held-out quality, transfer versus exact stage resume and local versus deleted-hosted recovery remain distinct. The current learner's `linear()` continuation is unchanged.

Assigned files are settled. Parent owns final 32-topic metadata/frozen rendering, study/export integration, complete reader/browser/visual review and source/protected-child preservation. No learner milestone, hosted run, commit or publication was performed by this authoring work.
