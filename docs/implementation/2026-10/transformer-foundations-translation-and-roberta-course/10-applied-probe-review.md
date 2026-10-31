# Topic improvement: independent applied-source and numerical review

Recorded 2026-10-13. I added only `examples/model_learning_probes.py` and `examples/test_model_learning_probes.py` to the course source. These are arithmetic, gradient and information-flow probes which inspect the existing complete translation, MLM and QA modules. They do not create another model/trainer or import the previous course's probe source.

## Verified source and runtime

From `tutorials/projects/p00/transformer-course/v1/examples/`:

```bash
python -m unittest test_foundation_math translation_checks roberta_checks course_session_checks test_model_learning_probes test_course_study
python model_learning_probes.py
```

All **107 checks pass** after study integration: 71 existing checks,29 applied checks and7 study/export checks. The final combined command takes5.595 seconds under the declared local runtime. The actual verification interpreter is `/mnt/projects/ai/.ai/envs/attention-is-all-you-need/bin/python`, Python3.12.13/PyTorch2.14.1+cpu. New arithmetic uses CPU float64; the existing tests retain their declared source-specific precision. `topic-probe-result.json` records exact modules, result/stdout, source hashes and the compact deterministic report. All 20 original example assets matched `topic-improvement-baseline.json` before integration; only the parent-owned exporter is intentionally revised by the source/study workflow. Original model/data/trainer/math/checkpoint source remains preserved.

The independent command `python course_recovery_checks.py --project all` also passes: exact translation/MLM/QA fresh-process continuation, corrupt-generation fallback, partial-write rejection, incompatible/source/scheduler identity refusal and two retained verified generations. This checks local persistent continuation under the declared runtime; it does not establish deleted-hosted-runtime recovery.

The previously used child runtime path disappeared during this task because external workspace work moved the children beneath `ai_projects/projects/transformer-labs/`. I did not restore those folders or alter an environment. Every child file covered by the original baseline still has the same contents at its new location. Child `.git` controls are absent in the new layout, so Git from those folders resolves to the enclosing Projects repository. Its returned HEAD/status cannot verify either former independent child HEAD. `topic-probe-external-workspace-move.json` records this distinction and the baseline commits rather than treating the enclosing HEAD as a changed child commit.

## Independently worked model fixtures

### Translation: a complete valid-target objective

For two supervised rows with logits `[0,log3]`, targeting1 then0, plus one ignored row:

- Both rows have probabilities `[1/4,3/4]`.
- Summed NLL is `log(4/3)+log4=log(16/3)`; the mean divides by **2 valid targets**, not three positions.
- The mean logit gradients are `[1/8,−1/8]`, `[−3/8,3/8]`, then `[0,0]`.
- Unequal microbatch counts use a common total denominator; simply averaging their means produces different gradients.

The hand values agree with actual `translation_train.token_statistics`, explicit stable formulas, autograd and selected central differences. The actual complete Transformer also has matching SGD updates for a batch with3 and2 valid target positions split into unequal microbatches. Gradients reach its source embedding, target embedding and output projection.

### Translation: learned rectangular attention

The independent fixture uses two query feature rows and three source rows, width2/one head. Its learned Q/K scores are `[0,log3,log3]` and `[log2,0,log2]`, so the attention rows are `[1/7,3/7,3/7]` and `[2/5,1/5,2/5]`. Transported features become `[8/7,18/7]` and `[8/5,9/5]`; an independently specified output projection produces `[8/7,10/7]` and `[8/5,1/5]`.

The manual backward probe derives score softmax gradients, Q/K/V gradients, all four projection-matrix gradients and **both** K/V contributions to source-feature gradients. Those agree with autograd and the actual `translation_attention.Attention` module after copying matching transposed matrices and zero biases. A separate two-head rectangular case has masked/empty-row `gradcheck`, analytic-gradient checks and SDPA parity with explicit dropout0.

### RoBERTa: selection is different from replacement

Controlled input `[BOS,6,7,8,9,EOS,PAD]` selects ordinary IDs6/7/8 and assigns one mask/random/unchanged branch each. Labels remain `[-100,6,7,8,-100,-100,-100]`; the unchanged selected row still contributes loss. For100 eligible positions and selection probability.15, expected selected/mask/random/unchanged counts are15/12/1.5/1.5. These are expectations and conditional branch probabilities, not guaranteed finite-batch counts.

The real collator has **no forced-one selection rule**. Probability0 yields zero targets without changing IDs. The actual `training_step` then leaves model parameters, optimizer state, schedule and update count unchanged, while sampling/corruption RNG can still advance. The selected-only NLL probe checks that an unselected logit row receives zero direct loss gradient. It does not claim unselected input tokens receive zero contextual/parameter gradient.

### RoBERTa: tied weights receive joint gradients

The actual MLM decoder's weight is the same Parameter as the input table. With an output hidden row `[2,0]`, a summed vocabulary projection contributes `[2,0]` to every table row. Repeated lookup ID5 adds `[2,2]` to that row; a padding lookup adds nothing. Thus the PAD row still receives `[2,0]` through the tied output path. The optimizer sees the table once.

A width8/vocabulary20/one-layer/context16/ratio2 model has920 backbone parameters and108 unique MLM-head additions, total1028. The analogous QA head adds18 parameters. Exact counts agree with the actual modules. A real MLM update creates AdamW moments; backbone transfer copies equal tensors into a separate QA model, preserves its new head, rejects a mismatched config and begins with an empty task optimizer state.

### Extractive QA: endpoint gradients and legal spans

For probabilities `[.1,.2,.3,.4]` over four packed positions, start target2 and end target3 give mean endpoint loss `log(25/3)/2`. Its start gradient is `[.05,.1,−.35,.2]`; end gradient is `[.05,.1,.15,−.3]`. The actual model matches these normalization rules: a missing window has both endpoints ignored and zero direct logit gradient; valid-window CE normalizes across **all packed positions**, so question/special/padding logits compete too. Context-only legality is imposed later by span decoding, not by the source loss.

An independent candidate fixture has context start scores `[1,8,0]` and end scores `[9,2,7]`. Independent argmax would reverse start/end. Joint ordered search with at most two tokens chooses the second-through-third context tokens, score15. Higher question logits are excluded by `None` offsets. Original-character extraction returns `b c`.

Unmerged `é` has two byte tokens with the same original character interval; `café` maps five bytes onto four half-open character ranges. Endpoint alignment excludes the following punctuation when the answer ends at character4. Overlapping windows use original coordinates and label only an available span. A byte-token window can split a multibyte code point; this does not make every window's standalone UTF8 decoding valid. The original-text slice is the extraction authority.

## Additional model controls and limits

- The actual untrained translation model preserves earlier logits when future decoder inputs change, ignores source-tail tokens under a fixed padding mask, and changes outputs when real source content changes. These are architectural information-flow checks, not evidence of trained source use or held-out translation quality.
- Actual RoBERTa hidden states can use right context and ignore a masked padding tail. Its content features are bidirectional; selected-only output supervision does not mean a selected token attends only leftward.
- Actual encoder blocks with zero attention/FFN branches apply LayerNorm **after** each residual addition; the result is two post-normalizations, not the original input. The inspected FFN uses exact GELU, whose derivatives at0 and1 are independently checked; it is not the original Transformer paper's ReLU FFN.
- Attention dropout is reproducible after restoring RNG in training, uses the inverted expectation rule, and consumes no RNG in evaluation with dropout0. `eval()` controls dropout, while graph mode is separate.
- Local BLEU clips repeated n-grams, aggregates corpus counts, applies a brevity penalty and a geometric mean without smoothing. A short exact match can score0 when the requested four-gram precision has no denominator. The fixture is not interchangeable with SacreBLEU's13a/exponential-smoothing signature.
- QA normalized EM and multiset token F1 have hand examples with duplicate tokens. The source's normalization is English/article/ASCII-punctuation oriented; it is not a universal multilingual evaluation policy.
- Raw start+end logits across windows are not invariant to a separate additive constant per window, although each window's softmax probabilities are. The existing helper uses that simple raw-score comparison; do not claim calibrated window scores or a learned no-answer decision.
- `translation.evaluate` restores its previous mode. `fixed_mlm_evaluation`, `fill_mask` and `predict_qa` leave `eval()` active; the MLM trainer explicitly switches back to training. Unlike translation's update function, the inspected RoBERTa training function has no explicit finite-gradient rejection before stepping. No backend source is changed by this review.

## APIs for notes and runnable exercises

`token_loss_and_gradient` returns loss sum, valid count, mean and a manual last-axis logit gradient. `accumulated_logit_gradients` exposes the common-denominator rule. `projected_attention` returns output, attention rows, Q/K/V and merged context; `projected_attention_backward` returns each input/projection gradient. The probes use mathematical row-oriented weights and True=allowed masks. No attention dropout is hidden in them.

`translation_parameter_budget`, `roberta_parameter_budget` and `attention_score_elements` derive dimensions/resource counts. Explicit score-array counts are not measured fused-kernel memory. `corruption_expectations` and `controlled_mlm_probe` separate eligible/selected/replacement roles. `utf8_byte_character_map`, `qa_endpoint_loss_and_gradient` and `valid_span_candidates` expose the coordinate/loss/decoding contracts. `multiset_f1` and `bleu_from_counts` expose metric arithmetic after the specified normalization/counting protocol. `model_control_probe` checks actual model visibility with restored global RNG; `report` emits compact deterministic JSON.

Importing this probe module uses only the standard library; running Torch operations or importing actual models requires the frozen dependencies. No general programming primers or notebook-global implementation state are introduced.

## Primary references checked

- [BERT paper](https://arxiv.org/pdf/1810.04805): selected-token MLM replacement roles and QA endpoint formulation.
- [RoBERTa paper](https://arxiv.org/pdf/1907.11692): dynamic corruption means generating masks when an example is fed to the model; paper recipe/scale differs from this miniature.
- [Original BLEU paper](https://aclanthology.org/P02-1040.pdf): clipped counts, corpus aggregation, geometric weighting and brevity penalty.
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html): logits/class-index inputs, ignored targets and mean reductions.

All fixtures, probes and exercises in the new source are original. The original model/data/trainer/math/checkpoint modules remain preserved. `topic-snippet-checks.json` records **39 explicitly runnable Python blocks across all32 topic lessons**, each executed in a fresh process with a fresh disposable source copy and no preloaded names. All pass. Explanatory source excerpts are deliberately labeled text; online/Colab-only recipes are not locally executed.

## Final study and export verification

The independent `test_course_study.py` exercises every manifest topic in both the notes and cleared study notebook; unique cell IDs/compilation; exact refusal to overwrite an existing personal notebook, ZIP or neighboring Colab notebook; every archive member hash; source/study identity; and runnable fresh extraction. It also runs every optional study cell sequentially in a clean interpreter, then executes the extracted applied probe directly. These tests found the topic9 tuple/result-interface mistake before release; the parent fixed it by calling `projected_attention(*learned_attention_fixture())`. Raw insertion tags were removed from the notes, and the specific final-topic notes prompt was retained.

The retained verification artifact directory is `/mnt/projects/ai/.ai/course-checks/tutorial1-topic-rewrite/study-export/verified-danrwjge/`:

- `tutorial1-source.zip`:26 hashed payload files plus the manifest,27 members. SHA256 `f512e9896f6eb99897dd7dfb25398cce7212481f3e7974f7c3e612e9f9e3a249`.
- `extracted/study.ipynb`:46 cleared cells,13 code cells; all13 executed sequentially from the neighboring extracted sources. All32 topic headings are present.
- `extracted/study-notes.md`:the same32-topic blank workbook, hash-checked against the export and canonical notes.
- `tutorial1-source.colab.ipynb`:five cleared, compilable code cells. Upload/install/Drive/training cells are compiled only; no hosted execution or dependency installation was performed.

`topic-study-export-verification.json` records exact artifact paths, archive/study/notes/Colab hashes, cell counts and the stdout of every executed study cell. This ignored verification export is separate from the user's prior personal notebooks and any final delivery artifacts created by the parent.
