# Translation topic authoring and review — 2026-10-13

Displayed lessons 15–21 are complete and frozen. They follow the saved 32-topic outline and apply Tutorial 0's foundations to the actual complete model rather than repeating a generic primer. The seven files contain substantial topic-specific explanations, defined dimensions/formulas, original worked calculations, complete runnable reference probes, failure checks, two revealable practice items each and specific personal notes prompts. The actual model/data/training/recovery sources remain unchanged.

## Coverage and implementation contracts

| Displayed lesson | Final coverage and preserved source detail |
|---|---|
| 15 Encoder stack | Full-source memory retains the source-position axis; exact post-norm/GELU route, independent layer ownership, padding isolation induction, nonzero padded-query caveat and mask reuse into retrieval. Two-layer invariance/object-identity probe passes. |
| 16 Decoder stack | Three target residual sublayers, custom Attention(src,tgt) argument roles, rectangular source retrieval, shifted causality through depth, shared-memory gradient sums and encoder-detach failure. Two-layer future isolation plus encoder-gradient probe passes. |
| 17 Complete model | All actual configuration defaults/validations and state names, separate unscaled embeddings, input-dropout departure, initialization .02, full embedding/memory/prefix/logit route, exact 30796 parameters/2048 fixed scalars, complete finite backward and representative nonzero paths. |
| 18 Training updates | Plain unweighted valid-token CE; all-ignored microbatch rejection; weighted accumulation; rank-based AdamW decay groups/betas/epsilon; global clipping; completed gradients retained until next clear; actual LambdaLR indexing and used-versus-next rate; bounded fixed-batch fit/distributed weighting boundary. |
| 19 Generation | Per-row BOS/EOS/PAD and max_len semantics; source encoding once, full-prefix recomputation, last-position restricted argmax; eval/no_grad and mode restoration; controlled independent completion/budget probes; worked greedy/beam/log/length scoring comparison; no implemented beam or decoder cache claim. |
| 20 Evaluation | Six-pair/two-recombination evidence scope, token totals/NLL/PPL/accuracy versus free generation, original clipped n-gram/BP calculations, fixed-order unsmoothed metric and optional signed SacreBLEU adapter, trained mismatched-source intervention, EOS/error/resource/data-presentation interpretation. |
| 21 Recovery/Colab | Full optimizer/schedule/RNG/source/tokenizer/runtime state, fixed translation batch versus later named generators, caller-owned completed-update boundary, marker/hash generation acceptance versus physical durability, exact new-object/fresh-process replay, corruption/identity/retention/budget/session counters, stage transfer and deleted-runtime hosted drill. |

The original Transformer, pinned upstream implementation and miniature reference remain distinct. The latter preserves the inspected post-norm/GELU, separate unscaled embedding and AdamW/cosine/plain-CE choices. It does not silently become the original post-norm/ReLU/Adam/inverse-root/smoothed paper design. The inspected embedding route also lacks an embedding-plus-position dropout module. Its causal/FFN/mask APIs and original helpers are preserved.

## Executed and independent checks

Eight standalone Python blocks across these seven lessons pass. The numerical agent executed them with fresh processes/source copies and recorded final block/lesson/source hashes in `topic-snippet-checks.json`. The author additionally ran the encoder, decoder, complete model and recovery probes. Recovery prints `same next update: 4 fallback: 3`; the stronger existing recovery suite separately checks new-interpreter continuation for all three tasks.

The Lesson 20 actual seed-7 CPU source-control run measured correct-source token NLL `0.00013815597631037236`, accuracy 1 and 30 targets, versus NLL `3.655841318766276` and accuracy `0.5666666666666667` after rolling source IDs/masks by one row while retaining target prefixes/labels. This is controlled fixture evidence, not benchmark translation quality. Its metric probe yields two-order `70.71067811865474`, four-order 0 and short-prefix `71.65313105737893` under the stated settings.

`translation-topic-practice.json` contains fourteen question/answer/rubric records, two per owned chapter. The numerical agent independently confirms changed mask mixture `[14/3,2]`, per-encoder count 872/add 1744, cross-score gradients[-.375,.375], vocabulary growth 195, fixed-buffer growth 1024 scalars/4096 bytes, uneven objective 1.25 versus 1.5, warmup used/next rates, greedy .36 versus beam .396, BLEU 54.772256/51.341712 and rollback 15 counters 90/900/450. Local package links and all fourteen practice markers were checked.

The source helpers `translation_model.py`, `translation_attention.py`, `translation_data.py`, `translation_train.py`, `translation_metrics.py`, `course_checkpoint.py`, `course_session.py`, `course_recovery_checks.py` and `course_session_checks.py` match the pre-improvement baseline byte hashes. The ignored pinned upstream model/train cache also matches the recorded revision `2f18a5470c4ef0a3d413dd7b18d07941f45779e9`. Captions/cache provenance remain preserved; no transcript wording is copied.

## Integration handoff

Retain existing chapter IDs `translation-model`, `translation-training`, `translation-generation`, `checkpoints-and-colab`, including their stable old filenames. Add `encoder-stack`, `decoder-stack`, `translation-evaluation`. Practice records include their owning chapter IDs for objective mapping. Existing graph markers `encoder-decoder`, `token-weighted-update`, `translation-generation` and `recovery-generations` remain linked.

Two new diagram proposals are `encoder-memory` (independent encoder layers, persistent source-valid key mask and memory retaining S) and `decoder-retrieval` (target residual stream through causal self-attention, rectangular cross retrieval and FFN, with memory K/V and target Q). Main owns diagram assets/metadata, notes/export, publisher/reader QA and canonical preparation snapshots. The prose-spacing pass briefly altered the source ID `wmt14`; main restored that identity before final integration. Further lesson polishing must preserve URLs, identifiers and code.

## Primary sources inspected

The original/pinned source contracts are supported by the already preserved paper/source catalog and re-read local code. Additional primary checks: [PyTorch 2.14 EncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html) and [DecoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerDecoderLayer.html) establish why defaults are not interchangeable with these custom layers; [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) establishes decoupled decay semantics. Main should register `torch-adamw` for its lesson link.

[Papineni et al., BLEU](https://aclanthology.org/P02-1040/), including its primary PDF, was inspected for clipping/corpus aggregation/brevity; the lesson's words, examples and calculations are original. Main should register `bleu-paper`. [SacreBLEU source/documentation](https://github.com/mjpost/sacrebleu) establishes signature/reporting context; the local adapter specifies its exact calls separately.

Author checks and written material do not advance learner implementation, held-out benchmark or deleted hosted-runtime recovery gates. No child/model implementation, services, commit or publication was changed by this authoring task.
