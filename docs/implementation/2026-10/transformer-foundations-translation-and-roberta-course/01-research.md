# Tutorial 1 topic improvement — 2026-10-13

The 32-topic outline and integrated author/source/study work are complete. Final isolated reader checks pass at both root and prefix, across all four widths. See [plan](07-topic-improvement-plan.md), [integrated review](12-integrated-topic-review.md) and the topic-prefixed evidence JSON. Earlier dated sections below retain the original 24-topic phase.

---

# Source findings and teaching choices

Reviewed 2026-10-12. Primary source is Priyam Mazumdar's five supplied videos plus MIT PyTorch-Adventures at 2f18a5470c4ef0a3d413dd7b18d07941f45779e9. Three translation topics and two RoBERTa/QA topics feed one course. Captions/metadata and raw source remain ignored; safe URLs, hashes and chapter alignment are retained in source-inspection.json and the course source-map.

Translation source lives under PyTorch for NLP/Seq2Seq for Neural Machine Translation. It uses separate English/French tokenizers, post-norm encoder/decoder, sinusoidal positions and GELU; optimizer/scheduling/embedding/FFN choices are distinguished from the paper. The source's small random held-out partition is not the official WMT test. Attention conventions preserve interface/projection names, allowed-key Boolean masks, rectangular cross shapes and explicit/efficient parity without promising FlashAttention.

RoBERTa source has inconsistent PAD defaults, masks created after padding, batchwide without-replacement random tokens, simple sequential positions and untied decoder. QA source needs half-open character boundaries, ignored partial spans, overflow windows and joint context-only bounded decoding. The course explains and checks these fixes. Its own byte-BPE/vocabulary/pretokenization/parameter names do not imply official checkpoint compatibility. MLM dynamic corruption and fixed held-out corruption have separate denominators/protocols.

Original maths/diagrams teach tensor contractions, stable NLL/gradients/Adam, token/offset coordinates and complete state ownership. Reference code is separate from project/helper/teaching prose. Authoring has not replaced the existing learner repositories or advanced mastery/hosted gates.

Review found missing translation source/scheduler identity and a runnable Colab session. Both are corrected and independently checked: source/protocol digests reject incompatible continuation; explicit new/resume sessions have fixed total schedules, persistent budgets, verified generations, failure stop and an output-cleared notebook using the common trainer. Local proof remains separate from real Drive/deleted-runtime execution.

[Reference verification](reference-verification.json) reports 71 unit checks, three fresh-process comparisons and bounded training. Translation exact results are six training plus two held-out combinations; QA exact results are eight seen answers. Neither is real benchmark reproduction. Review/build/browser/preservation evidence is recorded in 06-review.md.
