# Tutorial 1 integrated topic review — 2026-10-13

The outline preceded the topic rewrite. Main authored topics 1–14 and integrated the source/reference/notes records; two bounded authors completed translation 15–21 and RoBERTa/QA 22–32. An independent numerical reviewer inspected the actual source and ran hand/finite-difference/autograd/behavior checks, standalone lesson blocks and reconstructed study exports. Reader review passes on isolated root/prefix builds.

## Teaching acceptance

Each of the 32 lessons has a named applied topic, prerequisite/goal, mathematical or state derivation, worked intermediate result, actual source interface, falsifiable check/failure and specific personal-note prompt. All 64 native questions change a condition from the lesson's worked case and include a worked answer/rubric. No ordinary Python primer is introduced. The useful existing Q/K/V/head/mask calculations were expanded with rectangular/backward/cost/offset reasoning; the general foundation course is a prerequisite rather than retaught as setup stories.

The course covers complete encoder–decoder translation and bidirectional MLM→QA: separate embeddings/output rows, fixed/learned positions, attention projections/heads/masks, teacher forcing, FFN/normalization/stack ownership, counted optimization, generation/evaluation, complete state recovery, corruption, tied output gradients, compatible stage transfer, original character/window alignment and constrained metrics. Debugging/profiling and the GPT handoff preserve learner evidence gates.

## Deliberate choices and proof limits

Original paper, pinned inspected source and miniature source are distinguished. Source translation uses GELU, unscaled embeddings, no embedding dropout, hard-target CE and AdamW/cosine. Tiny RoBERTa uses own-format byte-BPE, selection .35 and ordinary-only random replacements; zero-selection skips the update. QA loss normalizes over all packed positions, while decoding restricts legal context spans. Weight transfer creates fresh task state; complete-stage resume restores existing state. PAD-position changes, tied PAD output gradients and raw cross-window calibration are taught as actual contracts.

All original model/data/trainer/math/checkpoint code is unchanged; the only original source edit packages study assets in the exporter and protects existing outputs. New applied probes remain self-contained and do not import a mutable sibling tutorial. Personal notebooks and prior exports are preserved by refusal to overwrite.

## Verified source/study evidence

107 checks pass (71 original + 29 independent applied + 7 study/export). All 39 explicitly runnable lesson blocks pass in independent source-copy processes. The 46-cell cleared study notebook covers all 32 topics; its 13 code cells execute sequentially from a verified source ZIP. The final user export has exactly the same hashed payload as that independently executed export. Three exact fresh-process recovery comparisons pass; hosted deletion/recovery remains unperformed.

The concurrent D053 migration moved learner folders and retired nested Git controls. Core source/notebooks match the baseline; path/ownership documentation and Tutorial 0's migration render changed externally. This task neither resets those changes nor recreates nested repositories. Original 24 Course 1 chapter IDs/paths and Small GPT are preserved. Canonical learning remains P00-M1 setup complete with linear() next.

Reader/layout proof passes at 112 records per base across all four widths; parent visually inspected the selected desktop/mobile diagrams, formulas and study controls. Recovery display math and unique scenario insertion were corrected; the supporting link update was repaired from the verified baseline, preserving all 20 original question/answer pairs. Final fingerprint is recorded in topic-final-verification.json. No commit, remote publication, paid execution or hosted capability is claimed.
