# Complete topic rewrite review — 2026-10-12

## Result

Tutorial 0 is fully rewritten as **Deep learning foundations for Attention Is All You Need**. The saved 25-topic outline preceded authoring. Each topic has essential explanation, defined notation/axes, derivations and worked numbers, runnable/precise checks, failure cases, native transfer practice and notes. Advanced Python is assumed; general programming primers are omitted.

Mathematics and training foundations lead into text encoding, embeddings, positions, attention/masks/heads, position-wise FFN, residuals/LayerNorm, regularization, encoder–decoder learning, evaluation/recovery and paper assembly. Original post-norm/ReLU/Adam/sinusoidal choices are explicit; pre-norm/GELU/AdamW are comparisons. The final paper lesson maps equations 1–3, traces a full translation batch, counts 2138 parameters/592 explicit-score bytes in a declared miniature design, explains complexity/path length, BLEU, beam/length penalty and inference checkpoint averaging.

The user explicitly requested multiple agents after the outline/initial drafts. Three bounded agents supplied final lessons 21–25, new numerical probes and independent coverage/snippet review; main authored 1–20 and integrated/verified the complete reader and exports. Their reviews are9–11 in this package. No remaining substantive coverage/formula/integration issue was identified for the original-paper prerequisite scope.

## Corrections made during review

- The actual classifier uses linear warmup then constant rate, distinguished from the paper's inverse-square-root decay.
- Causal masking receives a target-shift preview before its full seq2seq lesson, eliminating a forward prerequisite gap.
- The reusable attention CLI fixture differs deliberately from Lesson 15's worked input; both are named and independently checked.
- Native transfer questions change distributions/logits/targets/positions/values/optimizer history/norm inputs, with independently checked answers.
- Notes generation extracts actual lesson-specific prompts and has no eleven-section assumption; final notes grammar and all 25 sections are reviewed.
- The reader's old prediction-story intro and hardcoded eleven-lesson assertion are replaced by topic guidance and metadata-derived order checks.
- Direct lesson links preserve practice/legacy-route/evidence assets under the existing nonrecursive publisher; no publisher-policy expansion is needed.
- Final LayerNorm parameter gradients use readable display maths; repaired display delimiters build without KaTeX errors and pass final browser checks.

## Local evidence

| Verification | Result and limit |
|---|---|
| Existing numerical/text/tensor/network/trainer/recovery source suite |80 passing checks; full shared trainer and new-process continuation remain controlled local references |
| New foundation probes |34 passing CPUFP64 checks: embedding/tied PAD paths, positions, attention analytic/finite-difference/autograd/mask invariance, projected heads forward/backward parity, FFN/norm/residual/dropout/smoothing/schedule/counts |
| Standalone lesson code |26 blocks executed independently in fresh processes; changed practice values checked separately |
| Publisher |6 declared Python/C++ examples, manifest validation and final frozen render pass;48 sources/8 claims/25 lessons/25 native practices/15 diagrams/2 explorations |
| Notes/source export |22 verified payload files plus manifest;25 topic sections;35-cell study notebook with9 local executed code cells; both notebooks syntax-valid and output-cleared; personal overwrite rejected |
| Reader |9 content tests; final separate root and `/project_docs/` builds and102-record browser suites at320/390/768/1440 widths pass, including all courses, links, navigation, practices/scenarios, downloads and print |
| Visual review |Mobile embedding/attention matrices and weighted outputs, desktop final LayerNorm gradients are readable; final console has no errors |
| Preservation |All 133 later-course bytes unchanged, both child HEAD/status/source bytes unchanged, all 11 old lesson filenames/IDs retained; archive hash verified |

The final render fingerprint is in `topic-reader-verification.json`. New source/student notes are in the ignored `.ai/course-exports/2026-10-12/topic-foundations-final/`; prior delivered notebooks are preserved. The final ZIP hash is `b3f5ac884815e25f79d567a70d23c0140e4c4b994e497058de3f5ec1e65cf0ef`. The runnable study copy has the imported source modules beside it.

## State and delivery

Canonical guides/D052 and all six compact README/AGENTS snapshots are refreshed. The course-reference TUI row stays hidden, and explicit legacy CLI/course identities remain available. Only the earlier export/study generators change among existing example sources; new educational probes are separate. Source and model implementation files remain Python, with the optional cleared notes notebook as a study companion.

Actual learner P00-M1 is still setup complete and linear() next in the existing child. Its complete contract is now Lesson 1; the old Lesson 3 URL contains a continuation link. Author verification does not complete learner explanation, translation/RoBERTa/GPT implementation/training or deleted hosted-runtime recovery. No commits, paid service, publication or deployment occurred.
