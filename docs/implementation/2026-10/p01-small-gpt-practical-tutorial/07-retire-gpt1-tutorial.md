# Retire the superseded GPT-1 tutorial — 2026-10-10

User request: remove “GPT-1: build the mental model” now that the integrated Small GPT course exists. This supersedes the earlier preservation requirement for that tutorial only. P00-M1 remains not-started.

Scope: remove the old package and its catalog card, update live guide/course references to the new P01 route, and migrate site/CLI tests that use the old course as a fixture. Keep dated historical evidence and the roadmap's optional GPT-1 architecture study intact. Preserve existing working changes and a verified ignored recovery archive before removal.

Status: complete, locally verified.

Removed the 20-file package from active sources and its index entry. A byte-verified archive and SHA-256 manifest remain at workspace home in ignored `.ai/retired-tutorials/2026-10-10/`. Current guide links, examples and all six compact snapshots now identify the integrated P01 route. Removed stale “preserved tutorial” references from P00's final handoff and P01's introduction and re-rendered both courses; their code and teaching sequence were unchanged.

Verification:

- `./ai learn list` returns exactly P00 Transformer Foundations and P01 Small GPT.
- Both current courses validate and render successfully.
- Nine native Docs tests pass; both prefixed and independent root builds/browser suites pass all 49 records at widths 1440, 768, 390 and 320.
- The removed slug is absent from active guides, tutorial packages, site tests and CLI tests; generated search, old pages and old assets are absent after a clean rebuild.
- CLI fixture migration retains the validation, containment, passive-export and failed-render preservation checks: 77 frozen pytest tests pass, and TUI smoke reports only P00 and the two documentation services.
- Docs and CLI whitespace checks pass. No commit, remote publication, model code change or learner milestone advancement.

Next learning route: P00, followed by the integrated P01 Small GPT course. The original delivery's preserved-file and 57-record results remain dated evidence in the earlier review; they do not describe this later cleanup.
