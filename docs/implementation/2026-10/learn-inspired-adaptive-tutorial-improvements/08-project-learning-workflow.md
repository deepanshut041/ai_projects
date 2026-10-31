# Project learning workflow adaptation — 2026-10-15

Current scope: adapt canonical workspace skills/helper/templates and Bike Demand Planner private records for phase-by-phase implementation with understanding as the main outcome. No model implementation/training, commit or publication is included.

Observed issue: the shared planning helper only provides delivery packages, including human plan, detailed implementation plan and task-ID tables. The learning/tutorial skill routes to complete published courses but has no local incremental project-study mode. Bike learning is split across docs plus a sibling phases folder, and its setup work record still reads as delivery management.

Chosen change: add project-learning with six focused files (resume, learning points, notes to understanding, tasks, comparisons and tracker), integrate a learning domain and phase creation with the existing shared helper, route existing project/tutorial skills to it for local study, preserve separate delivery/research behavior. Move Bike phases to docs/phases, expand only the current phase, refactor existing setup notes/tracker and preserve source/initial commit. All project study/work material stays ignored; shared tutorial visibility is unchanged.

Verification to run: helper learning/delivery/phase creation and overwrite/path safety; shared CLI regressions; skill/link checks; private-file Git/package exclusion; source/hash/index preservation and whitespace. Recoverable pre-edit copies are in workspace .ai/learning-layout-migration/2026-10-15-bike-demand-planner/.

Progress: source/owner/route inspection complete; implementation underway. Learning phase 00 has not started and all model/training/learner gates remain pending.

Latest user correction: phases must always stay outside implementation. Use sibling `docs/phases/` and `docs/implementation/`; the helper rejects implementation roots for phase creation.

Final user clarification: phase folders are the long project arc only, with one overview README each. Detailed learning points, understanding notes, tasks, comparisons, resume and tracker belong in docs/implementation. Bike's current records were moved to docs/implementation/01_prediction-contract-and-one-update/; all ten docs/phases folders now contain only README.md. The new skill, original skill routing, helper and regression checks follow this boundary. The learner has already watched tutorials/read important PDFs; each update should consolidate that prior study instead of restarting prerequisites. No phase/model implementation has begun.

## Verified completion

- Full shared CLI suite plus new learning helper regressions: 111 passed, 0 failed. Coverage includes six learning records, overview-only phase creation, private paths outside implementation, duplicate phase/overwrite protection, traversal/symlink escape and standalone listing.
- Five modified/new skills validate. Thirty-five skill/template links and 87 private Bike-doc links resolve. Scoped whitespace checks pass.
- All ten Bike phase folders contain only README.md. Six detailed current working records are under docs/implementation/01_prediction-contract-and-one-update/. No active human-plan files remain in Bike docs.
- Docs is ignored/untracked and excluded from rebuilt source/wheel packages. The original 12 hashed public source/metadata files, full public Git status, signed source commit and unrelated staged index are preserved.
- Shared TUI smoke passes. No model/data/training/application/hosted work, learner assessment or new commit/publication was performed. Preparation is complete; next is the current Bike request contract and numeric update when the learner starts.

## Natural learning points and diagram guidance — 2026-10-15

During the first Bike Phase 00 session, the learner found the writing too compressed and requested fuller human-readable explanation, longer natural learning points and useful Mermaid diagrams. Updated canonical project-learning/SKILL.md, its 01-learning-points.md template and developer-learning-tutorials routing. Learning points should connect what/why/example/limits in complete paragraphs, not fragment lists or arbitrary token targets. Detailed notes retain the complete worked derivation/source trace; diagrams clarify information timing or state flow and must be explained in prose.

Bike's current learning points and understanding notes are rewritten accordingly, including a genuine single-observation code example. Three Mermaid diagrams render in a local browser; the two current learning/update diagrams are visually checked. The new template/skill validate, and no learner mathematical answer or mastery is fabricated. Earlier helper/CLI evidence remains separate from the new numerical checks.
