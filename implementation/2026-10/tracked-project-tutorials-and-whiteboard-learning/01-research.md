# Research — Tracked project tutorials and consolidated documentation

Updated: 2026-10-09
Scope: local documentation consolidation, reusable tutorial/whiteboard skills and authoring/reader integration. No model implementation, training, new Research experiment or remote publication.

## Source findings

Inspected Dashboard developer-learning-tutorials and feature-whiteboard-defense at the revision and hashes in source-provenance.json. Reused evidence/claim IDs, selected snapshots, chapters, semantic visuals, scenarios, practice/rubrics and whiteboard preparation. The source Dashboard requires ignored/private lessons; the user's direct instruction overrides that boundary for this AI workspace. Dashboard files remain unchanged.

Inspected canonical AI planning workflow and actual CLI registry, roadmap, package/scaffold routes, site rendering and tests. Prior state had a separate project_docs submodule plus umbrella docs records and several root Markdown documents. The user's correction establishes one consolidated ai_projects/docs repository. The former project_docs repository and its Git history moved to docs, alongside the existing documentation records. Hashes for 139 files were verified before link edits. Original index/config and the full hash manifest are recoverable under ignored .ai/relocation/docs-consolidation/; docs-consolidation.json records the result.

## Implementation decisions

- docs/ owns all shared Projects documentation: README entry point, roadmap/status/checkpoint/decisions, setup, catalogue, tutorials, implementation records and verification. Projects root keeps only controls, short README/AGENTS, launcher, skill link and projects/.
- Preserve the documentation submodule's history and existing CLI ID/port/URL. Update discovery to resolve project-docs at ai_projects/docs. Do not create a replacement repository or duplicate reader.
- Planning from either Projects owner or Docs owner selects docs/implementation, including standalone helper invocation inside Docs. Roadmap reads use docs canonical files. Model scaffolds link to those files.
- Tutorial authoring and rendering stay agent/CLI operated. Captured evidence uses evidence/ instead of the ignored sources/ archive. Sources and artifacts need freshness/hash checks, bounded package paths and passive SVGs.
- C++ and Python are the learner's programming choices. The existing site JavaScript only displays material/declared scenarios. It does not execute model code or author tutorials.
- P01 orientation is a reference-design study: six diagrams, two annotated tables, one four-case leakage exploration, four retrieval checks, C++/Python toy examples and whiteboard preparation. The paper and pinned original train.py were rechecked. Toy averages, shapes and cache arithmetic are educational derivations. All model milestones remain not-started.

## Verification and limits

CLI 54 tests pass, including path ownership, scaffold links, non-overwrite, snapshots/freshness, bounded lesson paths, coverage and passive artifacts. Docs 7 content tests and build pass; browser covers 14 records plus diagram loading, all scenario outcomes, answer/print behavior and mobile containment. Toy examples match independently declared results in both languages. Skill entry points pass quick_validate. Portable launchers/doctor/TUI pass. Browser output and runtime remain ignored; source material is Git-eligible. No claim of full GPT training, GPU/SSD performance, mastery or external publication.

## Follow-up findings

Both old roadmap copies have identical bytes. Sections 1–8 and 11–12 contain model plans, resource links, sizing, numerical protocols, optimization/adaptation exercises and dated source/release notes. Section 9 contains a proposed layout, report template and bounded experiment questions; section 10 contains completion gates. Merge these into existing docs/roadmap.md, docs/SETUP.md and docs/progress.md. Update the old monolithic layout proposal to the actual independent repositories/docs structure. This is content consolidation, not fresh verification of inherited model/release claims. Remove the Projects wrapper and direct commands to the shared workspace launcher. Historical evidence remains dated history.

## Skill readiness audit

2026-10-09: All seven canonical entries pass frontmatter validation and their local references resolve through both owning umbrella links. Tutorial/whiteboard/coordinator UI metadata matches skill names. Specialist skills are valid without optional UI metadata. Read actual helper and command paths; behavioral audit is underway. Keep this work in the existing documentation package, not new root files. No model/research training is implied.

Behavioral findings: displayed practice coverage was checked using all manifest questions rather than chapter-referenced questions; declared examples absent from chapter links were excluded from lesson fingerprints; rendering wrote earlier diagram outputs before a later insertion/file failure; copied core retained an unused alternate publisher without frozen exports. Fix those contracts and keep one renderer. Expose the helper’s existing level option through the CLI. Scope experiment/paper instructions so ordinary Project learning/source work remains in its owner. Preserve prior artifacts on failed rendering/snapshot provenance lookup.

## Skill readiness verified

2026-10-09: Seven local skills pass structure/reference checks through both owning umbrella links. Fixed practice coverage, declared-example fingerprinting, stale example proof, failed-render and snapshot preservation, duplicate-worded practice identity, duplicate renderer removal and CLI --level. Planning instructions now route ordinary Project paper/experiment work to Projects and avoid docs/docs. Shared task guidance carries C++/Python and the local tracked tutorial boundary.

CLI: 64 tests and TUI pass. Docs: 7 tests, build and browser for 13 records pass. Current C++/Python examples pass and the lesson was rendered again. An isolated disposable workflow exercised all four specialist routes plus the coordinator combination, paper/experiment record creation and source registration; tutorial init, non-overwrite, registered foreign-source capture, authoring, validation, both-language execution, rendering/indexing, stale-source rejection and actual browser checks pass. The disposable reader had 12 records because the historical GPT starting-plan record was not copied.

Usage is documented in SETUP.md. Root chats follow explicit canonical paths; both umbrella links resolve. Optional specialist UI metadata is not required. Whiteboard rehearsal remains one question at a time when requested; no learner mastery or model/research result was inferred from these checks. See skill-readiness.json and skill-workflow-exercise.json.

## Roadmap context in entry points

2026-10-09: User requests that agents and README expose the current project and prior work, not just point at a roadmap. Read canonical resume/progress/roadmap and current entry points. Verified current position: P01 GPT-1, P01-M1 not-started. Prior work is workspace/docs cleanup, roadmap/source consolidation, seven validated skills and the reference-only P01 lesson with checked C++/Python examples. No GPT repository or training exists. Embed this dated compact context in existing workspace/Projects entry points, keep detailed state in docs and require reconciliation/read-update at session boundaries. Reuse this package and add no root files.

Entry-point audit: all six README/AGENTS snapshots match the current P01-M1 checkpoint, local navigation links resolve, GPT repository remains absent and the Projects root remains clean. No model milestone changed.
