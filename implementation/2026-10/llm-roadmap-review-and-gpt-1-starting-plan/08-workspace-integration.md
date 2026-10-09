# Workspace integration — authorized implementation

Updated: 2026-10-09. Lead: AI project workflow through the planning coordinator.

## Request and scope

Create the canonical Projects roadmap and durable progress/resume/decision records, refer to them from workspace instructions, documentation and the CLI, and move the CLI outside ai_projects. Keep Projects (known-model implementation/training/learning) distinct from Research (narrow questions, focused reproduction and improvement). No model implementation/training or external publication is requested in this phase.

## Proposed artifacts and ownership

- ai_projects/roadmap.md: canonical learning plan with P01–P19 and completion gates.
- ai_projects/progress.md: authoritative status/evidence ledger for the roadmap, with finer milestones for P01.
- ai_projects/resume.md: current project, next step and context links.
- ai_projects/decisions.md: append-only durable ownership/learning decisions.
- ai_research/roadmap.md: scoped research intake/gates; no premature experiment schedule.
- cli/: independent top-level workspace tooling repository, preserving the extracted CLI history and its current dirty files.
- Root launcher, workspace configuration, owner instructions, relevant READMEs/planning helper and Projects site guide: corrected entry points and canonical links.
- ./ai roadmap [projects|research] [--json]: read-only discovery/content entry point, without executing document text.

Preserve both older detailed architecture catalogues and previous review evidence as reference/history. New work reads the canonical roadmap/tracker/checkpoint first. Focused Research documentation belongs in Research, including the provisional inference-track proposal from the earlier review.

## Execution order

1. Record intake/status and CLI content hashes; extract committed CLI path history without rewriting Projects history.
2. Relocate the current CLI directory intact, initialize independent Git history/index without checking out over local edits, and update launchers/runtime paths.
3. Author canonical plans/tracker/checkpoint/decisions; relocate the focused research proposal and leave a forwarding reference.
4. Update ownership and guidance, CLI roadmap/task discovery, and site rendering from canonical files without duplicate authored sources.
5. Run the CLI suite, launcher/TUI/doctor/roadmap checks, planning ownership checks, site content/build/browser checks, and migration/document preservation checks.
6. Save evidence and resume state. Leave Project 1 as ready/not-started; no model repository/training implied.

## Acceptance and recovery

CLI works from the workspace, Projects wrapper and unrelated cwd; independent history is retained; existing CLI dirty files are unchanged at the move; snapshots include the new CLI Git repository; the canonical roadmap is reachable through instructions, CLI and local documentation; tracked Projects learning states are separate from Research protocols/results. No resets/cleans/pushes/deployments. Preserve ignored environments and active manager-owned service records. Relocation can be reversed by moving the directory back and restoring the launcher paths if validation reveals an unrecoverable issue.

Status: implemented and locally verified. CLI suite: 46 passed; TUI/doctor and unrelated-cwd launchers passed; Projects content: 6 passed; build/browser: 12 records and 7 source references, all links/downloads, roadmap/resume navigation and desktop/mobile layouts passed. Canonical checks: 23 files, 86 local links and 19 matching IDs. Original catalogues and Projects HEAD unchanged; 22 CLI files matched immediately after move; source/history snapshot inclusion/exclusions passed. See the independent CLI owner's tracker. No model implementation/training or publication started.
