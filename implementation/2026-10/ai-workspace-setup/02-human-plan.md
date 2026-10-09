# AI workspace setup plan

Date: 2026-10-09. Owner: this local workspace.

## Requested outcome

Adapt the Quantum workspace's outer organization and CLI workflow to AI in `/mnt/projects/ai`. Include projects and research, omit master's study, and create durable documentation before detailed implementation.

## Sequence

1. Inspect the reference workspace's ownership, CLI, source/research tooling, documentation sites, and tests.
2. Establish independent Projects and Research umbrella repositories and their local submodules. Keep the workspace root as a lightweight launcher and ownership guide, matching the reference.
3. Implement a portable AI terminal home and headless commands for discovery, status, doctor checks, tasks, documentation services, logs, and reproducible project/research scaffolding.
4. Add a small, runnable set of AI examples with meaningful tests, documented scope, and explicit research questions. Record primary sources and avoid paid-provider dependencies.
5. Add browsable project and research documentation with source provenance, templates, and a reproducible research workflow.
6. Verify CLI commands, terminal navigation, service start/stop, project tests/demos, documentation builds and pages, portability, and repository boundaries.
7. Replace provisional README text with tested quick starts and persist the final verification report.

## Completion criteria

- Only Projects and Research appear in the AI home.
- CLI paths work from other current directories and after a relocated copy.
- Tasks, service management, logs, source records, and scaffolding are usable from documented commands.
- Independent implementations and research docs have explicit repository ownership.
- Seed implementations are runnable locally with no credentials or remote model calls.
- Tests address real numerical, workflow, process, and navigation failure cases.
- Documentation builds and local serving are verified; publication remains a separate state.
- An implementation/evidence record survives context loss and identifies remaining limitations.

## Working decisions

Use the Quantum implementation as evidence, not as an unreviewed recursive copy. Choose small local examples to exercise the complete workflow. Preserve project identity and research provenance. Prefer a reproducible, locked Python toolchain where the reference supports it.

## Follow-up outcome and sequence

Keep the sample repositories and the user-owned LLM roadmap. Match the Quantum root layout. Put SETUP in Projects and keep source/continuation records in this owning package. Remove transient root/browser output. Add the source-derived AI planning coordinator, shared workflow, helper and eight canonical templates. Expose the shared collection inside both umbrellas. Verify links, repository-context routing, plan preservation, locked CLI checks and site builds; then remove regenerated QA output.
