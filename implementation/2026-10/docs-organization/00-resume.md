# Docs organization

Updated: 2026-10-09
Status: complete. Lead: shared planning / AI project workflow. Local organization and compatibility work are authorized; tutorials are protected. P00-M1 remains not-started.

## Request and boundaries

Optimize the Projects docs root completely: combine redundant guidance, group website implementation, remove needless folders, and ignore disposable generated files. Preserve every byte and path in tutorials/, existing edits, independent repositories and history. The user chose two root compatibility links for the untouched tutorial README's roadmap and kickoff references.

## Audit

- The docs root has 25 visible entries: ten Markdown files, three website configuration files and twelve directories. Dependencies and compiled output are already ignored, but mixed into the navigation root.
- docs/ is an independent Git submodule, so its own short AGENTS.md remains necessary for repository-specific rendering and ownership rules.
- START_HERE.md, SETUP.md and PROJECT_OVERVIEW.md overlap and can form one getting-started guide. Roadmap, live resume, historical progress, durable decisions and full Colab recovery contracts have distinct purposes.
- Website code, content, catalog, templates, assets, tests, dependency lock and local outputs can move together into website/. Tutorials remain a sibling read-only input.
- Selected verification records are authored evidence, not disposable screenshots. Move them under implementation/evidence/ and preserve their files.
- CLI discovery currently assumes package.json in the repository root; roadmap reading assumes root Markdown. Adapt those interfaces without changing repository identity or executing authored text.
- Existing docs and umbrella edits are present. Baseline hashes, file inventory and statuses are saved outside the repository in /tmp/ai-docs-organization-before.json.

## Plan and tasks

| Task | State | Acceptance |
| --- | --- | --- |
| T01 Audit instructions, source consumers, Git state and protected tutorials | done | Current ownership and baseline recorded |
| T02 Consolidate guidance, group website source, relocate selected evidence | done | Small purposeful docs root; no lost source/evidence |
| T03 Update site/CLI path consumers, instructions and links | done | Portable commands and stable reader routes |
| T04 Verify content/build/browser, CLI and exact tutorial preservation | done | Native checks pass at root and prefixed hosting |
| T05 Record final state and review | done | Results, limitations and continuation saved here |

## Delivered layout

```text
docs/
  README.md, AGENTS.md
  roadmap.md -> guide/roadmap.md
  START_HERE.md -> guide/getting-started.md
  guide/                  # Six canonical guides
  tutorials/              # All 62 files unchanged
  implementation/         # Existing work packages and evidence/
  website/                # Source, npm lock/package, tests, ignored outputs
```

The docs root now has eight visible entries instead of 25, with only two authored regular files. Three guidance files are combined into getting-started.md. Resume and progress remain separate because live continuation and historical evidence serve different purposes. The local AGENTS file is shortened and retained because this is an independent submodule. Existing implementation records and selected evidence are preserved as dated history; generated screenshots/builds/dependencies remain ignored. No redundant current guidance copy is introduced.

## Verification and review

- Content suite: 8 passed. CLI suite: 68 passed, including nested-package execution/containment, grouped checkpoint reads, and current/legacy scaffold links.
- Builds and browser checks: 40 records, 13 sources at both / and /project_docs/. All record links/downloads, search/filter controls, chapter/support/complete lesson navigation, interactive tutorial controls, print and mobile checks pass.
- Desktop workbench and mobile tutorial chapter screenshots were visually inspected. Browser output is disposable: 45 temporary files (28,882,260 bytes) were removed after verification.
- All 62 tutorial files retain their exact paths and SHA-256 hashes. All 168 original tracked files have preserved destinations or explicit merged content. Non-tutorial local Markdown links resolve; tutorial links are validated by their unchanged manifests and reader tests.
- Generated dependencies/builds/browser outputs/caches are ignored; guides, lockfile, tutorial sources and selected historical screenshots are not ignored.
- A relocated source copy passes the eight content tests and builds all 40 records at /. Relative compatibility links resolve after relocation. Its temporary directory is removed.
- TUI smoke and all four canonical sections work through the absolute launcher from an unrelated cwd. The managed Projects site is restored through website/ at its usual URL.
- Existing local changes and Git history are preserved. Detailed results are in [verification.json](verification.json). No model work, hosted run, commit, push or publication occurred.

Next learning action remains independent transformer-foundations-lab kickoff and P00-M1 tensor/gradient/runtime/source/Free budget contracts. No cleanup task remains.
