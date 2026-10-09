# Pre-P01 local commit review — 2026-10-09

Request: make clean local commits across the existing repositories before starting Project 1. Lead: the shared planning coordinator with the AI project workflow. Resume the existing workspace setup package; preserve repository histories and all prepared work.

P01 GPT-1 / P01-M1 remains not-started. This review records workspace preparation, not model implementation, training, Colab recovery or learner mastery. Workspace home is intentionally outside Git; its existing launcher and README/AGENTS remain in place.

## Reviewed scope and commit order

Nine independent repositories were inspected. Three seeded Projects implementations were already clean and need no new commit. The remaining six own these pending changes:

| Repository | Reviewed changes |
| --- | --- |
| Research generalization-lab | Selected, previously measured synthetic OLS verification artifact. |
| Research Docs | Shared planning/tutorial/whiteboard skills, deferred research intake, synthetic pilot documentation/download, print checks. |
| Independent CLI | Roadmap/task/tutorial commands, owner/path routing, locked dependencies, relocation evidence and architecture sources/assets. |
| Projects Docs | Consolidated canonical roadmap/checkpoints/setup, P01 reference tutorial, Colab requirements, website reader and verification records. |
| Research umbrella | Canonical deferred intake/checkpoints, shared skill link and child revisions. |
| Projects umbrella | Documentation submodule move, removal of extracted CLI/duplicate launcher, owner guidance, ignore rules and child revision. |

Save child repositories first, then stage the umbrellas' exact child revisions. Use focused commit subjects and preserve existing commits. No remote push is part of this request. Final commit identities, working-tree cleanliness and parent/child alignment are checked after the documentation commit; Git history is authoritative for the resulting identities.

## Current verification

| Check | Result |
| --- | --- |
| CLI: `uv run --frozen pytest -q` | 64 passed after correcting the stale diagram-export assertion. |
| CLI TUI smoke and doctor | Passed; all six registered child entries present and required tools available. |
| Workspace launcher from unrelated `/tmp` cwd | Passed; resolves the current CLI and child paths. |
| Projects Docs: `npm test`, build, browser suite | 7 content tests passed; 14 records / 7 sources; browser routing, controls, downloads, print and mobile checks passed. |
| Research Docs: `npm test`, build, browser suite | 5 content tests passed; 4 records / 7 sources; browser checks passed. |
| Generalization Lab: frozen pytest | 4 passed. |
| Seven shared skill descriptors | All pass the installed skill validator. |
| P01 lesson validation | Passed; both C++20 and Python examples match their declared outputs. |
| Synthetic evidence copies | All three selected copies are byte-identical. |
| Reviewed candidate files | No credentials, datasets, weights, runtime environments or raw archives included; private-key and common credential patterns found no matches. |
| Desktop/mobile visual review | Current Projects tutorial and Research record/home screenshots inspected; layout is legible. |
| Diff whitespace | Reviewed changes pass `git diff --check`; staged changes are checked before each commit. |

The first CLI run was 63 passed / 1 failed: the test still expected six SVG diagrams after the tutorial added its seventh learning-path diagram. The assertion now compares exported SVG paths with the lesson's declared diagram IDs, preserving the passive-asset rejection check and avoiding another stale fixed count. The final CLI suite passes all 64 tests.

Selected authored reports, diagrams and verification artifacts are eligible for Git. New test screenshots, builds, environments and process state remain ignored. Earlier dated package statements about no commits describe their original sessions; this explicit user request authorizes the pre-P01 local commits. No external publication or paid compute is claimed.

Next project action: create/resume gpt-lab and complete P01-M1 source/config/tokenizer/data and Colab Free setup/storage/budget contracts.
