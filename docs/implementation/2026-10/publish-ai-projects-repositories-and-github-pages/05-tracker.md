# Publication tracker

Updated: 2026-10-31

| Task | State | Evidence |
| --- | --- | --- |
| Inspect ownership and preserve source/history | done | Original heads, diffs, verified Git bundles and tutorial inventories in ignored recovery storage |
| Review publication boundary and strengthen ignores | done | Eligible source/current history scans; Gitleaks findings individually reviewed; private project docs and runtime output excluded |
| Fold Docs into ai_projects | done | 21eb3c45ec21879cc0329398036018d40bcd81f5; all 1,608 source blobs identical; both histories retained |
| Prepare parent Pages workflow | done | .github/workflows/pages.yml; recursive collection checkout; docs/website public build; /ai_projects/ prefix |
| Validate source and browser behavior | in progress | 27 tests pass; full prefix suite passes; root suite running; 26,917 internal links have no missing targets |
| Create and push six public repositories | in progress | Chrome creates target repositories; first three collections pushed and exact heads verified |
| Publish parent public submodule URLs | in progress | Five HTTPS GitHub URLs replace local-only collection URLs; Docs no longer a gitlink |
| Verify Pages and anonymous recursive clone | pending | Await parent push and GitHub Pages configuration |

One existing authored tutorial source-map has a blank line at EOF; its bytes are preserved. Source checks otherwise pass whitespace validation with blank-at-EOF excluded. Publication does not advance learning/model/hosted-compute checkpoints.
