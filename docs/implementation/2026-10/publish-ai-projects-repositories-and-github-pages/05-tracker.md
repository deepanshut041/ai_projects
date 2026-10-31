# Publication tracker

Updated: 2026-10-31

| Task | State | Evidence |
| --- | --- | --- |
| Inspect ownership and preserve source/history | done | Verified Git bundles, original heads/diffs, original Docs metadata and tutorial inventories saved locally |
| Strengthen ignore rules and review source/history | done | 54 ignore probes pass; no forbidden current tracked files; all Gitleaks findings reviewed as checksums/tutorial wording |
| Fold Docs into ai_projects | done | Integration merge 21eb3c4 retains both histories and all 1,608 original source blobs |
| Build Pages from the parent repository | done | .github/workflows/pages.yml recursively checks out collections and builds docs/website at /ai_projects/ |
| Validate source and browser behavior | done | 27 tests, public root/prefix builds, all four browser suites at both bases, 26,917 internal link checks |
| Create and push six public repositories | done | Created using signed-in Chrome; GitHub public/default-main metadata and exact remote heads verified |
| Publish portable submodule URLs | done | Five public HTTPS submodule URLs; anonymous recursive clone initializes all five |
| Verify GitHub Pages and clean independent build | done | Workflow 41240646452 succeeds; fresh anonymous clone npm ci/test/build passes; 293 HTTPS checks cover all 286 record routes and core assets |
| Verify live Chrome lesson | done | MHA lesson navigation, zero math errors and all 26 image elements loaded |

One existing tutorial source-map has a blank line at EOF; its bytes were preserved. Local-only videos and private study records are deliberately excluded from the public build. Learner/model outcomes are unchanged. Selected receipt: verification.json; review: 06-review.md.
