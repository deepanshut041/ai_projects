# Tasks

| ID | Task | Dependencies | Status | Evidence |
| --- | --- | --- | --- | --- |
| T01 | Inspect source skills and owner contracts | — | done | 01-research.md, source-provenance.json |
| T02 | Consolidate all shared docs and preserve history/files | T01 | done | docs-consolidation.json; 139 file hashes |
| T03 | Update CLI/path/scaffold/planning guidance | T02 | done | CLI tests, portable launcher checks |
| T04 | Adapt tutorial/whiteboard skills and authoring helpers | T01,T02 | done | skill validators and learning tests |
| T05 | Author P01 reference orientation and examples | T04 | done | tutorial lesson.md, example-checks.json |
| T06 | Integrate existing reader and controls | T04,T05 | done | Docs content/build/browser checks |
| T07 | Verify layout, ignore boundaries, checkpoint and service | T03,T06 | done | final-verification.json; site HTTP checks |
