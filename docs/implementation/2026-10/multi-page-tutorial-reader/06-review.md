# Multi-page tutorial reader review

Updated: 2026-10-09

Implemented locally in Projects Docs. Tutorial entry pages now lead to separate Markdown chapter pages in one reader. P00 has twelve chapters and P01 has three. A shared contents menu marks the current page, chapter position and previous/next links guide reading, and mobile uses a collapsible menu. Seven supporting Markdown documents (workbook, research, whiteboard notes/questions) open inside the same reader. Complete lessons remain available as dedicated reading/print views and unchanged downloads.

Dashboard's manifest-ordered, one-chapter-at-a-time reader supplied the behavior; the existing static site uses ordinary page links. No new framework, dependencies, services or model implementation. Only overview cards appear in the library; full chapter/support text participates in search. Existing tutorial overview URLs remain stable, with new nested chapter URLs based on stable manifest IDs.

## Verification

| Check | Result |
| --- | --- |
| npm test | 8 tests pass, including per-file composition, navigation, validated asset/source/Markdown links, fragments, prefixes and stale exports |
| npm run build | 40 records, 13 sources at /project_docs/; includes 26 tutorial pages |
| SITE_BASE=/ npm run build then npm run test:browser | Root hosting passes |
| npm run test:browser at /project_docs/ | All 40 records and downloads/links pass; fifteen chapter routes, history/reload, active menu, support pages, library grouping, search, mobile menu and JS-disabled native navigation covered |
| Existing teaching controls | All full-lesson diagrams decode, all scenario cases and practice reveals work, answers print and restore correctly |
| Print/PDF | Navigation hidden; complete lesson export/PDF works |
| Desktop/mobile screenshots | Inspected chapter tops, full chapter and expanded mobile contents; no body overflow, large diagrams scroll inside their region |
| Preservation | Every intake chapter, example, workbook and whiteboard source hash unchanged; Dashboard inspected files unchanged |
| Six entry snapshots and canonical state | Refreshed with reader completion; P00-M1 remains not-started |
| Live workspace site | Started through ./ai start project-docs after prior process stopped; direct chapter HTTP 200 at port 4273 |
| git diff --check | Pass |

Evidence: [verification.json](verification.json), [source-provenance.json](source-provenance.json), [research](01-research.md). After the requested Docs cleanup, [selected reader screenshots](../../evidence/2026-10-09-tutorial-reader/README.md) are retained; disposable test-results screenshots/PDF were removed. Existing local edits were retained. A concurrent P00 lesson/diagram render update was detected and preserved; verification records its changed files separately. Reader work did not rewrite authored tutorial content.

## Authoring and limits

Declare chapter order, stable IDs, content_path and typed insertions in lesson.json. Validate/render after changing sources, then build the site. The site reads each verified chapter plus its declared diagrams/tables/scenarios/practice. Supporting Markdown already included in render.json gets a reader page automatically. Package-relative links retain the existing lesson-directory contract. Complete lesson.md is an export, not the only tutorial page.

All required local site checks passed. No remote publication, commit or model training occurred. Learner understanding and Colab recovery remain unassessed. Next learning action remains the independent transformer-foundations-lab / P00-M1 kickoff.
