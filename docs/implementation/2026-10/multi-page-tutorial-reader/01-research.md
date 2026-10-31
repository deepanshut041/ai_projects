# Reader source comparison

Updated: 2026-10-09

The user requests multiple Markdown files presented as one tutorial, using Dashboard as the reference. Projects Docs owns the site; no model implementation or remote publication is requested. Lead: AI project workflow; workspace tutorial and whiteboard instructions preserve tracked source/content.

## Verified source

- Dashboard `TutorialChapters.tsx` selects one manifest chapter at a time, reflects its ID in the hash, supports ordered controls and previous/next navigation. Its tutorial page composes authored Markdown with each chapter's diagrams, tables, scenarios and practice. The manifest owns identity/order. Source revision and inspected file hashes are in source-provenance.json. Dashboard stays read-only.
- AI `scripts/tutorials.mjs` verifies render.json hashes, then reads only the combined lesson.md as one record. Existing separate chapter Markdown cannot be navigated as pages; linked workbook/whiteboard files open as raw assets.
- AI `readRecords` permits only category/file paths. Build exports each record statically. Existing renderer supports math, scrollable diagrams, declarative scenarios and practice/print controls. Existing site does not need a new framework or dependency.
- P00 has twelve declared chapters plus workbook, research and whiteboard documents; P01 has three chapters and research/whiteboard documents. Both have frozen complete lesson exports. Existing local P00 changes and test changes are preserved; hashes of every existing tutorial file are saved before edits.

## Implementation choices

Use the manifest's content_path and IDs to compile chapter records directly from frozen authoring files plus declared insertions. Give each chapter a stable static route in the same site, a shared contents menu and previous/next links. Match Dashboard's chapter behavior through native links appropriate to the existing static stack. Retain the original tutorial entry URL as an overview, full lesson as a dedicated reading/print route and unchanged Markdown download. Exported supporting Markdown gets an in-reader page.

Only overview records appear as tutorial cards; all pages remain searchable. Keep source/artifact verification and reject undeclared or stale chapter content. Link rewriting must preserve fragments, validated assets, source citations and root/nested deployment prefixes. Browser checks must cover all authored chapter controls, direct load/reload, history, supplementary links, print and mobile menu/overflow. Learning remains P00-M1 not-started.

## Implemented and first verification

Verified per-file composition now generates 26 tutorial records: two overviews, fifteen chapters, seven supporting documents and two complete lessons. Normal site has 40 records and 13 sources. Contents menus/current state/previous-next, direct URLs and browser history pass; supporting Markdown stays inside the reader. All eight content tests and the first prefixed browser suite pass, including existing full-lesson diagrams, scenario cases, practice, printing and JavaScript-disabled native navigation. No authored tutorial source was edited by this task. Root-hosting and final responsive inspection are next.

## Concurrent source update observed

Preservation audit found changed P00 lesson.json, validation.json, render.json, lesson.md and dependencies.svg compared with intake. This reader task has not written those files; the tutorial chapter source bytes are unchanged. Preserve these concurrent authoring updates and validate the current frozen snapshot; do not restore intake hashes. Later verification records exact changed paths separately.

## Root and responsive verification

The browser suite also passed after SITE_BASE=/ build. Desktop and mobile chapter/menu screenshots were inspected: contents remains accessible, active chapter is marked, narrow layouts collapse the menu and diagrams scroll within the article. Complete lesson print/PDF and all teaching controls retain earlier behavior. A subsequent build restored /project_docs/ for the running workspace site.

## Final local state

Chapter/support text finds its overview through the existing search. Print CSS hides the tutorial menu/pager, and native links work with JavaScript disabled. Eight content tests and the final prefixed browser suite pass. The previously running service stopped during the turn; ./ai start project-docs restored port 4273, and a direct attention-chapter HTTP 200 confirms the delivered local site. Canonical progress/resume/D027 and six entry snapshots refreshed without model gate advancement.
