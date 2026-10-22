# Reader and numbered learning-record verification

Updated: 2026-10-19.

Implemented: the collection/project reader, separate Phases and Implementation views, numbered record folders and real filenames, local-only private notes, linked code/video, Mermaid, portable links, staged build output and local doc watching. Updated the reusable skill/helper and migrated five existing project records without changing model algorithms or historical evidence.

Checks completed:

- Website content/unit checks: 11 passed, including discovery, relative links, ignored/private separation, symlink exclusion, media allowlists and byte ranges.
- Shared helper and relevant CLI learning/planning checks: 35 passed. Both updated skills validate.
- Prefixed browser suite: 112 shared records and all 64 current project/collection pages; desktop/mobile, existing tutorials, note navigation, filenames, Mermaid, downloads, video ranges and private-export checks passed.
- Active project Markdown links: 218 checked, none missing (numbered-layout-link-check.json).
- Index preservation: ai_projects, Projects Docs and Research Docs match their pre-change staged indexes (preservation-check.json).
- Video selected-source identities match the current relocated files. The original production locator/hash remains in the ignored video's source-relocation-2026-10-19.json. Raw historical JSON and supporting backup files retain their contents.
- TUI smoke passed after updating the foundation lab checkpoint resource.

Earlier checks exposed a test attempting to click a collapsed folder and a concurrent build replacing the output being tested. The final tests operate on separate private output, expand native folder controls and run prefix changes sequentially. Live rebuild output is staged and pending dev builds are awaited.

Final checks passed: 127 CLI tests; both / and /project_docs/ browser suites (112 shared records and 64 collection/project documents); the final 11 website tests/build; and the live doc addition/removal check, with all ten sampled page requests returning 200. The default prefixed export is restored and the user service is running at port 4273. Verification summary: verification.json. The foundation lab implementation ignore rule covers its numbered personal records; its original staged delivery records remain untouched. See before.json, numbered-layout-before.json and numbered-layout-migration.json for ownership, preserved state and migration details. No commit, publication or learner/model gate advancement.
