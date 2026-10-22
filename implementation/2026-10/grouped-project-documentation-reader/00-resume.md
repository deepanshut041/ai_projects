# Grouped project documentation reader

Updated: 2026-10-19.

Implemented collection navigation for Foundations (four projects), Transformer Labs (five projects) and Language Models (overview only). Project pages show separate Phases and Implementation folders. Implementation records are numbered 00_, 01_, etc., with actual filenames and titles inside; month folders have been removed from project learning records.

Five existing project records were moved with recoverable copies retained under the ignored workspace .ai/project-docs-layout/2026-10-19-numbered-records/. The canonical project-learning skill, planning coordinator/workflow and helper now create numbered learning records and reject --month for that domain. Existing phase overviews remain outside implementation. The foundation lab launcher resource was updated to its moved checkpoint. Model source and all three Git indexes remain unchanged.

The live reader remains http://127.0.0.1:4273/project_docs/. Local project notes are rendered into ignored website/.local-dist; default exports include project READMEs only. Native folder controls, breadcrumbs, code/math, Mermaid, downloads and the existing Bike video are supported. Builds prepare a separate output before swapping it into place; the dev server waits for pending rebuilds.

Verification: see 05-tracker.md and the JSON evidence in this package. All required work is complete: root and prefixed browser runs passed, the default prefixed static export is restored, and verification.json records the final checks. No commit or publication. Bike remains in Phase 02 with the hour/working-day baseline next; learner/hosted gates are unchanged.
