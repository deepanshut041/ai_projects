# Resume — Grouped ordinary Projects folders

Updated: 2026-10-13
State: requested layout/tooling work complete; docs-render validation limitation recorded separately.

All four projects are ordinary directories in ai_projects/projects/transformer-labs/. ai_projects/projects/language-models/ is prepared for later complete language-model builds. Project IDs, runnable files and learning state are retained. Projects creation defaults to language-models, with --collection transformer-labs available. No model gitlinks or nested .git remain. Documentation and Research repository ownership is unchanged.

Lead: ai-project-workflow using shared planning guidance. Recovery artifacts: workspace .ai/project-layout-migration/2026-10-13/; original child Git metadata is also retained in the umbrella metadata. No commit or remote publication.

- CLI suite: 90 passed; grouped creation/discovery, task-owner routing and existing TUI behavior covered.
- TUI smoke: UI ok; four Projects entries; foundations selected; planned encoder projects have no execution actions.
- Attention/matrix tests from the moved local environment: 35 passed.
- Saved prediction: 6/1/5, locally and through the workspace launcher.
- Managed attention demo: PASS, exit 0 at the new path.
- Local project Markdown links: all 51 resolve.
- Recovery: two all-reference bundles cloned/fsck-checked at original HEADs; checkout archive bytes and all moved files/symlinks verified.
- Parent index: zero model gitlinks, 55 ordinary project files; unrelated staged entries unchanged.
- Captions, local environments and runtime/data/checkpoint paths retained and ignored. Original environment path text retained before repairs.
- Workspace discovery from /tmp: passed.
- Tutorial 0 render refreshed and validates after relocated-link edits.
- Docs content tests: 8 passed, 1 blocked. Course 1 has 24 pre-existing edited lesson files that do not match its prior render manifest; its first lesson contains a site route rejected by the existing publisher. No unrelated course rewrite was performed.
- Fresh docs build/browser checks: blocked by that pre-existing Course 1 state; not claimed as passed. The last successful docs build is served via the managed preview at the existing URL until a valid course render permits normal dev startup.

Next learning action: linear() in the relocated foundations lab; the two new encoder projects remain planned. Next documentation maintenance action belongs to the existing Course 1 rewrite: finish its source/render reconciliation, then run the normal docs build/browser checks and restart the managed dev service. Do not repeat migration or recreate any lab.
