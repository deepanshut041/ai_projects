# Quantization Lab documentation navigation

Status: complete, 2026-10-26.

The user reported that the newly prepared project area was missing from the Projects website. The live reader's homepage and sidebar contained Foundations, Transformer Labs and Language Models, but no Quantization Lab. Its collection URL returned 404. Document Assistant's tutorial and media were already present and reachable by direct tutorial links.

The website owns a collection registry in `website/scripts/projects.mjs`; it did not include the new `projects/quantization-lab` folder. The TUI registry was already updated and is independent. The existing Document Assistant README already links the first shared lesson, so no tutorial/media regeneration or source edit is needed.

Lead: workspace AI project workflow, with Playwright browser validation. This compact reader-maintenance record follows the planning coordinator. Changes belong to Projects Docs; unrelated Attention edits and all existing authored media remain preserved.

## Implementation and acceptance

- Add Quantization Lab to the website collection registry. The existing renderer then discovers both child projects, their local study records, portable README links and collection entries.
- Add a regression for homepage/sidebar reachability, both projects, the real tutorial route, and local/public study-record separation at root and prefixed paths.
- Exercise homepage → Quantization Lab → Document Assistant → first lesson and actual video playback on desktop/mobile.
- Run the required native checks, static build and full browser suites for both hosting bases. Restart the managed reader so its imported renderer uses the fix.

## Verification

- `npm --prefix website test`: 22 passed, including the new collection/tutorial/public-local regression.
- `npm --prefix website run build`: passed for `/project_docs/` and `/`.
- `npm --prefix website run test:browser`: passed for `/project_docs/`: 209 records, four project collections / 146 local project pages, video playback and 175 highlighted pages. The new route is exercised at 1440px and 390px.
- `SITE_BASE=/ npm --prefix website run test:browser`: passed with the same 209 records, four collections / 146 project pages, real desktop/mobile project-to-tutorial playback, video fixtures and 175 highlighted pages.
- Managed reader restarted successfully at port 4273. Its collection index now includes Quantization Lab / two projects. Actual homepage → collection → project → first lesson navigation and play/pause passed; 390px has no horizontal page overflow. All 24 existing Document Assistant video bindings remain present.
- All 137 Document Assistant / Quantization Lab files in the earlier production source-pin manifest are unchanged. Its unrelated Attention pins belong to a subsequent, pre-existing course revision and are outside this fix.

Temporary logs and screenshots live in ignored `website/test-results/quantization-*`; no generated media is added to Git.

The normal managed reader is available at <http://127.0.0.1:4273/project_docs/projects/quantization-lab/>. Document Assistant's 24-lesson tutorial starts at <http://127.0.0.1:4273/project_docs/records/tutorials/document-assistant-quantization-lab-v1/chapters/contract/>. Next: use the restored collection entry to continue the existing project/tutorial route.

This change repairs documentation access only. Model implementation, model results, learner understanding and hosted execution retain their existing states. No narration/provider requests or remote publication are involved.
