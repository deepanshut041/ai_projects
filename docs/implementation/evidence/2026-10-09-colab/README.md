# Colab guide verification — 2026-10-09

Shared requirements and local website only; no model notebook or hosted training was run.

- `npm test`: 7/7 content checks passed.
- `npm run build`: 14 records and 7 source references, configured /project_docs/ base.
- `npm run test:browser`: 14 records, search/filters/downloads/links/print/mobile passed.
- Playwright CLI: new guide loaded; 1440×1000 desktop and 390×844 mobile screenshots visually inspected. Mobile document width 390 equals viewport width 390; three tables present; zero console errors/warnings.
- Local Markdown links in the 14 targeted documents resolve. Projects root visible files remain README.md/AGENTS.md. No gpt-lab repository exists; roadmap milestones remain not-started.

[Desktop screenshot](colab-desktop.png) · [Mobile screenshot](colab-mobile.png)

Sources and design assumptions: [research](../../2026-10/colab-free-training-and-recovery-requirements/01-research.md). Canonical requirement: [COLAB.md](../../../guide/COLAB.md).
