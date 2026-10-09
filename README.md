# AI Projects documentation

An independent local documentation repository with authored Markdown, validated internal links, math rendering, full-text record search, source filters, Markdown/catalog downloads and print/PDF export.

```bash
npm ci
npm test
npm run build
npm run test:browser
npm run dev
```

Requires Node 22+ and npm. Browser checks require Playwright Chromium (`npx playwright install chromium` if missing). Open http://127.0.0.1:4273/project_docs/. From the workspace use `./ai start project-docs`. `SITE_BASE=/` supports root hosting; default local builds use `/project_docs/`. `PORT` selects another local port.

`content/` owns authored records. `catalog/sources.json` tracks seven primary references with explicit access/review statuses. `sources/` is an ignored local archive, excluded from builds; no raw source downloads are required to build. `dist/`, browser results and dependencies are generated/ignored. Ordinary builds use no external source archive. `npm run catalog:sync` updates archive counts from saved acquisition metadata.

This repository is a local submodule. It has no GitHub remote, deployment workflow or live publication configured. See the workspace README for local repository backup/relocation. Publication should be configured against an actual requested repository/domain.
