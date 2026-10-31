# Reference inspection and design evidence

## Verified outer structure

Inspected on 2026-10-09:

- `/mnt/projects/quantum/README.md`: lightweight root launcher; independent Projects, Master's, and Research areas.
- `/mnt/projects/quantum/AGENTS.md`: explicit repository ownership; research implementations and docs remain independent; local source archives are excluded from publication.
- `/mnt/projects/quantum/quantum.code-workspace`: editor folders correspond to ownership boundaries.
- `/mnt/projects/ai`: empty before this setup. No user files or existing checkout need migration.

## Initial adaptation

Keep the lightweight workspace root and the independent Projects/Research ownership model. Remove the Master's area from the AI design. Review the Quantum CLI and research helpers before deciding which pieces can be reused safely.

## Completed reference inspection

The CLI registry/launcher, controller, detached runner, terminal navigation, process/port ownership, task templates, source catalog, static rendering/server, tests and installed toolchain were inspected. The adaptation decisions and primary sources follow.

## Inspected implementation and decisions

- Read the reference CLI registry, manager, detached runner, CLI parser, Textual TUI and their tests. Reused the tested process ownership, PID start-time/token checking, process groups, bounded logs, port checks and action controller. Adaptation hashes are in `reference-provenance.json`.
- Removed the Master's defaults and tables. Projects and Research are the only groups. AI runtime is `.ai/`; initial site ports are 4273 and 4274, separate from Quantum.
- Added Python repository discovery and locked uv setup/test/build/run plans. Generic CMake support remains; Quantum-specific demo mappings were removed.
- Read the reference static renderer, server, dev watcher, source catalog validation and browser checks. Adapted the renderer into two independent documentation repositories. Raw archives and old environments were not copied.
- Added explicit project/research scaffolding, implementation task packages, source registration, bounded explicit acquisition, aggregate checks, source/Git-history snapshots and local remote relinking after relocation.
- Initial repository identity is local only. Relative `.gitmodules` URLs point at local bare snapshots under `.ai/repositories/`; `./ai relink` rebuilds those after relocation without changing tracked URLs. Remote creation/publication is not assumed.
- Selected three small CPU project examples and a synthetic research protocol to validate the entire manager workflow, with independent numerical and failure-case checks.

## Primary sources inspected on 2026-10-09

| Source | Use | Scope |
| --- | --- | --- |
| [Attention Is All You Need v7](https://arxiv.org/html/1706.03762v7) | Equation 1 and section 3.2.1; attention forward contract | No reproduction of the paper's training/results |
| [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html) | Least-squares objective | Local gradient implementation compared with NumPy |
| [NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) | Independent SVD-backed numerical reference | Pinned runtime dependency separately from current docs |
| [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | TF-IDF/cosine baseline definition | Tiny authored corpus; no external embeddings |
| [scikit-learn evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#ranking-metrics) | Ranking metric context | Local definitions are explicit, binary relevance only |
| [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) | Locked environment behavior | Initial dependency installation needs network/cache |
| [Textual DataTable](https://textual.textualize.io/widgets/data_table/) | Terminal table/navigation API | Locked Textual version retained from reference |

## Evidence rules

Record inspected source separately from design choices and proposed research. Source references are starting points, not experimental results. Do not claim deployment, remote repository creation, or model quality without direct evidence.

## Cleanup and planning adaptation follow-up

The user clarified that extra root files/folders absent from Quantum should be cleaned, including the root `.agents` link. Sample repositories are retained. The master's repository points at Quantum Research Docs' shared planning coordinator. Its saved-package workflow is being adapted for AI Projects and Research, with one shared collection exposed inside those owners. The existing LLM roadmap is preserved unchanged.
