# Workspace repository map

Read [roadmap.md](roadmap.md), [progress.md](progress.md) and [resume.md](resume.md) for the learning sequence. The shared CLI is independently owned at [../../cli](../../cli/README.md); focused Research has [its own plan](../../ai_research/roadmap.md).

| CLI id | Owner | Native checks | Current boundary |
| --- | --- | --- | --- |
| `project-docs` | `ai_projects/docs` | npm content/build/browser | Local Projects site and canonical roadmap views |
| `linear-model-lab` | `ai_projects/projects/linear-model-lab` | pytest; NumPy reference; finite differences | Dense synthetic least squares |
| `retrieval-lab` | `ai_projects/projects/retrieval-lab` | pytest; hand-calculated ranking answers | Tiny lexical corpus and binary relevance |
| `attention-lab` | `ai_projects/projects/attention-lab` | pytest; mean-value/mask/permutation invariants | Single-head forward operation |
| `research-docs` | `ai_research/research_docs` | npm content/build/browser | Local Research site, four authored records |
| `generalization-lab` | `ai_research/generalization-lab` | pytest; known parameters; repeatable aggregates | Small synthetic protocol with five seeds |

P01 GPT-1 and the other roadmap models are not started; retained seed examples do not complete them.

Each code example has a standalone README, public Python package, demo, meaningful tests, pinned environment and wheel/sdist build. None needs credentials, a GPU, provider calls or downloaded models. Initial dependency installation may need network/cache access.

Possible extensions are saved in the sites' open-question records. Broader datasets, neural training, hosted APIs and reproduction studies need a concrete question and protocol before expansion. These starter projects establish a working toolchain; they do not establish production model quality or mastery.
