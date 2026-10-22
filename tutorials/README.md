# Project tutorials

All tutorials live here, inside the consolidated Projects docs repository. They are eligible for Git. Use C++ and Python; preserve editable diagrams, sources, examples, scenarios, retrieval exercises and whiteboard preparation beside each lesson.

[Foundation Lab: Bike Demand Planner](projects/bike-demand-planner/foundation-lab/v1/lessons/01-prediction-contract.md) follows the complete hourly demand project in ten lessons. Its notes to understanding retain the worked maths, source, actual comparisons, diagrams and 51 short captioned videos before their matching explanations. Learning points, comparisons, setup and reference records sit beside that route.

Begin with [Tutorial 0 — Deep learning foundations for Attention Is All You Need](projects/p00/transformer-foundations/v1/lessons/01-prediction.md). Advanced Python is assumed. The [25-topic outline](projects/p00/transformer-foundations/v1/project/course-outline.md) covers core maths/training, data/text/embeddings/positions, attention and all original Transformer sublayers, sequence/evaluation/recovery contracts and paper assembly. Each topic has essential explanations, calculations, checks, practice and personal notes. [The prerequisite map](projects/p00/transformer-foundations/v1/project/prerequisite-map.md) connects each concept to its next use.

Continue into [Course 1 — Transformer Foundations: Translation and RoBERTa](projects/p00/transformer-course/v1/lessons/01-start.md): 24 topic lessons and two complete model builds. Then follow [Course 2 — Small GPT](projects/p01/small-gpt/v1/lessons/01-next-token.md). P00/P01 IDs, existing lesson URLs and the actual foundations child checkpoint stay stable.

Tutorial availability and author checks never establish learner understanding or hosted completion.

Try each prediction before revealing its worked answer. For live teaching, ask the agent to work through one question at a time; learner knowledge is assessed from actual answers. `index.json` tracks P00–P19 without generating placeholder courses for the entire roadmap.

Each lesson revision lives at projects/PROJECT_ID/topic/revision/, using either a stable roadmap ID or a named application ID. Its lesson.json and lesson Markdown own the authored source; render.json binds the compiled Markdown, diagrams and downloadable artifacts. Reinspect changed sources, validate and render again before rebuilding the existing documentation site.

From workspace home:

```bash
./ai learn list
./ai learn validate --lesson projects/p00/transformer-foundations/v1
./ai learn examples --lesson projects/p00/transformer-foundations/v1
./ai learn render --lesson projects/p00/transformer-foundations/v1
./ai build project-docs
```

Use the workspace-local [developer-learning-tutorials](../../../../ai_research/research_docs/.agents/skills/developer-learning-tutorials/SKILL.md) and [feature-whiteboard-defense](../../../../ai_research/research_docs/.agents/skills/feature-whiteboard-defense/SKILL.md) skills, exposed through .agents/skills/ in the Projects umbrella. From workspace home read the canonical paths; there is no root skill-link directory. Add lessons project by project using real evidence. Keep model weights, datasets, environments and generated runs ignored.

## Read tutorials across Markdown files

The site displays one tutorial overview with ordered lesson pages, a shared contents menu, previous/next links and a collapsible mobile menu. Chapters are declared in lesson.json with stable IDs and content_path; each Markdown file is composed with its declared diagrams, tables, scenarios and practice. The site reads the verified files directly, so lesson.md is a complete export rather than the only reading page.

Exported project, practice and research Markdown opens inside the same reader. Package-relative links use the lesson directory as their base, following the existing authoring/compiler contract; links to validated Markdown become reader routes and retain heading fragments. Other validated artifacts remain downloads. Only tutorial overviews appear as library cards; chapter/support text participates in search.

After authoring changes, validate/render the lesson and run the site checks. Render hashes still reject unreviewed changes. To download or print every chapter, use **Complete lesson** in Contents. Existing overview URLs remain stable. See [reader implementation and verification](../implementation/2026-10/multi-page-tutorial-reader/06-review.md).
