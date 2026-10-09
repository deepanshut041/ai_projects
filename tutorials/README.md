# Project tutorials

All tutorials live here, inside the consolidated Projects docs repository. They are eligible for Git. Use C++ and Python; preserve editable diagrams, sources, examples, scenarios, retrieval exercises and whiteboard preparation beside each lesson.

Start with [P01 GPT-1 foundations](projects/p01/gpt-1-foundations/v1/lesson.md). It is a reference-design orientation with runnable toy examples, not a completed model or demonstrated learner mastery. The remaining projects are planned in index.json; add their lessons as actual project work proceeds.

The P01 route now includes a prerequisite check, concept-dependency map, motivated masking/target derivations and a misconception check. Try each prediction before revealing its worked answer. For live teaching, ask the agent to work through one question at a time; learner knowledge remains unchecked until an actual answer is given. See the [improvement plan](../implementation/2026-10/learn-inspired-adaptive-tutorial-improvements/02-human-plan.md) for the source comparison and scope.

Each lesson revision lives at projects/pXX/topic/revision/. Its lesson.json and chapters own the authored source; render.json binds the compiled Markdown, diagrams and downloadable artifacts. Reinspect changed sources, validate and render again before rebuilding the existing documentation site.

From workspace home:

```bash
./ai learn list
./ai learn validate --lesson projects/p01/gpt-1-foundations/v1
./ai learn examples --lesson projects/p01/gpt-1-foundations/v1
./ai learn render --lesson projects/p01/gpt-1-foundations/v1
./ai build project-docs
```

Use the workspace-local [developer-learning-tutorials](../../../ai_research/research_docs/.agents/skills/developer-learning-tutorials/SKILL.md) and [feature-whiteboard-defense](../../../ai_research/research_docs/.agents/skills/feature-whiteboard-defense/SKILL.md) skills, exposed through .agents/skills/ in the Projects umbrella. From workspace home read the canonical paths; there is no root skill-link directory. Add lessons project by project using real evidence. Keep model weights, datasets, environments and generated runs ignored.
