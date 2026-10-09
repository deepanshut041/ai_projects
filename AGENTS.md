# AI projects

This umbrella owns `cli/`, project templates, workspace tools, and implementation/verification records. Independent implementations are submodules under `projects/`; the independent `project_docs/` submodule owns the Projects website. Research belongs in sibling `../ai_research/`.

Read each child's instructions before editing. Preserve child history and unrelated edits. Submodule commits and parent pointers are separate decisions. Local initialization snapshots are part of setup; do not push or publish without a user request.

For CLI changes run `cd cli && uv run --frozen pytest -q` and `../ai tui --smoke`. For numerical changes compare with analytic answers or an independent reference. Test failure cases and state tolerances. Do not judge a model using its training loss alone.

Use `../ai task new TITLE --owner projects` for substantial work. The shared AI workflow is `../ai_research/research_docs/WORKFLOW.md`. Runtime state belongs in ignored `../.ai/`. Stop only processes whose recorded identity belongs to this manager.
