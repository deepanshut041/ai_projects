# AI Projects

This umbrella owns the AI terminal manager, reusable tools, and project implementation/verification records. It organizes independent local repositories as Git submodules:

| Repository | Working scope |
| --- | --- |
| [project_docs](project_docs/README.md) | Projects website, source register and implementation guides |
| [linear-model-lab](projects/linear-model-lab/README.md) | Gradient-based least squares with an independent reference |
| [retrieval-lab](projects/retrieval-lab/README.md) | Deterministic TF-IDF retrieval and explicit ranking metrics |
| [attention-lab](projects/attention-lab/README.md) | Scaled dot-product attention, stable softmax and masks |

Run `./ai` here or `../ai` from the workspace home. See [the CLI guide](cli/README.md), [setup](../SETUP.md), [the project map](PROJECT_OVERVIEW.md), and [verification records](docs/verification/README.md). Research implementations and research documentation belong in the separate `../ai_research` umbrella.

The initial repositories and submodule pointers have local initialization snapshots. Bare local copies live in ignored `../.ai/repositories/`; no GitHub remote has been created. After copying/restoring the complete workspace, run `../ai relink` to rebuild local remotes. Use `../ai snapshot /path/outside/ai-workspace.tar.gz` for a portable source/history backup.

Add a new independent implementation with `../ai new project SLUG --title TITLE`. It creates a local initial snapshot and stages the umbrella's submodule entry, leaving that parent change reviewable. Read child instructions, commit child work in its child repository, then update the parent pointer intentionally. Publish remotes only when requested, replacing the local URLs with actual published repository URLs.
