# AI Home CLI

One terminal home for **Projects** and **Research**, with headless commands using the same controller. Run `./ai` from the AI home, or the launcher inside `ai_projects`. Python 3.12, uv and Linux are required for the CLI. The environment is locked to the adapted reference's Textual version.

## Terminal controls

| Key | Action |
| --- | --- |
| Arrow up/down | Select entries, continuing between Projects and Research |
| `f` / `Ctrl+U` | Focus search / clear it |
| `s` / `x` / `r` | Start / stop / restart selected site |
| `b` / `t` | Build / test selected repository |
| `o` / `p` | Open site / focus port editor |
| `Esc` | Focus the selected project table |
| `q` / `Ctrl+C` | Exit; owned runs keep running |

Buttons also provide Run, Setup and Cancel. The active table alone shows a selection cursor. Search covers both groups; status refresh and terminal resize preserve selection and unsaved port edits. Logs have separate server/job selectors. Only valid actions are enabled. The UI is checked at 80×24 and 120×36.

## Headless commands

```bash
./ai list
./ai status --json
./ai doctor --json
./ai plan attention-lab test --json
./ai start project-docs
./ai restart research-docs --port 4294
./ai stop research-docs
./ai port research-docs 4274
./ai run attention-lab --detach
./ai logs attention-lab --job
./ai stop attention-lab --job
./ai check --action test --group Projects
./ai stop-all
./ai tui --smoke
```

`--root PATH` before the subcommand selects another AI workspace. Job actions wait by default and return failure for failed/cancelled jobs. `--detach` returns immediately. `start` waits up to 20 seconds for HTTP readiness; `--no-wait` returns immediately. A readiness timeout retains the owned process and logs for inspection.

`doctor` distinguishes required uv/Git/Node/npm from optional native build tools, checks Node compatibility and missing checkouts, and returns failure when required prerequisites are missing. `check` aggregates the supported actions, preserves each job's logs and returns failure when any requested check fails.

## Discovery and custom commands

The manager reads submodules in `ai_projects/.gitmodules` and `ai_research/.gitmodules` each time. Python projects get frozen uv setup, pytest, wheel/sdist builds and `demo.py` runs. Node projects get their native scripts and setup; generic CMake build/test support is retained. Missing submodules remain visible as missing. Discovery never alters parent pointers.

`ai-workspace.toml` adds or overrides entries. Groups must be Projects or Research. Commands are nonempty argv arrays executed without a shell:

```toml
[projects.my-service]
path = "ai_research/my-service"
group = "Research"
label = "My service"
port = 4390
url_path = "/"
start = ["python3", "server.py", "--port", "{port}"]
test = ["uv", "run", "--frozen", "pytest", "-q"]
```

`{port}`, `{project}` and `{workspace}` expand inside argv arguments. `PORT` is supplied to services. Paths must stay inside the workspace; raw `sources/` cannot be managed as a project. New sites should honor `PORT` or use explicit flags. Additional auto-discovered websites use ports starting at 4480 for Projects and 4380 for Research.

## Authoring and backup

See [setup](../../SETUP.md) for `new`, `task`, `research`, `snapshot` and `relink`. Research records support paper, experiment, question and collection drafts. Source registration does not download or mark a source reviewed; explicit `research fetch` performs a bounded download with provenance.

## Ownership and state

`.ai/` holds process records, port overrides, environments, builds, repository caches and logs. Before signalling, the controller checks PID start time, launch token, command line and process-group identity. State writes are locked/atomic; malformed JSON fails visibly without overwrite. A removed registry entry can still have an owned run; `stop-all` includes it. Each log rotates at 2 MiB with one archive; the latest three runs are kept, and the UI reads a bounded tail.

Run controller/authoring/navigation checks with `cd cli && uv run --frozen pytest -q`. Tests use real detached services, unrelated listeners, concurrent launches, stale PID protection, child cancellation, bounded logs, command isolation, true local submodules, task preservation, download limits and headless TUI controls.
