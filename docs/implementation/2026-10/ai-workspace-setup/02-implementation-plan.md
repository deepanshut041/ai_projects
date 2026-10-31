# Implementation — AI workspace setup

The root `ai` launcher resolves its own location and delegates to the locked `ai_projects/cli` environment. Registry discovery reads both umbrellas' Git submodules and workspace TOML overrides. One controller serves the headless CLI and Textual UI. Token/PID-start-time/process-group identity controls signalling; locked atomic JSON and bounded logs preserve process state.

Node workflows use each site's own scripts. Python workflows sync frozen locks into `.ai/envs`, run real pytest suites/demos and build wheels/sdists into `.ai/builds`. Generic native CMake tests remain supported, while unsupported run actions fail clearly.

Research records, primary-source catalog, archives, templates and AI skills are owned by `ai_research/research_docs`. Raw archives stay ignored and never become required site inputs. The two local site links follow saved companion ports.

Repository creation produces only local initialization snapshots and a staged parent Gitlink. Source/history snapshots preserve editable work and Git histories while excluding caches, secrets and raw archives. `relink` repairs local cache/config paths after relocation and preserves declared external URLs.
