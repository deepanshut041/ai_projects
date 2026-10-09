# AI workspace verification — 2026-10-09

This report records local implementation and verification. No external repository, deployment, hosted model call, raw dataset download or scientific acceptance was performed.

## Verified checks

| Surface | Command/evidence | Result |
| --- | --- | --- |
| CLI controller, authoring and terminal navigation | `cd ai_projects/cli && uv run --frozen pytest -q` | 39 passed; real process and workflow tests |
| Terminal startup | `./ai tui --smoke` | `ui: ok`, six entries, two groups |
| Toolchain and checkouts | `./ai doctor --json` | Required tools available, Node compatible, no missing repositories |
| Native setup | `./ai check --action setup` | All six entries passed |
| Implementation/content tests | `./ai check` | All six entries passed; 19 project/research numerical tests and 10 content tests |
| Distribution/site builds | `./ai check --action build` | Four wheel/sdist builds and two static sites passed |
| Demonstrations | `./ai check --action run` | All four CPU implementations passed |
| Prefix-hosted websites | `npm run build` and `npm run test:browser` in both sites | Search, filters, all records, links, Markdown downloads, print/PDF and mobile routing checked |
| Root hosting | `SITE_BASE=/` build/browser checks in both sites | Recorded separately; ordinary prefixed output restored afterwards |
| Skills | skill-creator `quick_validate.py` for four local skills | All four valid; referenced workflow exists |
| Git ownership | umbrella `git submodule status` | Six actual independent local submodules with separate histories |

Exact outputs, artifacts and per-check results are saved in `ai_projects/docs/verification/2026-10-09-workspace/`. Browser-generated screenshots and PDFs originate in each site's ignored `test-results/`; selected evidence is copied into the verification directory. Terminal screenshots cover 80×24 and 120×36.

## Measured starter behavior

- Linear demo held-out MSE: `0.010899796762890794`; maximum fitted-weight difference from NumPy least squares: `8.881784197001252e-16`.
- Causal attention demo: future-position weight maximum `0.0`; independent invariants and stability tests passed.
- Retrieval demo: original four-document corpus and three query judgments; raw rankings/metrics retained. This is implementation evidence, not broad retrieval-quality evidence.
- Synthetic OLS pilot: five seeds over four training sizes, with raw per-seed metrics, descriptive standard errors and implementation/environment provenance. The authored experiment record and downloadable JSON artifact contain measured results. These do not establish external validity.

## Issues found and corrected

- Reference-domain assumptions in test fixtures were replaced with AI ownership and ports.
- A service-control UI refresh could briefly expose stale button state; control completion now renders a fresh snapshot.
- Long code blocks caused mobile record overflow through a grid minimum width; content grids and articles now shrink correctly. At 390 px the corrected page's document width is 390 px.
- Links between the two documentation sites now follow persisted companion port overrides in managed build/start plans; a regression test covers this.
- No-test scaffolds fail explicitly. Existing task packages, records, source archives and snapshots are preserved instead of overwritten.

- PDF visual QA found an invalid inherited print rule and a viewport minimum height that created a blank trailing page. Both were corrected. Browser assertions now check white print background, hidden navigation and zero minimum page height. The two representative exports render as single, legible pages; PNG previews and PDF metadata are saved.

## Relocation verification

A source/history snapshot was restored into a temporary path containing spaces and invoked from `/tmp`. Local remotes were rebuilt with `relink`; doctor and TUI smoke passed. The restored Attention tests, Retrieval demo, source listing and Research site passed, including the downloadable pilot artifact. All restored processes were stopped, and the temporary archive/checkout was removed. Exact outputs are in `relocation-checks.json` and the corresponding logs.

## Practical boundaries

- Process supervision currently supports Linux; this was tested on the current machine.
- Python 3.12 and pinned dependencies are intentional. Initial dependency installation may require network/cache access; demos make no network calls.
- Git URLs are local seed repositories, not published remotes. Use a full source/history snapshot for relocation, then `./ai relink`. Publish against real requested remotes separately.
- Raw archives, datasets, weights, environments and generated runs are excluded from snapshots and site output. Preserve any needed ignored research artifacts separately.
- Existing Quantum files and repositories were used as read-only references.

## Cleanup and planning follow-up

In progress: consolidate root records, adapt/discover planning in both owners, and verify workflow preservation. Initial verification above describes the original setup; the follow-up results will be appended.

## Cleanup and AI planning adaptation — verified

The clarified cleanup request is complete. Root extras were removed or consolidated into the owning repository. SETUP is `ai_projects/SETUP.md`; the canonical records are this package. Existing examples and user roadmap remain. Runtime/Git state and histories were preserved.

The shared planning skill now contains the source-style coordinator, workflow, AI domain routes, helper, UI metadata and eight templates. It is available through `.agents/skills/planning/SKILL.md` in both Projects and Research, with current-owner defaults and explicit CLI owner routing. Duplicate CLI templates were removed.

Checks run for this follow-up: 42 CLI tests, 10 site content checks, five skill validations, local link/template checks, exact root-layout assertion, unchanged roadmap checksum, source-snapshot restoration, both owning helper contexts, relocated CLI task listing and TUI smoke. Machine-readable restoration evidence is `cleanup-planning-validation.json`. No model code changes, remote publication or roadmap edits were included.

## Pre-P01 repository review — 2026-10-09

The user explicitly requested clean local commits before Project 1. The current [commit review](07-pre-p01-commit-review.md) records scope, child-first ordering, checks and the stale diagram-count test correction. All current CLI/site/skill/example checks pass. Final Git identities and umbrella alignment are verified after committing. P01-M1 remains not-started; next action is gpt-lab M1. Earlier no-commit statements are historical.
