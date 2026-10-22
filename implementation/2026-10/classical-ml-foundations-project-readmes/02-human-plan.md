# Plan — Classical ML foundations project READMEs

Date: 2026-10-15 (America/Toronto)
Lead: ai-project-workflow through the workspace planning coordinator.
Supporting guidance: workspace developer-learning-tutorials and feature-whiteboard-defense; reference-design preparation only, no live rehearsal or model implementation.

## Scope

Create projects/foundations/README.md and separate in-depth README files in bike-demand-planner/ and ecommerce-purchase-predictor/. Preserve unrelated edits and ordinary folder ownership. Update canonical route/preparation records and the six compact entry-point snapshots. No dataset downloads, scaffolds, environments, training, application code, launcher/TUI changes, commits or publication.

## Decisions and sources

- User selected Bike Demand Planner first and E-commerce Purchase Predictor second, both classical ML. Separate regression/classification coverage; later neural/embedding work is outside this stage.
- UCI dataset IDs 275 and 468, their schemas/licenses and scikit-learn SGD/common-pitfall guidance inspected on 2026-10-15. Primary links are in the project READMEs; installed dependencies will be pinned at kickoff.
- Bike contract separates advance-known calendar inputs from realized-weather conditional estimation. Exclude target components; use a proposed chronological split.
- Purchase contract is offline outcome scoring because timestamped prefix records are not established. Exclude transaction-linked PageValues and conservatively exclude unverified aggregate rate features from the main benchmark. Use frozen stratified/group-aware records, calibration/threshold partitioning and untouched test labels.
- From-scratch NumPy linear/logistic models, gradients, losses and update rules will expose learning behavior; library estimators are independent/practical comparisons. This is proposed work, not executed experiments.

## Tasks and acceptance

| ID | Task | Dependency | Acceptance |
| --- | --- | --- | --- |
| T01 | Inspect instructions, canonical state, repository changes and primary sources | None | Ownership/scope reconciled; existing edits preserved |
| T02 | Write collection and both project READMEs | T01 | Separate folders; application/data/phase/experiment/numerical/recovery/exit contracts |
| T03 | Reconcile canonical state and compact snapshots | T02 | New practical order recorded without claiming learner/model completion |
| T04 | Review content, arithmetic, local links and whitespace | T02, T03 | Independent arithmetic, link checks and scoped diff review |

## Evidence boundaries

Current content is a project plan, not an executable tutorial or rendered website route. Verify Markdown structure/links and the worked arithmetic; no application/runtime tests or site rebuild are justified by these documentation-only changes. Exact pilot budgets, feature sets, partitions, success thresholds and dependency locks must be finalized from data/implementation evidence at kickoff. Follow the preserved P00 learner record separately.
