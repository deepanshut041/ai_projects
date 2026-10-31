# Tracker — Air Quality Forecaster recurrent sequence README

Date: 2026-10-15 (America/Toronto)

| Task | Status | Evidence |
| --- | --- | --- |
| Q01 Source/context review | Done | UCI 501, official recurrent operator references and local project/guide state |
| Q02 Project README | Done | Separate 354-line README with task, causal data/window/state contracts, models, experiments, metrics, app and recovery |
| Q03 Overview and canonical summaries | Done | Fourth application/third stage, vision handoff, guides, D060 and six compact summaries |
| Q04 Verification | Done | 28 links, independent recurrence/BPTT/gate/count/timeline checks, fences/whitespace and scoped git diff --check |

Documentation preparation only. Preserve existing implementation/learner evidence and unrelated edits.

## Evidence and boundary

- Scalar RNN outputs approximately 0.462117 and 0.353724; half-squared loss 0.030326. Full-unroll central differences agree with derivatives [-0.135560, -0.099569, -0.351022] for input weight, recurrent weight and bias.
- LSTM cell update 0.925; GRU example approximately 0.415529. Analytic total parameters for F=8,H=32,Q=1 are RNN 1,377, GRU 4,065, LSTM 5,409.
- A one-step training target before 2015-03-01 is accepted; a two-step horizon crossing the boundary is rejected. Validation windows can use causally earlier history. Leap-day and aggregate source-period arithmetic checked.
- New recurrent folder contains only README.md; no nested Git repository or executable model was created.
- Concurrent Bike Demand Planner scaffold/source/lock/staging changes were observed and preserved. The overview was corrected to avoid a collection-wide no-code claim; that scaffold's runtime checks were not run or claimed here.
- No website rendering/source route changed, no model code exists for this project and no training/runtime/site suites were justified. Documentation checks are not learner mastery, task quality or hosted recovery.
