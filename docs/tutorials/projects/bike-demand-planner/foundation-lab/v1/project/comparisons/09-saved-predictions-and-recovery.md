# Comparisons: Reload predictions and recover the next training update

## Inference export and training checkpoint

| Question | Prediction bundle | Training generation |
| --- | --- | --- |
| What must be reproduced? | The same fitted response to a valid request | The same next update and eventual numerical state |
| Parameters and transforms | Yes; the group model has no learned scaler | Yes; the linear trainer retains its training-fitted scaler |
| Optimizer, order and progress | No ongoing training computation | Moments/index, permutation/cursor/RNG, example-based schedule and monitoring |
| Target access | None at prediction | Training targets only, verified against the saved context |
| Executed evidence | All 4,357 validation forecasts match after restart | 813-update/25,935-example final state matches after recovery |

The selected group model is exported unchanged. The separate bounded Adam run tests recovery without replacing that predictor or selecting a new model. Its training objective is not a new validation benchmark. Keeping weights but resetting moments changes the eventual parameters by up to 1.0029187 in the executed same-future-batch control.

## One live file and complete published generations

Updating one live file is simpler, but an interrupted write can destroy the only state. Complete numbered generations preserve a previous valid state and bind state/context through a receipt. The executed corruption and incomplete-publication probes reject generation 3, ignore the pending directory and restore generation 2. Incompatible contexts fail explicitly. This is a local single-writer design with file/directory synchronization; it does not establish hosted or arbitrary power-loss recovery.

## A forecast and an observed comparison

The API returns estimate, baseline and identities without a target. A separate observation endpoint supports historical comparison. Monday 08:00 predicts 349.5040; the optional recorded 420 produces −70.4960. The real missing 2 January 03:00 observation stays absent, with no invented zero or residual. Unsupported dates clear old output and show an error.

## Reproducible behavior and quality

Groups/raw remains the overall-MAE candidate at 91.5729, while the representative forest is stronger in June at 90.8782 versus 142.9952. Sorting the benchmark changes table order, not the active predictor. Stored fit/convergence/cost/failure records keep their actual scopes. Packaging and recovery preserve this choice and its limitations; neither improves its scores. Final-test assessment and the learner's own explanation remain separate next work.

Return to the [worked understanding notes](lessons/09-saved-predictions-and-recovery.md).
