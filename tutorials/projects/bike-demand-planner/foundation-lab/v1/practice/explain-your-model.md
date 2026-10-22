# Explain the model through one hour

These questions let you revisit the calculations in your own words. The worked answers are references; no learner response or completed review is inferred.

## Starting again from prediction 40, what happens if only the target changes from 45 to 35?

Residual changes from -5 to +5. With x=[1,2] and rate .01, gradients change from [-5,-10], -5 to [5,10], 5. Parameters become [9.95,4.9], 19.95 and the same input predicts 39.70. Compare both runs from the same initial parameters.

Check: Keep the initial state fixed, name the residual convention, and account for all three updates.

## A source has counts 10, an unrecorded hour and 30. Why is the observed mean 20 rather than 13.33?

There are two observed targets, with sum 40. Dividing by two gives 20. Dividing by three after inserting a zero manufactures a target and changes both the fit and evaluation question. The gap remains missing.

Check: Distinguish an absent row from an observed zero and retain the denominator.

## Why can January Monday at 08:00 and June Monday at 08:00 receive exactly the same group prediction?

The group key is only (hour, workingday). Both requests use (8,1), whose 2011 total is 86677 over 248 observations, giving 349.504032258. Month is not part of this fitted function even though the calendar schema contains it.

Check: Explain which fields the estimator actually uses and separate prediction from the later outcome.

## The fitted additive model converges but has validation MAE 116.7312, worse than groups. Is another optimizer guaranteed to fix that?

No. Reaching the chosen training objective does not make the additive function expressive enough to learn separate working-day hourly shapes, nor does it guarantee the validation objective improves. Inspect representation, loss and temporal shift separately.

Check: Separate optimizer convergence, function structure and later prediction error.

## Two runs each process 86450 examples, one with batch 32 and one with full batches. Do they take the same number of updates?

No. Work in processed examples is held fixed while batch size changes the number and noise of updates. A final partial batch uses its actual size. Optimizer state and schedules must use their declared update/example counters.

Check: Name the fixed budget and explain why update count and training passes are different quantities.

## If a raw model predicts -10 rentals and the observation is 5, what does clipping change?

Raw residual is -15 and absolute error is 15. A predeclared nonnegative policy returns zero, making residual -5 and absolute error 5. This is a different output function; it does not change the fitted weights or justify selecting the policy on final-test outcomes.

Check: Account for the changed prediction and error while preserving fit and selection boundaries.

## Why does averaging 64 trees not guarantee that the morning demand deficit disappears?

Averaging reduces variation when errors differ, but a shared historical underprediction remains in the mean. Training all trees on the same older demand period can preserve that common bias. Compare the actual later forecasts and counts.

Check: Separate diversity across trees from a shared error in their data or function.

## Why should a rush-hour slice report both MAE and its number of observations?

The subset answers a different question from all-hour error. Its mean must be weighted by the actual count when recombining a partition. A small or empty slice cannot be treated as equally supported or as zero error.

Check: Preserve the denominator, subset definition and undefined empty metrics.

## Why can restoring weights alone change the next Adam update?

The next step also depends on first/second moments, update index and bias correction, next batch/permutation/cursor, schedule, RNG and transformation/context. Restoring a compatible complete generation reproduces these; zeroing moments is a different continuation.

Check: Trace the next update state, rather than only whether a model file can be loaded.

## The forest has final MAE 103.6474 versus groups 125.4167. What happens to the claim if it is chosen now?

The original groups selection remains the validation decision. Choosing the forest now is informed by the final outcomes, so that same period cannot supply a fresh independent performance claim for the newly selected decision. Record the new choice and use new held-out evidence for a new claim.

Check: Retain the observed ranking reversal and distinguish selection evidence from independent assessment.
