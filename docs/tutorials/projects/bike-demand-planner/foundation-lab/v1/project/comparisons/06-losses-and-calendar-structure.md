# Comparisons: Change the loss, regularization and calendar structure

The [worked notes](lessons/06-losses-and-calendar-structure.md) follow the examples and code. This record keeps the alternatives together without treating every measured difference as the same kind of improvement.

| Question | Controlled change | Evidence | Interpretation |
| --- | --- | --- | --- |
| How much should a large miss pull? | Half-MSE / MAE / Huber10 / Huber50; same 45 inputs and Adam convention | Raw validation MAE 116.7294 / 117.5594 / 117.2829 / 116.7405 | A more robust loss improves the busy example but does not improve aggregate later MAE here. |
| How large should weights be? | L2 0 / 0.1 / 1 / 10; standardized coordinates and stable squared-error solve | Weight norms 105.5410 / 96.2645 / 53.7555 / 9.9291 | Strong shrinkage approaches a constant predictor. Small L2's raw-MAE advantage disappears after clipping. |
| How quickly does SGD move in chosen coordinates? | Raw versus training-standardized inputs; rate 0.1, 200 updates | Training half-MSE 4114.4970 versus 3167.5273 | Fixed-work optimization differs; direct-solve observed predictions agree within 2.3e-12. |
| What hourly curves can the model express? | One-hot45 versus cyclic8 versus hour/workingday69; no L2, stable training solve | Clipped validation MAE 109.1436 / 113.7926 / 100.5164 | Circle geometry limits a linear hourly curve to one wave; interactions add hour-specific day-type differences. |
| What output function is evaluated? | Raw versus max(0, prediction) | Exact one-hot MAE 116.7312 → 109.1436; interaction 108.6937 → 100.5164 | Clipping helps negative outputs without changing positive peak misses. |
| What should be carried forward? | Every retained model/policy plus both established baselines | Group/raw remains best, MAE 91.5729 | Freeze that development reference and retain the unsuccessful alternatives. |

The count is twelve comparison models plus two baselines. The zero-penalty one-hot direct fit is shared by the L2 and representation tables; it is not fitted again as independent evidence. One additional raw-coordinate least-squares solve checks prediction equivalence and is not a separate selection candidate. Twelve pilot fits choose rates before those main comparisons.

The objective family gives each loss the same three-rate and 200-update pilot allowance, selecting by its own training objective. The final 2,000 updates do not certify a global nonsmooth optimum. The scaling family deliberately uses fixed SGD work instead; the penalty and structure families use stable direct solves. These are separate questions, not a combined hardware-speed ranking. Full-batch operations are deterministic here, so repeated seed labels would create no independent order variation.

At the January 9 validation examples, both objectives and feature choices can improve one observation while worsening others. MAE predicts 218.3283 at target 420, Huber10 219.2912 and the interaction model 260.3130, versus the one-hot reference's 174.7785. The group predictor remains closer at 349.5040. At target 5 overnight, several new models still predict negatively before clipping. Neither a better training objective nor a positive-output policy establishes a complete forecasting solution.

Targets stay in rentals/hour, with no log transform. Huber is delta-scaled in its outer region, not normalized Smooth L1. MAE's zero subgradient is explicit, and MAE as a data loss is distinct from an L1/Lasso penalty on weights. Quantile, Poisson, log-target and L1-penalty models remain optional later extensions. None is implied by the implemented comparisons.

The final selection deliberately uses validation after all fitting and can therefore be optimistic after repeated development comparisons. The final test remains reserved. [Independent verification](evidence/06-losses-regularization-and-calendar-structure/independent-study-verification.json) reproduces predictions, scalar metrics, transform statistics, objective values, direct-solve stationarity and the frozen selection while preserving data and repository indexes.

Return to the [worked understanding notes](lessons/06-losses-and-calendar-structure.md).
