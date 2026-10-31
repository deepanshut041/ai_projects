# Comparisons: Build and compare simple calendar predictions

These comparisons keep the current model and evaluation choices connected to their purpose. They also give us a fair way to decide what to try next, without treating every possible variant as work to implement immediately.

## A global mean and a calendar-group mean

The global predictor fits one number from every recorded training hour: 143.7944 rentals/hour. Its strength is that we can explain its entire fit and prediction path. It estimates a global level and gives every later model a concrete reference on the same chronological validation hours.

The implemented calendar predictor fits one mean per observed hour/working-day pair, also using training rows only. Monday's 02:00 and 08:00 requests now select 7.5146 and 349.5040 against outcomes of 5 and 420. At 08:00, switching to a valid nonworking-day request selects 76.4348. The model represents two different daily profiles, rather than fitting one additive working-day offset for every hour.

Grouping divides the evidence across fitted values. We save support counts and use the row-weighted global training mean for an absent valid pair. All 48 pairs occur in this training year, so fallback usage is zero in the real comparison; the independent four-row fixture verifies the missing-group branch. An hour-only fallback, shrinking sparse groups toward the global mean or more specific weekday/month groups would each change the model and need its own declared comparison. They have not been fitted here.

| Held fixed | What changes |
| --- | --- |
| 2011 training rows, 4,357 chronological validation identities, feature schema, issue-time boundary, real-valued predictions and metric definitions | One global constant becomes up to 48 hour/working-day constants plus a global fallback |

Validation MAE falls from 154.2452 to 91.5729 and RMSE from 208.1723 to 137.3900. The relative MAE reduction is 40.6316%. This measured development result favors the group model under those size metrics, while signed error remains −72.3422. It improves 3,323 hourly absolute errors and worsens 1,034. The result supports a more useful reference predictor, not an unconditional winner for every hour or final-test performance claim.

## Fitting a mean and fitting a median

With a constant prediction `c`, minimizing training squared error leads to the mean. Minimizing training absolute error leads to a median. One way to feel the difference is with training targets `[0, 0, 30]`. Their mean is 10: absolute errors `[10, 10, 20]` give MAE 13.3333, while squared errors average to 200. The median is 0: absolute errors `[0, 0, 30]` give MAE 10, while squared errors average to 300.

The high target pulls the mean upward because squared error gives its large miss considerable influence. Both fits have clear objectives. Our current mean baseline is a squared-error reference reported under the project's common MAE/RMSE metrics. The numerical example explains why we should not call it the best constant under MAE. A real median comparison remains optional and has not been run on the validation period.

## MAE, RMSE and signed error

For residuals `+2` and `−8`, MAE is 5, RMSE is about 5.831 and mean signed error is −3. MAE measures typical absolute miss in the same rentals/hour units as the prediction. RMSE gives larger misses extra weight through squaring. Signed error describes the overall direction after high and low estimates offset each other.

The cancelling case `+10` and `−10` is a useful safeguard: signed error is zero, although both predictions miss by 10. The actual training result behaves similarly at aggregate scale. Training signed error is essentially zero by the algebra of the fitted mean, while training MAE is 106.5935. This is why the signed metric supports the error-size metrics rather than replacing them.

We have kept MAPE out of the headline results. Small or zero hourly counts make a percentage denominator difficult to interpret, and the project needs an error expressed directly in rentals/hour. Adding another metric later should answer a concrete decision question and retain its definition in the run record.

## An average deficit across all hours and across underpredicted hours

For deficits `[0, 8]`, the overall mean is 4 across two hours. The conditional mean is 8 across the one hour with a positive deficit. Changing the denominator changes the question. We save the two names and the underprediction count so later comparisons can be read without guessing which convention was used.

For the real validation period, the overall deficit is 112.8254 across 4,357 hours. The conditional deficit is 205.5962 across 2,391 underpredicted hours. A future model could change both the number of underpredicted hours and their severity. Keeping counts beside means lets us see that distinction. These are prediction deficits against observed rentals, with no station-stock or unmet-demand claim.

## Training error and later validation error

The global mean has MAE 106.5935 on training and 154.2452 on validation. The group means have 55.4280 and 91.5729 respectively. Training determined each model's fitted values; validation is later data kept outside the fit. The gaps provide evidence about these specific periods, rather than reserved-test performance or a unique cause of their difference. Extra groups cannot worsen the minimum training squared error because setting all their values to the global mean reproduces the simpler function. Validation improvement still needs measurement, and squared-error minimization does not guarantee the best MAE.

Both completed candidates use the same validation membership, row alignment and metric reductions. Later candidates retain that contract. Refitting either model with validation targets would produce a model with information from its own evaluation outcomes, losing the historical forecast comparison.

## Raw arrays and forecasts with an hourly identity

A pair of equal-length arrays is convenient, but it cannot tell us whether the third prediction belongs to the third target hour. Retaining record ID, target time and issue time makes that relationship checkable. The scoring function rejects missing, reordered and duplicated forecast identities before calculating a summary.

Identical constant predictions can conceal an ordering mistake in the numeric result. Calendar-dependent predictions will differ across hours, making the same mistake damaging. The identity contract is therefore established in this first evaluator and will remain fixed as the predictor changes.

[The complete walkthrough](lessons/03-calendar-baselines.md) follows these choices through source and actual outputs. [The tracker](project/implementation.md) records what has executed and what remains for the next comparison.

Return to the [worked understanding notes](lessons/03-calendar-baselines.md).
