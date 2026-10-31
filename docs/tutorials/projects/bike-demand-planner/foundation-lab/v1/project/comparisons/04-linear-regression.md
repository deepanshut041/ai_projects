# Comparisons: Fit linear regression and follow its gradients

The current fit is one deliberately bounded reference. These comparisons explain the choices without turning this part of the project into a large search. The [worked notes](lessons/04-linear-regression.md) contain the source traces and exact results.

## Raw indicators or standardized coordinates

Both can represent the same linear function with an intercept when every saved scale is nonzero. Raw indicators make a row easy to read: only four coefficients contribute to Monday at eight. Standardized coordinates center the training columns and give them population variance one, which makes a common gradient-descent rate easier to use across differently frequent indicators.

We choose training-only standardization and save an equivalent raw parameterization for inspection. The equality between their predictions is checked. Scaling does not add information or remove collinearity. Fitting a fresh scaler on validation would change the predictor and break the intended boundary. A constant training column uses scale one, and its unseen variation remains unsupported evidence.

## A scalar calculation or a matrix operation

The scalar fixture is the independent explanation of what the vector operations mean. For four rows and two features, it gives predictions [1,1,2,1], residuals [−4,−1,−2,−7], mean half squared error 8.75 and gradients [−5,1,−3.5]. The matrix implementation computes the same reductions in float64 and extends them to all 8,645 rows.

We keep both as different forms of evidence. The scalar Decimal calculation checks known numbers; central finite differences check local derivative consistency; unequal dimensions and rejected target-column shapes catch accidental broadcasting. We do not keep a second production trainer in the notes.

## Full-batch gradient descent or a stable least-squares solve

The iterative method exposes initialization, gradients, step size and stopping. The independent solve answers whether the fitted predictions agree with a stable solution of the same training objective. It handles the rank-42 design without an assumed inverse of a singular normal matrix.

This run converges in 200 updates, and prediction differences from the reference are below 1e-6 rentals/hour on observed inputs. Coefficients are not required to match because the 46-column design is dependent. The reference is a numerical check, not a second model selected using validation. We do not interpret an empty solver residual-summary array as zero loss; we calculate errors explicitly.

## A small safe step or a large step

The four-row example shows a concrete overshoot. Rate 0.1 lowers the first-step loss to 5.35625; rate 1.0 raises it to 13.625. The training-only real-design bound is about 0.93942, and the selected rate is 0.1. This choice is made before validation scoring.

We have not compared optimizer families, mini-batches, schedules or tuned rates. Those studies need the checked model produced here. This run establishes one correct full-batch path and records an explicit gradient tolerance and maximum update budget.

## An additive calendar model or separate hourly curves

The group baseline learns 48 `(hour, workingday)` means. The linear model learns hour, weekday, month and flag main effects, with no hour-by-working-day interactions. Neither family contains the other under the current features. A larger feature count therefore does not imply a richer version of the baseline.

The measured group/linear training RMSEs are 78.8394 and 79.5931; validation MAEs are 91.5729 and 116.7312. These results do not look like a linear optimizer that simply needs more updates: the stable solve agrees with it. Missing interactions and changing calendar/demand distributions provide concrete issues to examine later, without claiming one isolated cause for the entire gap.

## Retain negative predictions or clip them

The current unconstrained model has 724 negative validation forecasts. Keeping them preserves the common real-valued, unrounded and unclipped evaluation contract. It exposes an actual limitation, including the −70.6779 prediction for five observed rentals at 02:00.

Clipping at zero could be a reasonable policy to compare, but it changes the function being scored. We have not silently applied it or reported hypothetical improved metrics. Nonnegative models, transformed targets and alternative losses also change assumptions and belong to separate later comparisons.

## Optimize squared error or report absolute error

Mean half squared error gives a smooth objective with a simple analytic derivative and stable least-squares reference. MAE remains the primary evaluation summary in rentals/hour, with RMSE and signed error alongside it. Their roles differ. A smaller training squared-error objective does not guarantee smaller validation MAE, and signed errors can cancel even when individual absolute errors are large.

All three compared predictors use the same frozen validation identities and prediction policy. That holds the evaluation question fixed while the fitted function changes. The reserved test does not choose any of these settings.

Return to the [worked understanding notes](lessons/04-linear-regression.md).
