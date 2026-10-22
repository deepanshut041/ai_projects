# Comparisons: One hourly prediction and one numerical update

## Earlier information and later observation

The 07:00 calendar-only forecast uses information known at its issue time. A replay conditioned on the weather observed during 08:00–09:00 uses information from after that issue time. Both may be useful tasks, but their different information sets must be named before comparing performance. This worked contract excludes the later observation from the earlier input.

## Prediction and a training update

Prediction computes a number from the feature vector and current parameters. The loss/update path additionally needs a target. In the numerical example the same fixed [1,2] features first produce 40, then the hypothetical later target 45 produces a residual and gradient. The new parameters yield 40.30 on the same case; this is a training-step check, not a fresh forecast evaluation.

## Half squared error and squared error

The objective used in the code is one-half times squared residual. Its prediction derivative is the residual. Omitting the half doubles that derivative, so using the same learning rate would double this particular parameter step. Equivalent conventions require corresponding rate scaling. Neither convention changes the need for a common held-out task metric later.

## Targets on opposite sides of the same prediction

Hold features `[1,2]`, starting weights `[10,5]`, bias `20` and learning rate `0.01` fixed. Only the hypothetical outcome changes. These are independent alternatives, so the second row of execution does not inherit the first case's updated weights.

| Quantity | Target 45 | Target 35 |
| --- | ---: | ---: |
| Initial prediction | 40 | 40 |
| Initial residual, prediction minus target | -5 | +5 |
| Initial half squared error | 12.5 | 12.5 |
| Weight gradients | [-5, -10] | [5, 10] |
| Bias gradient | -5 | +5 |
| Weight changes | [+0.05, +0.10] | [-0.05, -0.10] |
| Bias change | +0.05 | -0.05 |
| Updated weights | [10.05, 5.10] | [9.95, 4.90] |
| Updated bias | 20.05 | 19.95 |
| Updated prediction | 40.30 | 39.70 |
| Remaining residual | -4.70 | +4.70 |
| Updated half squared error | 11.045 | 11.045 |

Both cases start five rentals from the target, and squaring removes the sign, so their initial objective values agree. The gradients retain the sign through the residual. Subtracting the scaled gradients raises the original underprediction and lowers the changed overprediction. Their equal final losses follow from the symmetric targets and identical features, parameters and rate; this is not a general claim that arbitrary targets or training steps have identical improvement.

Both weight directions follow the residual here because both features are positive. The corresponding direction can differ for a negative feature, while the formula `weight - rate*gradient` stays the same. See the [worked changed-target trace](lessons/01-prediction-contract.md#10-change-only-the-target-to-35-and-compare-the-update) for the parameter-by-parameter arithmetic and actual source.

## A readable scalar trace and a future vectorized model

The current standard-library functions express one observation's weighted sum and gradients. lesson 4 will introduce batch-shaped NumPy operations with explicit averaging. The scalar trace provides a known reference; it is not an implementation of that later batch trainer. Source, numerical checks and learner understanding remain distinct observations.

No optimizer ranking, real-data comparison or model-quality result has been produced.

Return to the [worked understanding notes](lessons/01-prediction-contract.md).
