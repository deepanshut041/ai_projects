# Change the target to 35 and compare both updates

## 0:00 — Compare two outcomes from the same initial state

Treat forty five and thirty five as alternative outcomes for one illustrative request. Each case starts from weights ten, five and bias twenty, so both initial predictions are forty. The loop stores returned updates separately. It does not assign them back to the starting parameters. Starting the second case from the previous update would instead ask about successive training steps. That would mix an outcome change with a parameter change. Holding the initial state fixed makes the effect of the changed target easy to isolate.

## 0:36 — The new target reverses the residual and gradients

Forty minus thirty five is positive five. The model now predicts five rentals too many. Half squared error is still twelve point five, since the squared residual magnitude is unchanged. Multiply the positive residual by the two positive inputs to obtain gradients positive five and positive ten. The bias derivative is positive five. Each positive derivative says that increasing that parameter slightly would raise the current loss. Gradient descent therefore takes the opposite direction, while keeping its subtraction rule and learning rate unchanged.

## 1:14 — Apply the same subtraction rule

Apply rate zero point zero one to the first gradient. Ten minus zero point zero five is nine point nine five. For the second weight, five minus zero point one is four point nine zero. For the bias, twenty minus zero point zero five is nineteen point nine five. All three changes are downward in this case. They have the same magnitudes as the upward changes for target forty five. The equality comes from changing only the sign of the starting residual while keeping its magnitude and both positive inputs fixed.

## 1:56 — Account for the smaller overprediction

Recompute using the new weights and the same two inputs. The result is thirty nine point seven zero. The first weighted contribution falls by zero point zero five, the second by zero point two, and the bias by another zero point zero five. Together they lower the prediction by zero point three. The remaining overprediction is four point seven zero, with loss eleven point zero four five. One update reduced the error; it did not have to land on thirty five. The rate specifies a local parameter step, not a promise to solve the example immediately.

## 2:33 — Read the two updates side by side

Put the original and changed cases side by side. For target forty five, the prediction moves from forty to forty point three, reducing a negative residual. For target thirty five, it moves from forty to thirty nine point seven, reducing a positive residual. Both losses fall from twelve point five to eleven point zero four five. Their gradients and changes have opposite signs because their initial residuals do. This comparison is a controlled numerical example. It does not measure how well a model predicts a future bike dataset.

## 3:12 — Keep the feature sign in the explanation

The statement that both weights go down depends on these inputs being positive. In a different example, a negative input would reverse its matching weight derivative relative to the residual sign. The bias derivative still equals the residual, because its forward derivative is one. Always trace the residual, the matching input and the subtraction. That explanation survives changed feature signs and scales. Run the linked example to inspect both saved cases, then continue to the real hourly data when you can account for their different directions.

