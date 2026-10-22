# Follow the chain rule through one update

## 0:00 — Follow two local changes through the chain rule

The loss depends on the prediction, and the prediction depends on the parameters. We need both links. Differentiating half squared error with respect to prediction gives prediction minus target. The factor two from the square cancels the half. Increasing a weight changes the prediction in proportion to its matching input. Increasing the bias changes the prediction one for one. The chain rule multiplies the local loss slope by the local prediction slope, giving a separate derivative for each parameter.

## 0:36 — Calculate all three gradients

For target forty five, the residual is negative five and the inputs are one, two. The first weight gradient is negative five times one, giving negative five. The second is negative five times two, giving negative ten. The bias gradient is negative five times one. The second weight is more sensitive in this calculation because its input is larger. That does not establish greater usefulness of the feature. It shows how input scale enters the derivative of this particular forward path.

## 1:14 — Subtract the scaled gradient

Choose the demonstration learning rate zero point zero one. Gradient descent subtracts that rate times each derivative. Ten minus zero point zero one times negative five is ten point zero five. Five minus zero point zero one times negative ten is five point one zero. The bias similarly becomes twenty point zero five. Subtracting a negative produces these increases. Their direction fits this underprediction with positive inputs, but the rule to remember is gradient subtraction. The rate controls the size of the step.

## 1:52 — Keep the before and after state separate

The source first forms the residual, weight gradients and bias gradient. Each new weight subtracts the same learning rate times its own gradient. The bias follows the same rule with its separate derivative. The update returns new values rather than modifying the old list. That makes before and after inspectable, and lets the alternative target start from the original state. Shape and finite-value checks guard this numerical contract. They do not replace the calculation or establish later forecasting quality.

## 2:32 — Check the new prediction and loss

Hold the same inputs and target fixed, and run the new parameters through the forward rule. The first weighted contribution rises by zero point zero five. The second rises by two times zero point one, or zero point two. Add the bias increase of zero point zero five. The prediction rises by zero point three, reaching forty point three zero. Its remaining residual is negative four point seven zero, and half squared error is eleven point zero four five. This is a same-observation arithmetic check, not an unseen-hour performance result.

## 3:13 — Use a boundary case to check the rule

What changes if the starting prediction already equals the target? The residual becomes zero, so both weight gradients and the bias gradient become zero. Subtracting their scaled values leaves the parameters unchanged. A zero learning rate also leaves them unchanged even when the residual is nonzero. The source checks these boundaries and compares analytic derivatives with central finite differences on another input. Our current successful step does not guarantee that every learning rate or later example improves. The next video changes the target while keeping the initial state fixed.

