# Build the prediction and measure its error

## 0:00 — Give the inputs matching parameters

Return to the original eight o’clock request and inputs one, two. Give the matching positions weights ten and five. Add a bias of twenty. The inputs describe the hour, while these three parameters define the current prediction rule. All parameter values are chosen for arithmetic practice. They are not learned demand estimates from the dataset. We will keep this starting state fixed when comparing two alternative targets, so the effect of the changed outcome stays visible.

## 0:33 — Calculate the two weighted contributions

Take the first input, one, and its matching weight, ten. Their product is ten. Pair the second input, two, with weight five. That product is also ten. Adding the two products gives twenty. This subtotal contains both weighted contributions; the bias still needs to be added.

## 0:56 — Add the common offset to obtain the prediction

The complete prediction adds a bias to the weighted subtotal. Here the subtotal is twenty, and the bias is another twenty. The resulting estimate is forty rentals. These arithmetic contributions do not establish separate causes of demand. The bias provides a common offset, and the calculation needs no later target.

## 1:20 — Read the same forward path in the source

The prediction function receives features, weights and bias. The source checks that the vectors have compatible nonempty shapes and finite values. It then pairs the entries, multiplies each pair and sums the products. The final addition supplies the bias. This displayed excerpt shortens the validation code, but preserves the executed arithmetic. The full file is linked directly below the video in the notes. A real-valued output is a point estimate. It can be fractional even though an observed number of rentals is an integer.

## 2:01 — Reveal the later outcome and retain the sign

Suppose the later observed count is forty five. This target is illustrative, just like the starting parameters. Use prediction minus target as the residual convention. Forty minus forty five is negative five. The model is five rentals too low. Keeping this sign matters because the update must move the prediction toward the target. Changing the residual convention in one function while keeping the old derivative elsewhere would reverse that correction. The target is used after the forecast, when we measure its error.

## 2:39 — Turn the residual into half squared error

Choose half squared error for this one-observation calculation. Squaring negative five gives twenty five. Multiplying by one half gives twelve point five. The half simplifies the derivative we will use next. There is no averaging denominator to add because this example contains one observation. This scalar has squared rental units. It is not an accuracy percentage, and it is not the project’s later primary evaluation measure, mean absolute error. The loss describes the current error size under a particular objective.

## 3:17 — Equal error sizes can need opposite corrections

Keep the inputs and parameters fixed, but change only the target to thirty five. The forward function still returns forty, because it never receives the target. The residual is now positive five, while half squared error remains twelve point five. Equal loss values do not mean equal update directions. One estimate is too low and the other is too high. Retain the signed residual and the dependence of the prediction on each parameter. The chain rule uses those relationships to calculate the correction, which is the next step.

