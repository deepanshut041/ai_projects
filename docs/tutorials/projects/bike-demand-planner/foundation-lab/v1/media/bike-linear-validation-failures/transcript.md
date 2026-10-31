# Compare the forecasts and explain the failures

## Return to the two later Monday hours

The later Monday examples make the failures concrete. On the ninth of January, twenty twelve, the two oclock target is five rentals. Linear regression predicts negative seventy point six eight. At eight, the target is four hundred and twenty, but the linear forecast is one hundred and seventy four point seven eight. The group predictor gives about seven point five one and three hundred and forty nine point five zero. The later eight oclock vector matches the earlier training example, so linear prediction is identical. There is no year or demand trend input. The negative quiet hour is impossible as a count. Both observations reveal limitations of this fitted calendar function.

## Score the same frozen validation rows

All three predictors score the same four thousand, three hundred and fifty seven validation identities, with the same metric definitions and unclipped prediction policy. The constant mean absolute error is about one hundred and fifty four point two five. The group model reaches ninety one point five seven. Linear regression gives one hundred and sixteen point seven three, about twenty five point one six worse than groups. Its root mean squared error is about one hundred and fifty seven point nine eight. Linear improves on the constant but loses this comparison. Convergence and stable solver agreement establish a numerical result, while this matched validation comparison measures its practical performance.

## The two model families express different patterns

Why can a model with month and weekday inputs lose to a smaller looking baseline? The additive model gives working day one shared coefficient. Conditional on the other main effects, that shift is the same across hours. The group model has a separate mean for every hour and day type pair. Its morning working day gap can differ from its nighttime gap. These model families are not nested under the current features. More columns do not guarantee that the group pattern is included. Even training root mean squared error is slightly worse for linear regression, and the stable solver agrees with its fitted predictions. Additional identical updates cannot create the absent interaction.

## Retain the function that was actually scored

The runner predicts using saved state, attaches the required record, target time and issue time, then calls the shared scorer. It does not silently clip negative values or round counts. There are seven hundred and twenty four negative validation forecasts. The minimum is about negative one hundred and three point four three. At the quiet Monday hour, subtracting the target five from negative seventy point six eight gives residual negative seventy five point six eight. Clipping could define a useful policy comparison, but it would be a changed prediction function requiring a retained result. Keeping the raw result visible lets us diagnose this model honestly.

## Use the result to choose the next question

The linear model underpredicts overall by about ninety five point zero eight rentals per hour. Its worst validation miss predicts about two hundred and ninety six point four eight against nine hundred and fifty seven. Calendar frequencies and average demand differ between periods, while omitted conditions may matter too. These observations do not isolate a single cause. More updates after convergence should give essentially the same solution. Adding interaction features changes the representable function and requires a separate comparison. The checked gradient path is ready for later studies, and the group model remains the stronger observed reference. For review, explain why one training hour can worsen while the shared objective improves, then connect that to the validation limitation.
