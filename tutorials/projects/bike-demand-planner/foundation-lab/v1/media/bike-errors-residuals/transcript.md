# Read an hourly miss without losing its sign

## Begin with a real later Monday

Start with Monday, January ninth, twenty twelve. At seven in the morning, issue a forecast for the hour beginning at eight. The already fitted hour and working day mean returns three hundred and forty nine point five zero rentals per hour. The recorded outcome later turns out to be four hundred and twenty. That answer was not part of the original request. The prediction is too low, but we need both its direction and its size. This review keeps the earlier fitted model unchanged and studies its saved predictions. No new linear regression is fitted.

## Keep one residual convention throughout

Our scoring residual is prediction minus observed target. Subtract four hundred and twenty from three hundred and forty nine point five zero. The result is negative seventy point five zero rentals per hour. A negative sign means underprediction with this convention. The absolute error is positive seventy point five zero. A boosting correction used target minus prediction, the opposite sign, because it described a move the model still needed to make. Here we are measuring a fixed forecast, not taking another optimization step. Naming the convention prevents the same number being interpreted in opposite directions.

## Change the requested hour, then combine honestly

Change the requested hour to two in the morning on the same Monday. The corresponding group predicts seven point five one against a target of five, giving positive two point five one. This is an overestimate. Adding the two absolute errors and dividing by two gives thirty six point five one. Adding the signed errors instead gives a mean near negative thirty three point nine nine. Overprediction can cancel underprediction in the signed mean, while both contribute to mean absolute error. These two hours explain the arithmetic. They do not replace the complete validation population.

## Check the row before reducing its error

The reader first compares record identity, target time, issue time and row order with the frozen prepared rows. It also verifies the saved target and checks that each exported error column agrees with prediction minus target. Only after this alignment do we reduce residuals into summary numbers. One integration check deliberately reorders a prediction file and updates its checksum. The identity check still rejects it. A matching checksum alone cannot establish correct target alignment. The shown excerpt combines those identity fields with the arithmetic; the complete source retains the validation branches and specific errors.

## A deficit has two useful denominators

Across validation, three thousand, one hundred and forty three hours are underpredicted. Their deficits total about three hundred and fifty seven thousand and eighty nine rentals across one hour records. Divide by all four thousand, three hundred and fifty seven observations to obtain eighty one point nine six. Divide only by underpredicted observations to obtain one hundred and thirteen point six one. Those summaries answer different questions about average burden and the typical deficit when one occurs. Neither measures unmet demand or station shortages. The dataset records observed rentals and does not establish an inventory or financial consequence.

## Keep the large failures in the headline result

The complete validation mean absolute error is ninety one point five seven, with root mean squared error one hundred and thirty seven point three nine. Squaring before averaging gives large misses more influence. Taking the square root returns the result to rental units. Mean signed error is negative seventy two point three four, showing overall underprediction. The largest retained miss occurs on March twenty third at five in the afternoon: about three hundred and ninety five predicted against nine hundred and fifty seven observed. We retain that difficult hour. Explaining a failure does not authorize removing it from the score.
