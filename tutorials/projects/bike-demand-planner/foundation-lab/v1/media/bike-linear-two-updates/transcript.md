# Work two batch updates with numbers we can check

## Use four rows and two features

Before trusting the real matrix, use four rows and two features. These values are an illustrative fixture. The targets are five, two, four and eight. Initial weights are one and one, with bias zero. Matrix times weights plus bias gives predictions one, one, two and one. The first row calculation is one times one, plus zero times one, plus zero. Its target is five, so its residual is negative four. Residual always means prediction minus target. Batch size four differs from feature count two. That makes incorrect transpositions easier to catch than a square fixture would.

## Average half squared errors

All four residuals are negative four, negative one, negative two and negative seven. Squaring and adding gives seventy. Mean half squared error divides by twice the batch count, which is eight here. The initial objective is eight point seven five. The factor one half cancels the two when differentiating a square. Averaging prevents the update magnitude from growing merely because the batch contains more copies of the same observations. This objective has squared units on the real rental problem. It is different from the reported mean absolute error in rentals per hour. A training objective and a validation reporting metric can serve different roles.

## Differentiate each shared parameter

Each row contributes its residual times its feature to a weight gradient. Add those contributions, then divide by four. The first coordinate gives negative five. The second gives positive one. The bias gradient is the average residual, negative three point five. Notice the last row. Its residual is negative seven and its second feature is negative one, so its contribution to that weight is positive seven. All predictions are too low, yet one weight gradient is positive. Feature signs matter. At learning rate zero point one, subtracting these gradients gives weights one point five and zero point nine, with bias zero point three five.

## Recompute before the next update

This shortened source trace calculates residuals, averages both gradients, then subtracts the selected step. After the first update, predictions are one point eight five, one point two five, two point seven five and two point four five. The objective falls to five point three five six two five. For the second update, recalculate residuals from those predictions. They are negative three point one five, negative zero point seven five, negative one point two five and negative five point five five. Reusing the first gradient would apply a different update. The new weight gradients are negative three point eight seven five and positive zero point eight eight seven five.

## Finish the second update carefully

Add the four new residuals. Their sum is negative ten point seven, so the new bias gradient is negative two point six seven five. Subtracting one tenth of that from zero point three five gives bias zero point six one seven five. The updated weights are shown beside it. These parameters produce the four displayed predictions and objective three point three one eight zero two seven. Independent decimal arithmetic caught a mistaken hand expectation for this bias before acceptance. Central finite differences separately check the analytic derivatives, with a discrepancy below one billionth. Known arithmetic and derivative checks help establish different parts of the implementation.

## Change only the step size

Now return to the original parameters and change only the learning rate to one. The same gradients give weights six and zero, with bias three point five. Predictions become nine point five, three point five, nine point five and fifteen point five. Their mean half squared error is thirteen point six two five, higher than the initial eight point seven five. The derivative was correct, but the step overshot. As another check, duplicate every fixture row. The averaged objective and gradients stay unchanged because their sums and row count double together. Correct gradients, appropriate reduction and a usable step size are separate requirements.
