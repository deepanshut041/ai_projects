# Watch the real training hour move, then let the batch converge

## Start at the training constant

The real fit starts with zero weights and the training target mean as its intercept. All eight thousand, six hundred and forty five rows initially receive one hundred and forty three point seven nine rentals. For record fifty four, that is below the observed one hundred and fifty four by about ten point two one. The initial bias gradient is nearly zero because the residuals sum to zero at the training mean. That does not make each row correct. Calendar weights can still explain the differences around that average. The initialization gives us a direct connection from the earlier constant baseline to learned calendar adjustments.

## Follow two real full-batch updates

The first standardized hour eight gradient is about negative twenty four point nine six. Multiplying by rate zero point one and subtracting gives a positive weight near two point five zero. January has a positive gradient near twenty five point nine six, so its weight becomes negative. The whole batch determines these directions. The full vector forecast for record fifty four moves to one hundred and forty seven point zero four after one update, then one hundred and forty nine point nine three after two. A separate scalar calculation reproduces both. We must calculate the whole vector forecast because centering makes originally inactive columns contribute too.

## Repeat one full-training calculation

Each iteration predicts every training row and averages its loss and gradient contributions. The norm combines weight and intercept gradients. The fit stops when that norm is at most one ten millionth. The maximum budget is ten thousand updates. History includes the initial state as step zero, followed by each actual update. This run converges after two hundred updates, with a norm near nine point seven two times ten to the negative eight. If the budget expired first, the record would say unconverged. Validation is loaded only after fitting and the independent reference solve. Its answers do not select the scaler or stopping point.

## A shared fit can worsen one hour

The final forecast for our training hour is one hundred and seventy four point seven eight. It is now above its target of one hundred and fifty four. Yet the full training objective fell from about eight thousand, nine hundred and fifty to three thousand, one hundred and sixty eight. Why did this hour become worse after the early updates? Each calendar coefficient is shared across many hours. A change can help enough other rows to lower their combined objective while hurting this one. Full batch optimization does not promise a steadily shrinking error for every observation. The single hour helps us read the calculation; all training rows determine the fitted compromise.

## Separate step safety from convergence

A training only curvature check guards the fixed step size. Append the intercept column and calculate the largest eigenvalue of the averaged design Gram matrix. For this quadratic objective, a positive rate below two divided by that curvature contracts positive curvature directions. The limit is about zero point nine four, so zero point one is inside it. Safety does not say how quickly the tolerance will be reached. We measure that separately through saved history. Changing validation outcomes in a fixture leaves the model and stopping record unchanged while the evaluation score changes. This is one checked full batch reference. Optimizer and schedule comparisons remain later work.
