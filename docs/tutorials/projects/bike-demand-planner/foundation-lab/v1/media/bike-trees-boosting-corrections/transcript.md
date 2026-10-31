# Add successive residual corrections instead of averaging forecasts

## Learn what the current prediction still misses

Boosting begins with a prediction and asks what it still misses. For squared error, the next tree learns observed target minus current prediction. This sign is opposite to the prediction minus target residual used by our scoring functions. Return to the four real training rows, whose initial mean is ninety four point two five. The two busy residuals are eighty four point seven five and one hundred point seven five. Their mean is ninety two point seven five, so the first busy leaf stores a positive correction. A target above the forecast needs an upward move.

## Apply part of the first correction

We apply one tenth of that correction, using learning rate zero point one. Ninety four point two five plus nine point two seven five gives one hundred and three point five two five. The quiet branch has correction minus ninety two point seven five and moves down to eighty four point nine seven five. These are deliberately partial moves. The first boosted prediction therefore differs from the direct stump mean of one hundred and eighty seven. The rate controls the contribution of a correction, while the stage count controls how many successive corrections are fitted.

## Recompute the residual after the prediction moves

After the first stage, the busy forecast is one hundred and three point five two five. The remaining busy residual mean is eighty three point four seven five, not the original ninety two point seven five. Applying one tenth of this new correction gives one hundred and eleven point eight seven two five. This dependence on the running prediction is what makes the fitting sequential. A forest instead averages finished component forecasts. Our known answer checks independently verify both stages, so the distinction is visible in numbers as well as terminology.

## Replay the fitted correction sequence

The snapshot replay starts with the saved initial constant and adds each saved tree value multiplied by the saved learning rate. These local names abbreviate the actual snapshot fields in the source. Prediction performs no residual fitting and receives no outcomes. The library constructed the residual trees during training. Our full benchmark uses squared error, depth three, leaf support ten, rate zero point one and one hundred, two hundred or three hundred stages. Validation early stopping is disabled. The declared stage count, together with the structural limits, is the actual stopping policy.

## Read the actual 300-stage hourly forecasts

The full model starts at the twenty eleven training mean, about one hundred and forty three point seven nine. At the real busy Monday request, its three hundred tree corrections sum to about seven hundred and forty seven point three seven. Applying the rate gives a final forecast of two hundred and eighteen point five three. At two oclock, the same model instead returns minus twenty five point five two. Negative correction leaves can take the total outside the nonnegative target range. Our explicit clipping policy changes that quiet forecast to zero and leaves the positive busy forecast alone.

## More stages improve this grid without winning selection

Increasing the declared stage count lowers training error and improves later error within this grid. Raw mean absolute error reaches about one hundred and four point eight four; clipping reaches one hundred and two point two two. Neither beats the grouped baseline at ninety one point five seven. All three declared seeds give identical scores in this setup. That is an observed lack of variation here, not three independent future periods. A different learning rate or loss would be another comparison. We retain the current settings and results instead of silently enlarging the search after seeing them.
