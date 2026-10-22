# Read the baseline comparison and its failures

## Hold the evaluation question fixed

The comparison entry point creates a fresh global run and an hour working day run. Both fit on the same training year. Both score the same four thousand, three hundred and fifty seven validation hours. Record identity, target time and issue time are checked before any error reduction. Metric definitions and the unrounded prediction policy also stay fixed. Each run saves the model, aligned predictions, source and dependency identities, frozen data identities and completion status. The parent saves their comparison. Only the predictor changes. We do not refit with validation answers, and we do not evaluate the reserved final test.

## Measure the average error improvement

The global mean has validation mean absolute error of about one hundred and fifty four point two five rentals per hour. The group model has about ninety one point five seven. Root mean squared error also falls, from about two hundred and eight point one seven to one hundred and thirty seven point three nine. The group model captures useful daily structure on this window. Its training errors are lower too, which is expected when separate group constants can reproduce the global constant as one possible setting. That training squared error argument does not guarantee a validation improvement. The later result needs measurement, which these saved runs provide.

## Calculate what 40.63 percent means

Subtract the group model's mean absolute error from the global model's value. The reduction is about sixty two point six seven rentals per hour. Divide that reduction by the original global error, then multiply by one hundred. This gives about forty point six three percent. The denominator matters. We are reporting a relative reduction against this reference score. It does not mean forty percent of forecasts became exact, or that every hour improved. It is also a development comparison for these frozen chronological periods. The reserved test and an application's quality requirements still need their own later evidence.

## Inspect a case that gets worse

Consider the eleventh of January at five in the afternoon, issued at four. The recorded target is one hundred and thirty four. The working day group predicts three hundred and ninety four point nine two, producing an absolute error of about two hundred and sixty point nine two. The global constant happens to be much closer, missing by about nine point seven nine. This hour gets worse. Across validation, three thousand, three hundred and twenty three hourly absolute errors improve and one thousand and thirty four worsen. The model knows a typical level for a key, but cannot distinguish all days sharing that key. We have not established why this particular hour was quiet.

## Keep size, frequency and direction separate

There is one more result to interpret carefully. The number of underpredicted hours rises from two thousand, three hundred and ninety one to three thousand, one hundred and forty three. Their mean deficit falls from about two hundred and five point six zero to one hundred and thirteen point six one. Mean signed error remains negative seventy two point three four, slightly more negative than before. Removing large quiet hour overpredictions can improve absolute error without fixing the later period's overall level. Error size, frequency and direction answer different questions. The implementation is ready for review. These results motivate the next modelling discussion, without starting a later phase or claiming learner mastery.
