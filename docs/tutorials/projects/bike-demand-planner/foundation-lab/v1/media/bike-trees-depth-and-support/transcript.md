# Let the tree grow and inspect what its smaller leaves remember

## A path can combine calendar conditions

The full tree sees the original forty five binary calendar columns. A path can test an hour and then ask about working day status or month. Its reached leaf is the intersection of those conditions. This lets the tree express conditional relationships without us adding every product feature explicitly. Each extra partition also divides the available training examples into smaller supporting groups. The practical question is whether that extra historical detail improves the later forecasts enough to justify it, rather than whether a larger model can lower training error.

## Declare support and size limits before fitting

The source declares depth, minimum leaf support and a maximum of one thousand and twenty four leaves. The library fits only training rows and targets. Our first setting uses depth three and support twenty. The second uses depth eight and support ten. The third removes the depth limit and uses support one, while retaining the leaf count cap. A sequence of one hot questions can become long, so depth does not simply count the number of calendar variables. These structural controls are the actual fitting limits, with no validation early stopping.

## Read training and later errors separately

Making the bounded tree more flexible lowers training mean absolute error from about ninety four to twenty six. It also improves this depth sweeps later error, from about one hundred and forty one to one hundred. Calling that sweep alone proof of overfitting would be inaccurate. Both measured errors improved. Compare the grouped baseline instead: it fits the training sample less closely, but scores better on later hours at ninety one point five seven. Lower training error is therefore insufficient to choose the function we carry forward.

## Follow the real later Monday at 08:00

Now follow Monday, January ninth, twenty twelve, at eight, whose observed target is four hundred and twenty. The actual deep tree first excludes hour seventeen and eighteen, then takes its hour eight branch. Further tests include working day status, January and weekdays. The reached leaf contains eight training observations. These are real saved conditions and support, not an illustrative tree drawn after seeing the target. The request supplies only the calendar vector. The later count is absent from its route and becomes relevant when we score the returned value.

## Reconstruct the busy leaf from its history

The eight supporting counts are shown here. They sum to one thousand, four hundred and forty six. Dividing by eight gives the stored mean, one hundred and eighty point seven five rentals per hour. Against the later target of four hundred and twenty, the absolute error is two hundred and thirty nine point two five. The broader hour and working day baseline predicts about three hundred and forty nine point five, much closer. Finer conditioning has captured historical detail that does not help this particular later commute. The result does not identify a single cause of changed demand.

## Change the same Monday request to 02:00

Change only the requested hour to two in the morning on that Monday. The actual saved tree follows another path and returns about six point five five. The observed target is five, so this is a useful quiet hour forecast. That success coexists with the busy hour failure and a worse overall score than the group model. We keep both examples and the full validation population, rather than selecting whichever makes the newest model look strongest. This is why support, path, training error and later error all belong in the explanation.
