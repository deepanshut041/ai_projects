# Choose a development reference using quality, cost and honest evidence

## Keep the hourly prediction question fixed

Every family answers the same hourly calendar only question. Training uses eight thousand, six hundred and forty five hours in twenty eleven, followed by the same four thousand, three hundred and fifty seven validation hours. No weather, target transform or demand history is introduced. The final test remains reserved. Different fitting algorithms do different work, so pretending a tree split and an SGD update are equal operations would not create a fair control. We instead expose each familys structural limits, search allowance, stopping rule and actual costs.

## Freeze the search before opening validation

Each tree family receives three configurations and the same three declared seed values, giving nine main fits. Across families that is twenty seven fits, plus four deterministic references with no newly tuned settings. A small training only timing pilot checks feasibility for each family before the full sweep. These pilots choose no quality setting. Every fitted main state is saved and hashed before validation is opened. The configured timing guards run after a fit returns; structural caps bound the actual model. We do not describe those checks as interruptible hard timeouts.

## Select configurations, not lucky seeds

For one configuration and output policy, take every declared seeds validation MAE and average those scores. This is an average of quality measurements, not an ensemble of the three seed models. Exact ties prefer raw output and then the declared configuration order. The first seed supplies a representative artifact by prior declaration. We never pick the seed with the most flattering score. The result record keeps both the configuration mean and that representatives score so the selected artifact and reported evidence can be understood together.

## Keep the stronger existing reference

The grouped mean remains best at ninety one point five seven mean absolute error. The best tree is near one hundred and the best forest near ninety nine point seven five. Clipped interaction linear regression is near one hundred point five two, and clipped boosting near one hundred and two point two two. Beating raw additive linear regression is not enough when a stronger simple reference already exists. We therefore retain groups with raw policy. Its outputs are already nonnegative, so clipping ties. Keeping that model is the result of the completed comparison, not evidence that the implementation was abandoned.

## Name what the resource measurements actually measure

The representative deep forest fits locally in about zero point six eight seconds and saves about sixteen point one megabytes of JSON. Replaying the validation batch takes roughly one tenth of a second through our inspectable numerical traversal. The three hundred stage booster fits in about zero point eight two seconds and saves about six hundred and sixty four kilobytes. These observations describe this machine and this implementation. JSON bytes measure storage rather than peak memory, and replay is not optimized library serving latency. We compare costs honestly without manufacturing a universal speed ranking.

## Carry evidence into the next learning question

Eighty three checks and an independent audit verify the saved predictions, scalar metrics, leaf membership, identities and fitted state replay. All written explanations were completed before creating these companions. These checks establish implemented behavior, while the learners own explanation remains a separate conversation. The same validation period has informed multiple development choices, so its selected score can be optimistic. The final test and application remain pending. The useful next question is why a more detailed historical partition predicted the busy Monday worse than a broader group. That interpretation is the purpose of this practical comparison.
