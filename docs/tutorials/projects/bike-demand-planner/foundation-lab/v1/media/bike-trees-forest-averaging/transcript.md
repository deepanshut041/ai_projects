# Average varied trees without assuming their shared errors disappear

## Fit varied historical views, then average

Our random forest contains sixty four trees. Each is fitted from a bootstrap sample of the training rows, with replacement. A draw has the original row count but can repeat some observations and omit others. At each split, the forest considers seventy percent of the feature columns, adding another source of variation. These mechanisms produce different fitted trees whose predictions can be averaged. All views come from twenty eleven training data. Validation outcomes do not become bootstrap samples or help construct the trees.

## Make the forest choices explicit

The library constructor exposes tree count, bootstrap policy, feature fraction and one fitting job. Depth and leaf support come from the declared candidate. We compare depth eight with support ten, depth sixteen with support five, and the deeper leaf capped setting with support one. Tree count stays sixty four across these choices. This is an explicit capacity sweep within one forest setup, rather than changing every setting after looking at its score. Numerical snapshots retain the component tests and values so the ensemble result can be independently replayed.

## Use all component outputs for the real busy hour

At the real later Monday eight oclock request, the first three trees return one hundred and eighty, one hundred and ninety point three, and one hundred and eighty seven point five. Those numbers explain components, not the finished ensemble. All sixty four saved leaf predictions sum to about eleven thousand, four hundred and thirty six point four eight. Dividing by sixty four gives one hundred and seventy eight point seven. That is the actual forest forecast, still well below the target of four hundred and twenty. The aggregation rule itself contains no new observation of future demand.

## Read bootstrap support in the correct units

A component leaf can include several copies of the same original observation. Its weighted sample count therefore differs from the number of distinct rows. Its fitted mean follows the bootstrap weighting, so counting unique rows alone can give the wrong arithmetic. This differs from the plain tree example, whose eight training rows were unweighted. We preserve both support counts in the numerical state. The learner can inspect what supported a component without silently interpreting repeated samples as additional independent historical hours.

## More trees cannot remove a perfectly shared miss

Consider a changed case where every component predicts one hundred and eighty seven for the busy request. Averaging three copies or sixty four copies still gives one hundred and eighty seven. Varied trees can reduce sensitivity to particular samples or splits. They cannot guarantee removal of errors they share. In the executed sweep, deeper forests improve mean validation error from about one hundred and sixteen point six to ninety nine point seven five. That is slightly better than one deep tree, while the grouped baseline remains better at ninety one point five seven. More machinery has not earned selection here.

## Keep configuration means separate from one artifact

The deepest forests ninety nine point seven five score is a mean over seeds seventeen, twenty nine and forty three. The retained seed seventeen artifact scores ninety nine point eight three. It was declared as the representative before seeing quality, rather than chosen as a lucky seed. The small reported standard deviation describes algorithm variation on these same hours, not uncertainty over future dates. That representative snapshot contains about sixteen point one megabytes of numerical JSON and one hundred and thirty one thousand nodes. We compare this cost with the measured quality before deciding whether to carry it forward.
