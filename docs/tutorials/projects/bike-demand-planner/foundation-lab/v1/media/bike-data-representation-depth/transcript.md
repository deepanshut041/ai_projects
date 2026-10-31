# What the calendar representation assumes

Year is known in advance, so omitting it is a modelling decision rather than a leakage restriction. Every training observation belongs to twenty eleven, where the raw year flag is zero. A linear weight gradient equals residual times feature value.

Multiplying by that zero feature gives zero for every training example. The data therefore cannot identify a year to year effect through this column. A year flag of one in twenty twelve would use an unsupported coefficient.

We leave it out of the first schema. We retain month and omit the coarser season code from this first representation. The source documents disagree about some season names, so we retain the codes without assigning uncertain names.

We have not tested whether including season would improve a predictor. A later feature comparison can ask that question using the same frozen validation period. Source documentation also differs on some weather normalization details.

We validate retained numeric ranges without inventing disputed temperature conversions. Exactly one hour indicator is active for every row. Add two to every hour coefficient, and subtract two from the bias.

The prediction stays unchanged, because one added two cancels the subtracted two. Full category groups and the intercept therefore contain dependencies. Some flags overlap with calendar groups as well.

A model can have useful predictions while individual coefficient values are not unique. We should be careful about treating one coefficient as a unique effect or a causal explanation. Dropping one reference indicator from each of the three category groups removes three columns.

Forty five becomes forty two, with the two flags retained. Coefficients then describe differences relative to the omitted categories for those groups. Other flag dependencies can still remain.

Regularization is another later modelling choice. Neither change has been silently added here. Our first schema keeps all categories visible, records the dependencies and leaves later comparisons to measure their consequences.

The hour domain, zero through twenty three, comes from the calendar. We do not discover it by reading future observations. This fixed encoder has no fitted statistics.

Its outputs are already zero or one, and no scaler is fitted. If we later learn a mean, scale or selected feature set, that learning belongs to training only. Validation applies the saved transform.

A chronological cross validation fold needs its own training fit, following the same boundary inside every fold. A valid calendar category can be missing from a small training sample. Its schema position still exists, although the model has no supporting examples for its effect.

An impossible hour, twenty four, is different and must fail validation. Missing fields, extra fields such as the target, and noninteger codes also fail. The working day flag must agree with weekday and holiday.

Training only feature counts describe coverage. Constant columns keep their positions, and we never divide by a zero standard deviation. Circular coordinates make a clock relationship numerical.

At hour zero, sine and cosine are zero and one. At hour twenty three, they are approximately minus zero point two five eight eight and zero point nine six five nine. Their Euclidean distance is about zero point two six one one, while raw hour numbers differ by twenty three.

That encodes closeness across midnight. A linear predictor over one sine and cosine pair still has a smooth single harmonic shape, which is a modelling assumption rather than evidence of better forecasts. Full one hot encoding with an intercept has another subtle property.

Every row activates exactly one hour indicator. Add five to all twenty four hour coefficients and subtract five from the intercept. The active hour contributes five more and the intercept five less, so every prediction stays the same.

Ten plus twenty becomes five plus twenty five. Both totals are thirty before the other shared terms. A coefficient can therefore lack a unique isolated interpretation even while the prediction is well defined.

Does that dependence mean the dataset cannot be used for prediction? Pause and distinguish predictions from parameter interpretation. Equivalent coefficient sets can produce the same prediction function.

Dropping a reference category or applying regularization can select a representation or solution, but those are explicit later choices. Do not call an isolated hourly weight a unique causal effect. Keep the schema fixed and evaluate choices with the same observations before claiming that one is better.
