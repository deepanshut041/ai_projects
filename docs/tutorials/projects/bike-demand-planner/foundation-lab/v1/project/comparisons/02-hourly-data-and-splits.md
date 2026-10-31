# Comparisons: Understand the hourly data, calendar features and time split

Handoff update, 18 October 2026: this page retains the data-preparation explanation. Its implementation is committed as `ad1b870` after 22 checks passed. Current predictions and results continue in the [baseline record](project/implementation.md).

The purpose of this page is to make the choices easier to reason about when returning to the project. The code already uses one recorded set of decisions. The alternatives below help explain what those decisions change; they are not benchmark results or extra implementations that have already been completed.

Keep the Monday example in mind: at 07:00, predict rentals for 08:00–09:00 using calendar information. The eventual source count is 154. That same forecast question is the reference point for each comparison.

## Choosing how to represent the hour

A raw hour value gives a linear model one hour coefficient. Its contribution at 08:00 is `8 * hour_weight`, and moving to 09:00 changes it by exactly `hour_weight`. Every other one-hour move has the same change. This is compact, but it imposes one straight-line progression across the day.

With one-hot hour encoding, 08:00 activates the hour-8 column and 09:00 activates the hour-9 column. Their contributions can differ independently. The change in a later prediction would be `weight_for_hour_09 - weight_for_hour_08`. That freedom is useful for representing peaks and dips, although it does not tell the model that neighbouring hours should be similar.

A sine/cosine pair offers another representation. It maps the hour around a circle, keeping 23:00 close to 00:00 with only two values. A linear combination of one pair has a constrained smooth daily shape; more complex patterns may need extra terms or a different model.

| Hour representation | Columns for hour alone | What it asks a linear model to learn |
| --- | ---: | --- |
| Raw number | 1 | One fixed change per clock-hour increase |
| Full one-hot group | 24 | A separate coefficient for each hour |
| Sine/cosine pair | 2 | A smooth daily pattern built from one circular pair |

Our prepared files use one-hot hour, weekday and month plus two flags, for 45 values in total. It is an inspectable first representation. It does not automatically add hour-by-working-day interactions, so a weekday morning peak and a weekend profile may still need more expressive terms later. A fair later comparison will keep observations, time partitions and evaluation rules fixed while changing the representation deliberately.

## Keeping all indicators or choosing a reference category

In our full hour group, exactly one of the 24 indicators is one. With an intercept, adding the same amount to every hour coefficient and subtracting it from the intercept leaves predictions unchanged. The coefficients therefore need not be uniquely identifiable.

Dropping one category from each group gives a reference-category representation. Hour effects would then be expressed relative to the omitted hour, and the total vector would have 42 rather than 45 values. This removes the obvious full-group dependencies, although other calendar flags can still overlap with the remaining columns. It does not make an individual coefficient causal.

We have retained the full groups so the initial mapping is explicit. Reference coding and regularization are alternatives to compare when fitting a model; neither is an unrecorded change to the current dataset.

## Preserving a source code or choosing a model feature

The parser checks that hour, weekday, month, year and season have valid source codes. That protects the meaning of the record. Choosing which fields and encodings the model receives is a separate decision.

Year illustrates the difference. It is known in advance, but its raw code is zero throughout the 2011 training period. The earlier gradient formula gives `residual * 0`, so a raw year coefficient cannot learn a change between years from those examples. We omit it from the first schema. Season is also left out while month is retained; this does not prove that season adds no information, and the source documents' season-name disagreement remains recorded.

Fixed integer codes in an audited row should therefore not be mistaken for an already approved continuous feature representation. The selected fields and their ordered output columns are saved explicitly.

## Using a field for an audit or using it for a forecast

The Monday row contains one casual rental and 153 registered rentals. Checking that `1 + 153 = 154` is a useful audit of its total. Feeding those components to a model for the same hour would instead reveal the answer. The values can be correct and useful while still being unsuitable forecast inputs.

Observed weather raises a related timing question. A measurement during the target hour describes what happened; an advance weather forecast describes what was expected before it happened. Our calendar-only task excludes the observed fields. A historical weather-conditioned task could be studied separately, but its scores would need to name the different information available to it.

## Finding an empty cell or finding an absent hour

An empty cell is a missing value inside an existing row. An absent hour has no row at all. The current file has complete required cells, yet it leaves 165 slots unrecorded against its naive hourly grid.

The jump from 04:00 to 06:00 on 2 January leaves 05:00 unrecorded. Inserting a row with target zero would assert that no rentals occurred. Leaving the gap visible makes fewer claims and preserves the observations we actually have. Any later imputation policy would need its own justification and a clear account of what it changes.

The two deliberately excluded midnight rows are different again: they exist in the file, but their forecast timing makes them unsuitable for the chosen evaluation boundary. They remain in the membership record. Missing slots and boundary exclusions should not be added together as though they were one kind of missing data.

## Mixing hours randomly or evaluating later periods

A random split can be useful for questions where observations are interchangeable. Here the intended use points forward in time. Training on later hours while evaluating earlier ones would give the fitting process a different history from the one available in the forecast scenario.

The current protocol uses 2011 for training, January–June 2012 for validation choices and July–December for reserved final evaluation. The boundaries were set without using target distributions or model errors to improve a score. They are frozen so a later representation or optimizer comparison uses the same task.

Chronology alone still needs the availability check below. Simply sorting the target start times is not sufficient for an advance forecast.

## Ordering targets or checking when a forecast can actually be issued

The last training interval starts on 31 December at 23:00. Its completed outcome is available at midnight under our interval-end rule. A forecast for a target starting at midnight would have to be issued at 23:00, before that final training outcome was available.

We leave that midnight target out and start validation with the 01:00 target, issued at 00:00. The July boundary follows the same reasoning after validation-based selection. The result is 8,645 training observations, 4,357 validation observations, 4,375 test observations and two boundary exclusions.

This rule uses timestamps rather than a fixed number of preceding rows, because missing hours make row distance different from clock-time distance. It assumes naive interval-start labels, interval-end outcomes and zero reporting/model-build delay. The source's timezone and daylight-saving details remain unspecified. The [worked timeline](lessons/02-hourly-data-and-splits.md#let-the-time-split-follow-the-forecasting-question) follows the exact January example.

## Applying a fixed definition or fitting a transform

The possible hours and months are known calendar domains. Our encoder uses those definitions and fits no statistics. All outputs are binary, so there is no fitted scaler. A training-only feature summary describes constant columns; it does not learn a transformation or remove columns using future observations.

A learned scaler or feature selector has a different dependency: its parameters come from data. It must fit only on training, then apply unchanged to later partitions. A later time-based fold repeats that rule within its own training portion. The held-out-change fixture checks the behaviour we have actually implemented: changing later flags and outcomes leaves training exports, training feature counts, schema and membership unchanged.

## Reusing known files or silently accepting different data

A dataset name alone does not tell us whether two runs used the same bytes. A changed source file, reordered feature vector or altered membership could change a model result even when its code and parameters stayed the same.

The saved hashes and manifest make those changes visible. Reusing the same source reproduces the prepared files; a conflicting frozen file raises an error. These are locally computed identities rather than official signed checksums. The [reference section](lessons/02-hourly-data-and-splits.md#reference-source-details-and-saved-evidence) keeps the exact values and source-document differences out of the main learning flow while preserving them for inspection.

No alternative on this page has been declared a benchmark winner. The current choices give the next baseline a fixed dataset and a clear prediction question; measured comparisons come after that handoff.

Return to the [worked understanding notes](lessons/02-hourly-data-and-splits.md).
