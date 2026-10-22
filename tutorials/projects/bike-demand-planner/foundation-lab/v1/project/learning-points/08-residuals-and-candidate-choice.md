# Learning points: Read hourly errors and freeze a supported candidate

The forecast and its observed outcome belong to different moments. At the real Monday commute hour, the saved group model predicts 349.5040 rentals from training history; the later count of 420 creates an error of −70.4960. At the quiet hour on the same date, prediction 7.5146 against target 5 gives +2.5146. Keeping the sign explains the direction of a miss. Taking its absolute value explains its size. We need both views because overestimates and underestimates can cancel in a signed average without becoming accurate forecasts.

On the complete validation population, mean signed error is −72.3422, while MAE is 91.5729. The difference is understandable through total deficit and excess. Average deficit over all hours is 81.9576 and average excess is 9.6154. Their sum gives MAE; excess minus deficit gives signed error. This connects the formulas to what actually happened rather than leaving them as unrelated metric names.

## The denominator is part of the question

An underprediction average can use every evaluated hour or only the hours that were underpredicted. Here 3,143 of the 4,357 hours have a deficit. Dividing total deficit by all rows gives 81.9576; dividing by the underpredicted rows gives 113.6141. Neither is a correction of the other. They describe the frequency-and-size burden across the full population and the typical magnitude conditional on a deficit. Their names and counts must travel with the numbers.

The same principle matters for slices. The 750 working-day rush hours have MAE 178.1954, while the other 3,607 hours have MAE 73.5616. Their overall MAE is a count-weighted combination, 91.5729. Averaging the two slice means would give a small population the same influence as a much larger one. Our implementation retains the error sums and checks that every exclusive slice dimension reconstructs the full population. For RMSE, the quantity that adds is squared error before taking the square root.

Working-day rush is also more specific than a rush clock window. We define six clock hours, and only some of those observations occur on working days. The labels are not interchangeable: there are 1,092 rush-clock rows but 750 working-day rush rows. An engineer reading a result should be able to reconstruct the membership rule and denominator, rather than infer them from a familiar-sounding heading.

A zero-count slice has no score. Autumn is absent from this Jan–Jun validation window, so its metrics are null, not zero. Summer means June only here; it does not describe July and August. Similarly, observed-demand bands are useful retrospective views of failure, but the observed target is unavailable when the forecast is issued. Choosing a predictor using that future band would leak the answer into the request.

## Errors can stay related across time

The chronological figures show a broader problem than occasional isolated misses. Monthly MAE rises from about 39.61 in January to 143.00 in June, and signed error becomes strongly negative. The group model returns one fixed value per hour/day-type pair, so it has no mechanism to follow a changing level within that pair. The plot reveals the pattern; it does not identify a single cause.

Neighboring residuals also move together. Their descriptive correlation is 0.8677 across 4,346 pairs whose timestamps are exactly one hour apart. The ten unrecorded slots are not replaced with zeros, and a gap is not silently treated as a one-hour lag. Weekly summaries keep their actual counts, including partial boundary weeks. This makes the temporal structure visible without pretending that every row is an independent repeat of the same situation.

The earlier forest seed variation answers a different question. Changing a fitting seed while holding the same validation period fixed describes variation of that training procedure. It does not create new months, new demand conditions or independent future outcomes. Its small standard deviation cannot serve as a confidence interval for future hourly mistakes. This review reports time structure descriptively and leaves calibrated uncertainty unclaimed.

## Matching calendar cells narrows a question without proving a cause

The selected group model ignores month, which makes a seasonal composition explanation plausible. We therefore perform a separate diagnostic: compare old and later observations with matching month, hour and working-day status. Historical cell means are calculated from Jan–Jun 2011 and weighted to the actual 2012 validation membership. Every validation row has a shared cell, so no unmatched outcomes are hidden.

Those reweighted historical means average 121.3077 rentals/hour; the later observed mean is 215.2001. A substantial difference, 93.8925, remains after matching these calendar variables. This makes a simple explanation based only on different calendar composition inadequate. It does not tell us how much comes from demand growth, weather, service changes or other unmeasured factors. The comparison is observational and exploratory, not a causal estimate.

It is also not another ranked predictor. Its purpose is to understand the failure while examining outcomes. The selected forecast mean is 142.8580, a different calculation using its original broader groups and full 2011 training history. Keeping the diagnostic separate prevents us from quietly changing the model and still presenting the result as an unchanged comparison.

## A valid request can reveal a model limitation

Moving a Monday 08:00 request from January to June leaves the group forecast at 349.5040. The request is valid; the selected function simply does not use month. Moving instead to a valid Sunday 08:00 request changes the day-type key and gives 76.4348. These are supported sensitivity results. They explain the stored function's behavior without establishing a causal effect of one calendar field.

The April holiday example is particularly useful. The source marks a real Monday at 08:00 as nonworking, so the same low group prediction is returned against an observed count of 597. A calendar label can be correct while its broad demand group is a poor predictor for that hour. We should not rewrite the flag after seeing the target. That would confuse a legitimate model failure with input repair.

Malformed or unsupported requests are different. Missing fields, nonfinite values, unknown categories or schemas, reordered model features, inconsistent clocks and out-of-domain dates are rejected before lookup. A training-mean fallback is reserved for a valid but absent fitted group; it is not a way to guess missing inputs. Named dictionary ordering is harmless because the encoder chooses the canonical positions. A mismatched ordered model schema can change what the numbers mean and must be rejected.

The candidate's domain is historical validation replay through June 2012, with exact naive hours and a one-hour lead. The original source does not establish timezone/DST semantics. Holiday metadata requires a trusted calendar; structural validation cannot establish its real-world authority. A valid estimate also does not create an observed label for an unrecorded hour. These are practical boundaries of what the application may later claim.

## A candidate needs its limits as well as its weights

Groups remains the lowest-error representative overall and on the defined working-day rush slice. The forest and tree do better in June. That is a meaningful tradeoff, so the notes show it beside the overall result rather than making one model appear universally best. The current objective weights validation hours equally; no separate operational cost or June-only objective has been agreed. This review therefore retains the original selection without tuning a new routing rule from the slices.

```mermaid
flowchart LR
    P["Unchanged saved predictions"] --> D["Counts, errors and supported-domain review"]
    D --> C["Groups/raw candidate with exact model identity"]
    C --> L["Carry limits into later export and app work"]
```

The arrows describe a handoff of evidence and behavior, not a new training loop. The candidate records the model's exact bytes/hash, feature schema/order, source/data/protocol identities, raw output, absent-group fallback, domain and known weaknesses. Copying a model and writing that configuration does not prove a serving API, fresh-process recovery, final-test performance or application usefulness.

All 96 checks and the independent audit establish the numerical and input behavior. They do not supply the learner's explanation. Return to the [worked notes](lessons/08-residuals-and-candidate-choice.md) to trace the code and figures, then discuss why a model can win the agreed overall metric while still needing an explicit account of its June and rush-hour failures.

Return to the [worked understanding notes](lessons/08-residuals-and-candidate-choice.md).
