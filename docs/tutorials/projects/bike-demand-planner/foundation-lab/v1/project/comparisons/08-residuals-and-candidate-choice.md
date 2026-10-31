# Comparisons: Read hourly errors and freeze a supported candidate

The review consumes the saved practical-model study without refitting. Its seven representative model/output-policy pairs retain the previous choices and feature/task identities. A representative artifact's error is distinct from its configuration's earlier mean over training seeds.

| Comparison | What the actual results show | What would go too far |
| --- | --- | --- |
| Signed versus absolute error | −72.3422 signed error and 91.5729 MAE summarize direction and magnitude differently. | Treat cancellation as accuracy. |
| Overall versus working-day rush | 750 rush rows have MAE 178.1954; the other 3607 have 73.5616. Counts reconstruct 91.5729. | Average slice means without their denominators. |
| Months/seasons | January MAE 39.6133 rises to June 142.9952; autumn has no rows. | Report zero autumn error or a full summer result. |
| Seed versus temporal variation | Forest seed variation holds these dates fixed; consecutive-hour residual correlation is 0.8677. | Use fitting-seed SD as an interval for future dates. |
| Calendar-matched historical levels | Reweighted 2011 means 121.3077 versus later observed 215.2001, difference 93.8925. | Attribute that difference causally or rank the diagnostic as a new candidate. |
| Valid versus unsupported changes | January/June Monday 08:00 both return 349.5040; invalid schema/order/date requests are rejected. | Treat month insensitivity as malformed input, or fallback as permission to invent fields. |

Rush hours are 07–09 and 16–18. Working-day rush additionally requires the recorded day-type flag. Month bins retain the twelve-category domain, with only Jan–Jun observed in validation; seasonal labels retain their actual observed-month coverage. Target-defined demand bands are retrospective only. Daily/weekly groups preserve counts and gaps, and no IID interval is reported.

The tradeoff relevant to model choice is concrete: groups has overall MAE 91.5729 and June MAE 142.9952; the retained forest has 99.8324 overall and 90.8782 in June. Groups also has lower working-day rush MAE, 178.1954 versus 207.8724. The previously declared objective is overall hourly MAE, not a new June-only or operational-cost objective. Keep groups/raw and record the weakness; do not silently retune a slice-routed model or move the split.

Four accepted sensitivity cases and fourteen rejected probes exercise the historical replay contract. The April holiday miss was identified from the error ranking after exploration; it illustrates a calendar-group failure and changes neither the source flag nor the fitted model. The candidate's model copy remains byte-identical to the original selection, with a saved hash, schema, task, fallback and limits.

The [worked notes](lessons/08-residuals-and-candidate-choice.md) retain the calculations, plots and consequential source. Error diagnostics and candidate configuration finish this step. Lessons 9 and 10 then implement saved serving, recovery and final assessment. Learner understanding remains a separate question.

Return to the [worked understanding notes](lessons/08-residuals-and-candidate-choice.md).
