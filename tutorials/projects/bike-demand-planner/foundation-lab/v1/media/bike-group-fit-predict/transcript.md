# Fit once, then predict from calendar keys

## Read the key from the fixed schema

The prepared row still has the same forty five ordered calendar values. We have not rebuilt the dataset for this model. The helper checks that one of the first twenty four hour positions is active. It also reads the named working day flag. For the Monday eight o'clock row, those selected values form eight, one. The model does not use month or the separate weekday columns. Its receipt records the twenty four encoded hour columns and the working day column as used inputs. The target stays separate. Extracting this key interprets a fixed schema; it does not learn a transform from validation data.

## Collect outcomes by training key

Fitting first calculates the global training mean for the fallback. The loop then visits paired training keys and targets. Strict pairing prevents silently dropping an unmatched value. Each key is validated before its target is appended to that group's list. For eight, one, the completed list contains two hundred and forty eight observed targets. Their sum is eighty six thousand, six hundred and seventy seven. No validation rows enter this loop. A group retains its mean and its support count, while the model retains the fallback and its full training count. These saved statistics are enough to repeat the prediction later.

## Solve squared error inside each group

Why use a mean inside each group? Take the same half squared error used for the global constant, now restricted to one group's training rows. Its derivative is the chosen constant minus that group's average target. Setting the derivative to zero gives the group mean. We solve this separately for every observed group. These values are fitted parameters, even though we obtain them by arithmetic rather than gradient descent. More groups permit a richer function. This objective is squared error. A reporting preference for mean absolute error does not make the fitted means into medians or guarantee the best validation score.

## Prediction selects saved values

Prediction builds a lookup from the saved group statistics. It validates each requested key and returns its fitted value, using the saved global fallback when the valid key is absent. For eight, one, it returns three hundred and forty nine point five zero. The method never receives the later target of four hundred and twenty. Repeating a call does not change a mean or increment a count. The runner attaches each result to its record, target time and issue time before scoring. The parameters stay frozen throughout validation. Updating them from validation answers would define a different evaluation protocol.

## Check the information boundary

A small integration fixture tests that separation directly. Its training data stays fixed while validation targets change from sixteen to one thousand. Both runs produce identical fitted group statistics and the same fallback. Their validation mean absolute errors change from nine to nine hundred and seventy five. The outcome change affects scoring, where it belongs. The fixture also runs after its raw file and test export have been removed. Thirty eight code checks pass, and the real predictions have separate arithmetic and identity verification. These checks establish implementation behavior. The learner's explanation and the later modelling handoff remain separate evidence.
