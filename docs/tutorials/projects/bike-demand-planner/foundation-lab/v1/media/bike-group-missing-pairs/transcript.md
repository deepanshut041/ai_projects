# A training-only fallback for an absent group

## A valid key can have no training rows

A small training sample may have working day mornings and no nonworking day at two. The pair two, zero is valid, but its group mean is undefined. Our declared fallback is the global mean of recorded training targets. We choose this before evaluation and save it with the model. Validation answers never fill the missing group. An invalid request is different. Hour twenty four or a working day value of two must raise an error. Treating malformed input as a missing group would conceal a problem in the request or schema. A fallback handles limited training coverage, not arbitrary invalid values.

## Use a fixture we can calculate by hand

Here is the complete training fixture. The pair eight, one has targets ten and thirty. Its group mean is twenty and its support is two. The pair two, one has one target of two. The pair eight, zero has one target of one hundred. Those groups have means two and one hundred, each supported by one row. We have four recorded training rows across three observed groups. The pair two, zero is absent. Keeping the counts matters because a group with two rows represents twice as many recorded hours as a one row group when we calculate the global fallback.

## Average rows for the global fallback

Add the four training targets. Ten plus thirty plus two plus one hundred gives one hundred and forty two. Divide by the four recorded training rows to obtain thirty five point five. Averaging the three group means equally would instead give about forty point six seven. That assigns equal weight to groups with different support. Our fallback estimates the level across recorded hours, so the original row weights must be preserved. A different fallback hierarchy, such as an hour only mean, would be a different modelling policy to declare and compare. It is not introduced silently when a difficult request appears.

## Follow an observed and an absent lookup

The executable example fits those four training rows. Requesting eight, one returns the observed group's mean of twenty. Requesting two, zero returns the global fallback of thirty five point five. Change the first requested key to eight, zero and its answer becomes one hundred, because that group exists. The absent key still returns thirty five point five. Neither call creates a new group or uses an outcome. Repeating the missing lookup leaves the model unchanged. The test also changes the original input lists after fitting and verifies that the saved model is unaffected. Its state contains the fitted statistics, not references to mutable training lists.

## Read real coverage separately from the fixture

The real training year contains all forty eight valid hour and working day pairs. The model therefore uses no fallback for its actual training or validation predictions. Those zero counts are saved in the run receipt. The real score establishes the performance of observed group lookup. The small fixture establishes what happens for a valid absent group. A branch never used by a successful dataset still needs a meaningful known answer check. If we later made the groups more specific, coverage and support would need fresh inspection. That would be another model comparison, while this baseline keeps its original forty eight pair definition.
