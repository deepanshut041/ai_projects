# Change the batch while keeping the time boundary intact

## Shuffle work inside frozen training membership

A chronological split fixes which observations may train the model. It does not require this independent row calendar model to visit training rows chronologically. We shuffle only the eight thousand, six hundred and forty five training positions from twenty eleven. Validation remains in its later frozen window. There is no recurrent state or lag window here whose internal sequence would be broken. An explicit PCG64 seed generates one permutation per epoch, and consecutive batches take slices without replacement. Each recorded row appears once in a completed pass. A different seed changes optimization order, while the same seed reproduces it. Neither operation moves a later validation outcome into fitting.

## Keep the tail and its exact position

The stream creates a new permutation only when the current one is exhausted. It then takes the next slice, bounded by remaining rows and the example budget. This means the final batch may be smaller than the requested size. At size thirty two, eight thousand, six hundred and forty five equals two hundred and seventy full batches plus five rows. Those five observations are retained as a final batch, making two hundred and seventy one updates per pass. The cursor moves by the actual count. Saving the cursor without the current permutation would lose the identity of the next rows. The generator state determines the following permutation, so both current order and future random state belong in a continuation.

## Average the rows that actually arrived

The objective is a mean over the actual batch. For weight gradients we multiply the transposed batch by its residuals and divide by its row count. The bias uses the residual mean. The tail has five rows, so its denominator is five. Dividing by thirty two would shrink that intended mean gradient to five thirty seconds of its value. Dropping the tail would discard five recorded hours each epoch instead. We do neither. A small constant feature fixture checks the expected update with independent scalar arithmetic, including the one row tail case. This detail matters most when comparing different batch sizes: a hidden tail scaling change would introduce a second update policy.

## Equal exposure produces different update counts

We choose ten complete training passes as the main work allowance. Every trial visits eighty six thousand, four hundred and fifty examples. A one row method therefore performs the same number of parameter updates. Batch thirty two performs two thousand, seven hundred and ten, including tails. Full batch performs only ten. The earlier proposed ten thousand update cap would truncate the one row trial. Before the main study, we raise that safety cap to one hundred thousand while retaining the five minute limit. All main trials finish at their example budget. Equal update counts would give very different data exposure. Equal wall time would mix algorithm behavior with hardware costs. State which comparison you intend before interpreting its winner.

## Read the batch result with its work allowance

Keeping SGD and rate zero point zero one fixed, mean validation absolute error is one hundred and eighteen point zero nine for single rows, one hundred and sixteen point one seven for batches of thirty two, and one hundred and forty six point nine four for full batch. These are averages across three paired seeds. Single row updates also vary more, with sample deviation six point zero four versus zero point five six. Full batch has only ten small steps and remains far from the training optimum. Our earlier two hundred step, higher rate reference did reach it. Local fitting times describe this machine and loop, not a general benchmark. The batch comparison exposes noise, work and progress together; it does not make full batch incapable of fitting the model.
