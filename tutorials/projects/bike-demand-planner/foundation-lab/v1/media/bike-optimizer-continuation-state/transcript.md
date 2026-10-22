# Save what determines the next update

## Pause in the middle of a real batch stream

Saving weights is enough for only part of the problem. Alongside the scaler and feature order, weights reproduce a forward prediction. They do not necessarily reproduce the next training update. Consider a four row fixture with batches of three. After one update, one row remains in the current epoch. The next batch must contain that exact row, and its mean gradient has denominator one. Starting a new permutation now would skip the intended remainder. Momentum also needs the accumulated velocity. Adam needs both moments and its update counter. We use this small mid epoch pause to test continuation precisely, without claiming a hosted or large training recovery has been demonstrated.

## Save the coupled pieces together

The continuation stores parameters and optimizer configuration. The first and second buffers preserve the history that determines the next direction, and the counter preserves Adam bias correction. The stream saves its current permutation, cursor and epoch, plus generator state for future permutations. An epoch seed alone is insufficient if its generator has already advanced. Examples seen and schedule settings determine the next rate. Source, lock, data, schema, feature order and scaler identities establish which calculation this state belongs to. Outer run records retain those identities, while the low level continuation validates its configuration and dimensions. A valid paused state is explicitly distinguished from a failed numerical trial.

## Compare resumed and uninterrupted execution

The fixture first pauses after one update. It serializes and reloads the state through JSON, exercising the saved representation rather than passing a live object in memory. Resumed training continues to the original example budget. A separate uninterrupted run uses the same configuration and seed. The check compares exact parameters, moments, counters, batch order, generator state and schedule progress for all three update rules. For this small same environment fixture, they agree exactly. Wall clock duration is recorded separately and naturally differs between executions. These checks establish a local continuation contract at a mid epoch boundary. They do not imply a remote runtime, device change or general training platform has passed recovery.

## Resetting Adam changes the second step

The earlier scalar Adam example shows why this is more than bookkeeping. After gradients plus two then minus one, the retained first moment remains positive at the second update. Continuing with that history takes the parameter from approximately negative zero point one to negative zero point one two six six. Resetting both buffers and calling the next step t equals one would respond to the latest negative gradient as a fresh start. It would take a different direction. Likewise, restarting a decay schedule at zero examples would enlarge the next rate, and regenerating the stream would change its rows. A plausible looking training curve cannot reveal every such reset. Exact continuation checks make those hidden dependencies observable.

## Keep successful continuation distinct from failure

A failed numerical candidate remains useful evidence about the chosen rate, but it is not a successful continuation checkpoint. The saved state marks whether it is resumable, and loading failed state is rejected. The run also retains why it stopped: example budget, update budget, time limit, pause, divergence or nonfinite calculation. The implementation now has sixty passing source checks, independent metric and identity audits, and source and wheel verification. These establish checked behavior. They do not establish learner understanding, final test performance or hosted recovery. The private notes contain the complete calculations and result comparisons. Before the next objective, explain which saved item determines the next row, next learning rate and next Adam correction. That connects the code to the stateful model being learned.
