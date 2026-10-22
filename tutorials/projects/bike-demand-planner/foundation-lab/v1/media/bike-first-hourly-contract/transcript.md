# What is known before the rental hour?

## 0:00 — Which hour are we predicting?

Imagine Monday, October seventeenth, twenty eleven. It is seven in the morning. We want the total rentals during the hour starting at eight. The target interval runs from eight, inclusive, to nine, exclusive. Lead time measures the gap from seven to the interval start. Its duration measures the separate hour we aggregate. The complete count can only be checked after that interval ends. This is an illustrative scenario, rather than a measured bike record.

## 0:34 — Keep the request separate from its answer

At seven, the requested date and hour are already known. We assume this example is a nonholiday working day. The later rental count is still unknown. Casual and registered rentals cannot become inputs either, because they reveal its total. Weather measured during the target hour also arrives later than the request. A historical file can place these values on one row. That physical arrangement does not erase their different availability times. The forecast function therefore receives a calendar description without the later answer.

## 1:15 — Forecast first; calculate a learning error later

The first operation uses the features and the current parameters to produce a forecast. Keep that forecast separate from the target that arrives after the hour. Once the target is available, we can calculate an error and a parameter update. Recomputing the same input after an update helps check the arithmetic. It does not turn the revised number into the forecast originally issued at seven. This order remains necessary when we replay an old file whose outcomes are already visible to us.

## 1:51 — Build the two demonstration inputs

Use one for the working day flag. The requested clock hour is eight, and we choose a fixed divisor of four. Eight divided by four gives two, so the ordered vector is one, two. This scale makes the first calculation easy to inspect. It is a demonstration choice, not a transform fitted from future data. Notice that the feature construction contains no rental count. Later lessons replace this tiny representation with the project’s forty five calendar columns.

## 2:26 — Positions carry meaning

The vector has two positions, and each position has a declared meaning. Its first weight will multiply the working day value. Its second weight will multiply the scaled hour. Swapping those inputs without swapping their matching parameters changes the rule. It is not an equivalent serialization. That is why a saved predictor needs an ordered feature schema, rather than a bag of numbers. The small example makes this contract visible before a longer calendar vector can hide an ordering mistake.

## 3:00 — Move the request to the next hour

Now change the request to the hour from nine until ten. Keep the same one hour lead. The new issue time is eight, and the requested clock hour becomes nine. Nine divided by four is two point two five, while the working day flag remains one. The new interval has its own later outcome. We cannot reuse the previous interval’s target as its answer. We now have a target-free description of a specific future hour. The next explanation attaches weights and a bias to calculate its estimate.

