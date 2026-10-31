# Learning points: Follow tree leaves, forests and boosted corrections

A tree turns a calendar request into a sequence of questions. Its fitted state is a set of feature tests, branches and leaf values. In our four-row example, the hour-8 test separates historical counts 179 and 195 from quiet counts 1 and 2. The leaves predict 187 and 1.5 because squared-error fitting chooses the mean within each region. Once the tree is fitted, a request supplies only calendar inputs. Its outcome is used later for scoring. The apparent simplicity of a prediction should not hide the historical learning that produced its questions and leaf values.

That small example also makes unsupported behavior easy to see. A request for 09:00 has the same zero hour-8 indicator as a request for 02:00, so the restricted teaching stump returns 1.5 for both. It has no other way to distinguish them. The full project model receives all 45 calendar columns, but it still treats requests with identical available calendar values identically. More elaborate fitting cannot manufacture an omitted observation. This is the same information-set constraint that mattered in linear regression, expressed through a different family of functions.

## More detailed regions need enough useful history

A deeper tree can condition on combinations of hour, working-day status, month and weekday. Unlike the additive linear function, it can create such conditional relationships through its paths without an explicit product column. This is additional representational freedom. Minimum leaf support and leaf-count limits decide how finely that freedom can divide the training data. They therefore affect both what can be fitted and how much history supports each stored prediction.

The most flexible bounded tree has much lower training MAE than the hour/working-day baseline: about 25.83 versus 55.43. Its later error is worse, around 100.00 versus 91.57. The actual January commute makes the distinction tangible. The reached leaf remembers eight training counts summing to 1,446 and predicts 180.75, while the broader baseline predicts 349.50 against a later target of 420. A narrower historical partition can capture the past more accurately while transferring less well. That result does not identify a single cause of the future change, and our depth sweep itself improved both training and validation. It would be misleading to label every deeper tree an observed overfit solely because it was deeper.

Changing the same Monday request to 02:00 sends it to another leaf, which predicts about 6.55 against target 5. The tree can be useful on one hour and poor on another. A model-level conclusion needs the full retained validation population rather than a selected example. This is why the notes keep both hours beside the aggregate table.

## Averaging helps when there is useful variation to average

The forest builds 64 varied trees from bootstrap training samples and feature subsets. Each component still makes an ordinary tree prediction. The ensemble then averages their leaf outputs. On the busy Monday, the first three components return 180, 190.3 and 187.5, but the full 64-tree mean is about 178.70. Showing the first three trees explains the components; it would be wrong to present their partial mean as the complete forest prediction.

Varied training views can reduce sensitivity to individual observations and split choices. They do not remove a shared failure to represent later demand. If every tree predicts 187, averaging sixty-four copies still gives 187. Our forest has slightly better mean validation MAE than the single tree, but it remains worse than the simple baseline. Its roughly 16.1 MB numerical snapshot also costs much more storage than one tree. Additional trees are a tradeoff supported by measured quality and resource observations, not an automatic promotion to a better model.

Bootstrap support deserves careful reading. A component can contain repeated copies of one training row. Its distinct-row count and weighted sample count therefore differ, and a leaf mean follows the latter. The plain-tree busy leaf's eight rows are unweighted, so its arithmetic is simply 1,446 divided by eight. Keeping these cases separate prevents a support label from quietly changing meaning.

## Boosting adds changing corrections

Boosting uses earlier predictions when deciding what the next tree should learn. For squared error, its correction target is observed target minus current prediction. That is the opposite sign from the prediction-minus-target residual used by our scoring functions. The first tree in the four-row probe sees a busy residual mean of 92.75. Applying only one tenth of that correction moves the initial 94.25 prediction to 103.525. The second stage sees smaller residuals and moves it to 111.8725. The numbers show why sequential correction differs from averaging independent finished forecasts.

A correction tree's leaf can be negative. Summing many such corrections can produce a negative total even when every training target is nonnegative. The full booster predicts about −25.52 at the quiet Monday hour and 218.53 at the busy hour. Explicit nonnegative clipping changes the former to zero and leaves the latter alone. The raw/clipped metric pair tells us what that output decision buys; a squared-error boosting objective does not itself enforce valid rental counts.

The declared stage counts are 100, 200 and 300. All use depth-three trees and learning rate 0.1, with no validation early stopping. Increasing the count lowers training error and improves this grid's later scores, but does not beat the group baseline. Changing the learning rate or loss would be a new comparison. We keep those settings fixed rather than silently combining many optimization decisions and attributing their effect only to the model family.

## A fair record exposes different work rather than pretending it is identical

Tree construction, forest averaging, sequential boosting and linear gradient updates do different work. Equal update counts would not equalize these estimators. Instead, the practical study declares structural caps, three configurations and three seeds per family, local timing pilots, fit guards and the exact stopping rules. The pilots only check feasibility using training data. Validation is opened after every fitted main model and deterministic reference has been saved.

Within each configuration, selection uses mean validation MAE across all declared seeds. The forest's 99.7549 is that mean; the retained seed-17 artifact scores 99.8324. Seed 17 was declared in advance as the representative, so the artifact was not chosen for having a lucky score. Variation across seeds describes the algorithm on these same dates, not independent samples of future demand. The identical boosting scores across seeds are likewise an observed result, not extra evidence about different periods.

```mermaid
flowchart LR
    F["Fit only 2011 history"] --> V["Evaluate fixed later hours"]
    V --> C["Compare quality, complexity and output policy"]
    C --> R["Keep groups/raw: MAE 91.5729"]
    R --> Q["Review why finer history failed on the busy hour"]
```

The arrows show the order of evidence and the resulting learning question. The existing group baseline remains selected because it has lower validation error while requiring less machinery. The project still gained working tree, forest and boosting paths and an explanation of their behavior. Implementation success and model selection are different outcomes.

The independent audit reconstructs numerical predictions, metrics, leaf membership and all artifact identities. The 83 checks include small known answers, state replay, malformed contracts and a changed-validation fixture that leaves fitted models unchanged. These establish what the code does. They do not prove final-test performance, hosted recovery or the learner's understanding. The same validation period has already informed several development decisions, so final assessment remains reserved. Return to the [worked notes](lessons/07-trees-and-ensembles.md) to follow the exact source, real-hour paths and complete results before discussing the next error-review stage.

Return to the [worked understanding notes](lessons/07-trees-and-ensembles.md).
