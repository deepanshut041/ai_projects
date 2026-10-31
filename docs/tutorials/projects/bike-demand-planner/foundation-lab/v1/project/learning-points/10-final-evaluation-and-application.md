# Learning points: Evaluate later hours and inspect the completed application

## The final test answers a question that must already be settled

At 07:00 on 2 July, the saved model can forecast the 08:00 hour from its calendar without knowing the later count. That simple ordering carries the whole evaluation idea. We first decide what function to measure, what inputs it may use and how success will be described. We then compare its saved forecasts with observations. If the count changes the function or the reporting rule before the result is presented, we have moved from assessment back into development.

The project made this order visible by saving an immutable protocol and every target-free final forecast before the raw target join. The protocol bound seven representative model artifacts, their transforms and output policies, metric/slice definitions and source/data/dependency identities. The normal development loader still refuses test labels. These controls make mistakes easier to detect and the result easier to reproduce. They do not make a weak model strong, and a hash alone does not certify a scientific claim.

Keeping the original 2011 fits was a deliberate choice. Refitting after validation could be a legitimate procedure if declared beforehand, but it would produce a different fitted function. Here we wanted to measure exactly what had already been selected and exported. The later score therefore describes these historical fits. It should not be stretched into a claim about the best possible model or every reasonable training procedure.

```mermaid
flowchart LR
    V["Development selects groups/raw"] --> F["Freeze the function and report"]
    F --> P["Save predictions without outcomes"]
    P --> E["Join later observations; measure error"]
    E --> D["Any new choice is test-informed development"]
```

The last arrow matters as much as the first. Once final outcomes help choose the next model, they are part of that new development history. Hiding that use would make the next claim stronger than its evidence.

## An accurate code path can return an inaccurate forecast

The 2 July working-day hour-8 group has a saved training mean of 86,677 divided by 248, or 349.504 rentals/hour. The raw target later reveals 568, so the signed error is −218.496. The predictor did exactly what its fitted function specifies. The underprediction exposes a limitation of that function's transfer to later demand; it is not automatically a numerical or serialization bug.

This distinction also explains why no update follows that residual. During training, gradients tell us how to change fitted parameters. During this final assessment, parameters must stay fixed so the result measures the chosen function. Reusing a newly observed test target in an update would change the evidence we are collecting, even if the update reduces that hour's error.

The additive linear reference gives a different forward calculation on the same input. The July row activates hour 8, Monday, July and working day, then uses the scaler fitted only on 2011. Its standardized dot product is 171.2404 and intercept 143.7944, giving 315.0349. Centering means inactive binary coordinates also contribute; summing only active standardized coordinates would implement a different function. The nonnegative policy is already selected and leaves this positive value unchanged. This individual forecast misses more than the group mean, while the additive model's aggregate final MAE is lower. A single request is a useful mechanism check, not a substitute for the defined evaluation population.

## A later ranking reversal is useful evidence

Groups/raw won the development comparison at validation MAE 91.5729, ahead of the retained forest at 99.8324. Later, groups scores 125.4167 and the fixed forest 103.6474. The honest reading is that the original choice did not transfer as well over this later period. We should keep both the decision and the reversal visible. Choosing the forest after seeing the table may be sensible for a subsequent development decision, but it is now informed by these final outcomes.

The forest result belongs to the first predeclared seed-17 artifact, while earlier configuration selection averaged over seeds 17, 29 and 43. Those two summaries have different meanings. Seeds vary the fitting algorithm on the same data; they are not independent samples of future environments. Neither a lucky seed nor a new final ranking should silently replace the already chosen representative or selection procedure.

The project did not define an operational acceptance threshold. Beating a global mean is useful evidence, but it cannot establish that a planner's inventory decisions improve. Observed rentals do not measure every person who wanted a bicycle, and absolute rental error does not specify the relative cost of an idle bicycle and a missed commute. Those are additional task and decision contracts, not facts to infer from a table.

## The denominator tells us what an error number means

Final MAE 125.4167 divides total absolute error 548,698.1119 by 4,375 recorded hours. Working-day rush MAE 265.4320 divides by its own 741 hours. The latter exposes a much harder part of the task without changing the original rush definition. Comparing averages while omitting their counts would conceal the population each number describes.

The total final deficit is 517,941.0085 rentals. Dividing by all recorded hours gives 118.3865 rentals/hour; dividing only by the 3,735 underpredicted hours gives 138.6723. Both are correct, but they answer different questions. The signed error of −111.3563 also includes overpredicted hours, so it is smaller in magnitude than the all-hour deficit. Positive and negative errors can cancel in signed error, while absolute error prevents that cancellation.

Forty expected naive slots have no recorded final outcome. Keeping them missing preserves the question the data can actually answer. A missing hour is neither a known zero count nor a known model failure. January–June month slices are empty in the final period, so their metrics are null, while the six observed month counts add back to 4,375. Different slice dimensions overlap: a rush hour is also in a month, so adding those separate dimensions would double-count it.

## A stable calendar function can miss a changed demand level

The model returns 349.504 for both 2 July and 3 December at 08:00 because both use key `(8,1)`. Their outcomes are 568 and 731. The selected function cannot use month to adjust the answer, even though month exists in the shared vector. On the recorded 4 July holiday, working-day status becomes zero and the estimate falls to 76.4348, against actual 153. These examples separate a feature that changes the selected function from one that is available but unused.

Final mean target is 254.1147 while mean forecast is 142.7584, consistent with the strongly negative signed error. This describes a level difference rather than establishing a cause. Nearby errors also move together: consecutive-hour residual correlation is .8349 over 4,369 observed adjacent pairs. Treating all hours as independent future trials would overstate how much independent evidence the dataset supplies. The report therefore retains descriptive day/week/slice summaries without an unsupported independent-hour confidence interval.

## Reproduction, usability and understanding remain separate achievements

The final page verifies the original saved bundle and shows final evidence beside the historical request interface. A prediction response contains no observed outcome; an explicit later action retrieves the observation. Missing observations stay missing, and unsupported dates clear old forecasts. The original request domain remains January–June because its saved recorded holiday authority covers those dates. Having later batch features for evaluation does not create unrestricted future-serving knowledge.

The earlier linear recovery run establishes exact local numerical continuation with complete state and explicit damaged-save fallback. The final assessment establishes measured forecasting error on a later period. Browser and installed-wheel checks establish usable packaged local behavior. Each result matters, but none proves the others, learner mastery or hosted recovery. All 122 checks pass while substantial underprediction remains visible.

The next useful review is to explain what changes if we adopt the forest now: the model decision becomes test-informed, the observed final comparison remains reportable, and a fresh independent performance claim needs new held-out evidence. That conversation consolidates the prior tutorials and implementation without requiring routine code typing. The [worked notes](lessons/10-final-evaluation-and-application.md) keep the complete calculations, source traces and results together.

Return to the [worked understanding notes](lessons/10-final-evaluation-and-application.md).
