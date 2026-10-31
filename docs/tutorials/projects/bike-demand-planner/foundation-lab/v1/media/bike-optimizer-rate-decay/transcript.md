# Reduce the rate using work already processed

## Make later steps smaller on purpose

With a fixed rate, later noisy batches can keep moving parameters around the training optimum. A decreasing rate may settle the fit. We test that question while holding SGD, batch thirty two, initial rate zero point zero one, paired orders and ten passes fixed. There is one declared inverse time schedule, rather than a search across many shapes after looking at validation. Its progress measure is examples already processed before the next batch update. That timing is part of the rule. We will compare both training squared error and later absolute error. Better optimization of the first quantity does not mathematically guarantee an improvement in the second.

## Anchor the schedule to training exposure

The base rate is divided by one plus examples seen over forty three thousand, two hundred and twenty five. That denominator equals five complete training passes. At zero examples, the rate is zero point zero one. At five passes, the denominator is two, giving zero point zero zero five. At ten passes it is three, so the next rate would be about zero point zero zero three three three three. The run has already reached its work budget then. A saved next rate describes what a future update would use; it does not imply an extra update happened. Writing the progress unit explicitly prevents confusion between parameter updates and examples processed.

## Evaluate the rate before consuming the batch

First calculate the rate from completed exposure. Then obtain the next batch and compute its actual mean gradient. Finally apply the optimizer step with that rate. If we advanced the example counter before evaluating the schedule, even the first batch would start below the declared base rate. At the tail, count the five examples actually visited, not thirty two imaginary ones. One hundred updates means one hundred examples for single rows, thirty two hundred for ordinary mini batches, or eight hundred and sixty four thousand, five hundred for full batches. Choosing examples as the clock makes the intended exposure meaning explicit. Choosing updates could also define a study, but it would be a different schedule contract.

## Read both the optimization and forecast results

The decayed rule gets closer to the training optimum. Mean half squared error falls from about three thousand, one hundred and seventy seven point nine zero to three thousand, one hundred and sixty eight point three five. The earlier converged reference is about three thousand, one hundred and sixty seven point five three. Yet mean validation absolute error increases from one hundred and sixteen point one seven to one hundred and sixteen point five six. That is about zero point three nine worse. Seed spread is smaller with decay, but that alone does not make the forecast better. The two objectives, their data windows and their units must stay distinct when reading the result.

## Inspect paired seeds before drawing a conclusion

Pairing shows that decay worsens validation absolute error for seeds seventeen and forty three, and slightly improves it for twenty nine. We keep that detail beside the mean. A quieter final trajectory therefore supports a claim about optimization stability in this run, rather than universal forecast superiority. All trials stop after their example allowance. We did not configure a convergence test or stop because a curve looked flat. Time, update, divergence and nonfinite stops have separate recorded reasons. Changing the decay interval after reading validation would be a new comparison. For continuation, save the example count and schedule configuration so the next rate follows the same clock instead of restarting at the base value.
