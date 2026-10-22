# Comparisons: Compare batches, momentum and Adam

The [worked explanation](lessons/05-batches-and-optimizers.md) follows the same real hour through the rules and the actual dataset study. This page collects the alternatives and the evidence that distinguishes them.

## Estimate a gradient, then choose how to use it

Batch size determines which residual contributions are averaged at the current parameters. The update rule determines what happens to that estimated gradient, including remembered state. Changing both together would mix two mechanisms. We compare optimizers at batch 32 and compare batch policies with plain SGD at rate 0.01.

Shuffling is restricted to training membership. It changes optimization order without mixing later validation outcomes into fitting. The five-row epoch tail is retained and divided by five. A fixed denominator of 32 would underweight it, while discarding it would change the exposure budget.

## Plain SGD or a gradient-accumulating momentum convention

SGD uses the current gradient directly. Our momentum rule accumulates `velocity = 0.9*velocity + gradient`, then applies the rate. The [PyTorch SGD documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html) shows this gradient-accumulating convention and distinguishes a rate-scaled velocity formulation. At a constant rate, suitable state scaling can make those formulations equivalent; changing the rate requires care about what the stored velocity means.

For the repeated real hour at rate 0.01, both first predictions are 148.3743. The second becomes 150.8989 for SGD and 155.0208 for momentum. The selected dataset rates differ, 0.01 and 0.001, because equal nominal rates do not imply equal effective movement with accumulated history.

## Current-gradient magnitude or Adam's moment normalization

Adam's directional and squared-magnitude averages rescale coordinate updates. We use corrected moments with t starting at one and epsilon outside the square root. The first repeated-hour forecast moves much less than SGD's at rate 0.01, while the dataset pilot selects Adam rate 1.0 on this unscaled rental target.

The small probe explains the rule; it does not rank its fitted results. The changing-gradient +2/−1 fixture checks remembered direction, and the tiny-gradient fixture checks epsilon placement. A different correction or epsilon convention would be another algorithm to identify explicitly.

## Equal examples, equal updates or equal time

| Budget held equal | What changes with batch size |
| --- | --- |
| Examples processed | Number of parameter updates and optimizer-loop cost |
| Updates | Number of visited examples and work per update |
| Wall time | Both exposure and updates, depending on implementation/hardware |

We use ten passes and report the other two quantities. Single-row, batch-32 and full-batch SGD perform 86,450, 2,710 and ten updates respectively. The full-batch run's poor fixed-budget score does not contradict the earlier converged full-batch solution; it has taken only ten steps at a smaller rate. Timings describe this machine and monitoring scope rather than a universal throughput ranking.

The original 10,000-update proposal would have truncated single-row training. The revised 100,000 cap is a safety limit, with the 300-second cap and equal-example budget recorded before main comparisons. Actual finite main trials end at the example budget rather than claiming gradient convergence.

## The same numerical rate or the same tuning allowance

We give SGD, momentum and Adam the same four-rate training-only pilot and 200-update allowance. The grid is 0.001/0.01/0.1/1.0. SGD and momentum at 1.0 fail the declared divergence guard; those failures are retained. Rate selection uses final training half-MSE among completed candidates, without opening validation.

The chosen rates are therefore explicit experimental settings rather than library defaults or a guarantee of globally optimal tuning. Pilot seed 17 also appears among the paired main seeds, a declared limitation of the small study. There is no large rate/batch/optimizer/schedule cross-product search or final-test selection.

## Constant rate or one inverse-time schedule

The schedule study changes only the rate policy for SGD at batch 32. It uses processed examples before each update and halves the initial 0.01 after five passes. Its smaller final training objective, 3168.3497 versus 3177.8966, is accompanied by slightly worse mean validation MAE, 116.5579 versus 116.1661.

This is evidence that training squared-error fit and later absolute-error quality can move differently. A different schedule or decay point remains an untested alternative. The final saved next rate is the rate for a possible continuation, not an extra performed update.

## Lowest training objective or lowest observed validation MAE

Momentum's mean training objective is slightly below constant-rate SGD's, while its validation MAE is slightly worse. Adam has higher training half-MSE but lower mean validation MAE across the three observed seeds. Its spread is larger, and a favorable seed is not a substitute for the full paired results.

All retain the same additive calendar function and unconstrained real-valued prediction policy. None beats the hour/working-day baseline's 91.5729 MAE. Features, interactions, alternative losses and clipping require later separate comparisons; they are not quietly included as optimizer improvements.

## Save a prediction or preserve the next training step

A forward predictor needs parameters, scaler and feature order. Continuation additionally needs moments/velocity, update index, schedule progress, permutation/cursor, RNG state and matching source/data identities. The numerical JSON pause/continue checks preserve parameters, moments and order exactly while elapsed time changes normally.

The saved real-trial state prepares the recovery contract. A production fresh-process restore and supported application interface remain later work. Learner understanding is also separate from code and media checks; no new explanation has been assumed.

Return to the [worked understanding notes](lessons/05-batches-and-optimizers.md).
