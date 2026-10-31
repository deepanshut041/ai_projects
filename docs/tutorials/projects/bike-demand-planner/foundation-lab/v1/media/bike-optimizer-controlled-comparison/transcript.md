# Freeze a fair comparison and read what it says

## Separate three questions

We want to know what an update rule changes, so we avoid a large cross product in which many settings move together. The optimizer family fixes batch size thirty two. The batch family fixes SGD and rate zero point zero one. The schedule family fixes SGD, batch thirty two and its selected starting rate. They share the same calendar inputs, training only scaler, zero weights and mean bias, frozen row identities and error definitions. The three seeds are seventeen, twenty nine and forty three. Matching a seed pairs the training order across comparable trials. These controls make a measured difference interpretable. They do not remove the additive model limitations already observed.

## Choose rates using training loss only

Before the main trials, each rule receives the same four candidate rates and two hundred pilot updates. We reset parameters and optimizer buffers each time. Selection uses final full training half squared error among completed candidates. Validation has not been loaded. SGD chooses zero point zero one, momentum zero point zero zero one, and Adam one. Their selected training losses are about three thousand, two hundred and seventy two, three thousand, two hundred and fifty five, and three thousand, two hundred and fifty six. Rate one makes the SGD and momentum pilots diverge. Those failures are retained, rather than erased from the allowance. We then write and hash the selected rates, budgets and seeds before fitting main trials.

## Freeze first, fit next, score later

This shortened source trace makes the information boundary visible. Training pilots determine the rates, and a saved protocol fixes those choices. Main fitting finishes before validation is loaded. The shared scorer then checks record, target time and issue time alignment. The rate one failures retain explicit work counters. A failed attempted batch can consume examples without completing another parameter update. The failure record therefore distinguishes examples visited from updates applied. A failed pilot is retained as evidence about its rate, without receiving a validation score.

## Count independent trials and verify isolation

Identical configurations reuse the same fitted trial for a given seed. There are twelve pilot candidates and eighteen unique main fits. Reused SGD entries are not extra independent repetitions, even when they appear in more than one family table. A small integration fixture changes validation outcomes and checks that pilot choices and saved model bytes remain unchanged. Independent audits also recalculate protocol and artifact hashes, training work counts, prediction identities and scalar metrics. These checks support the declared information boundary, while the later validation scores measure forecast behavior.

## Compare the same later hours

At batch thirty two and ten passes, mean validation absolute error is one hundred and sixteen point one seven for SGD, one hundred and sixteen point six zero for momentum, and one hundred and fourteen point three six for Adam. Adam has the lowest observed mean, but its sample deviation is two point three seven, larger than the roughly zero point six for the other rules. All remain worse than the group baseline at ninety one point five seven. Three orderings provide a small reproducibility comparison, not a population confidence interval. Pilot seed seventeen is also one main seed, which we declare openly. We retain every seed result and avoid presenting the most flattering one as the whole experiment.

## A lower average does not fix every hour

Return to the later Monday at eight. Its target is four hundred and twenty rentals. For seed seventeen, SGD forecasts one hundred and sixty eight point nine one, while Adam forecasts one hundred and fifty three point two one. Adam is worse on this hour even though its overall mean absolute error is lower. Average improvement is not a promise for every observation. Both rules still produce negative predictions at the quiet two oclock example. We keep raw predictions without hidden clipping or rounding. The optimizer changes the route through the same additive function family; it cannot invent an hour by working day interaction. The measured outcome supports a bounded comparison, while the stronger group baseline and remaining forecast failures stay visible.
