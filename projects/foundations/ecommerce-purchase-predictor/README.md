# E-commerce Purchase Predictor

I want to build an application that scores recorded shopping sessions, compares classifiers and shows how changing a decision threshold changes the mistakes. This is my second classical ML application, after [Bike Demand Planner](../bike-demand-planner/README.md).

The first project predicts a number. Here I predict a binary outcome and learn the distinction between a score, a probability estimate and an action. I will implement logistic regression and optimization explicitly, then compare practical classical classifiers.

**Status — 2026-10-15:** requirements and study plan prepared. Data acquisition, code, dependencies, training, benchmarks and application are pending. This project does not implement neural networks, embeddings or Word2Vec. Worked examples and budgets are illustrative or proposed, not dataset results.

[Foundations overview](../README.md) · [Previous project: Bike Demand Planner](../bike-demand-planner/README.md)

## What the application is for

The first user is an e-commerce analyst reviewing recorded sessions and model quality. The application will load a historical session or a validated feature payload, show its estimated purchase probability and compare it against a baseline. A threshold explorer will show the resulting label, confusion matrix, precision, recall and a declared hypothetical decision cost on the selected evaluation set.

An analyst can ask whether a model finds useful positive cases, whether its probability estimates match observed frequencies and which session types produce errors. This supports offline analysis and model selection. It does not prove that sending a coupon or changing a page causes a purchase.

### Prediction-time boundary

The source provides aggregate session records. A final-session page count or total duration cannot be assumed to be known halfway through a visit. Therefore the initial application is an **offline session-outcome scoring benchmark**, not a verified live intent detector. A live intervention model would need timestamped prefix snapshots, a fixed scoring point and features available before the remaining outcome; those data are not established here.

The UI must label this limitation. It should never show an early-session purchase forecast using a completed session as though it were a real-time observation.

| Item | Initial contract |
| --- | --- |
| Unit of observation | One provided session record |
| Target | `Revenue`: purchase outcome, mapped explicitly to `0/1` |
| Positive class | Purchase; the mapping is saved with the predictor |
| Model output | Finite probability estimate in `[0,1]` for logistic/probabilistic models, or a clearly labelled raw margin for an uncalibrated SVM |
| Decision | `positive` when the relevant probability is at least the stored threshold; other policies require their own artifact identity |
| Evaluation | Actual outcome is a separate field, never part of inference inputs |
| Unknown/missing input | Apply the documented learned preprocessing policy or return a clear validation error |

## Data source and feature audit

Use UCI's [Online Shoppers Purchasing Intention dataset, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). It contains 12,330 sessions, including 1,908 purchase outcomes and 10,422 non-purchase outcomes, and is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution to C. Sakar and Yomi Kastro and the [dataset DOI](https://doi.org/10.24432/C5F88Q).

The acquisition phase will download the official `online_shoppers_intention.csv` archive and record URL, access date, archive/CSV hashes, schema, license, row counts, label counts and validation findings. The authors describe sessions from different users, but do not expose user identities that let us independently prove that assertion. These records do not establish a current store's conversion behavior.

### Features need meaning before encoding

| Feature family | Questions and initial policy |
| --- | --- |
| Page counts and durations | Validate nonnegative values/units; describe them as completed-session aggregates in the initial benchmark |
| Month, visitor type, traffic type, region, browser and OS | Treat category codes as categories, not ordinal magnitudes; learn encodings on training data |
| Weekend / calendar proximity | Validate meanings and supported values; do not infer timestamps from row order |
| Bounce/exit rates | Audit how aggregates were calculated; exclude from the primary conservative feature set until provenance shows no held-out outcome contamination |
| `PageValues` | Exclude from the primary model: it is connected to transaction value and its aggregation/provenance must be justified before use |
| `Revenue` | Target only; never encode it into model inputs |

These exclusions are a conservative project policy, not a claim that every published use of these columns is invalid. An optional audit-only ablation can quantify their apparent benefit while clearly separating it from the primary result. If their historical aggregation used held-out outcomes, excluding the raw target alone would not repair leakage.

Check duplicates and near-duplicate feature vectors across splits, zero/long-tailed durations, unexpected category codes, invalid labels and missingness. If duplicate groups exist, freeze group assignments so identical records do not inflate evaluation. Distinct sessions can legitimately share features; report the grouping choice and resulting class counts.

## What carries over from project one

I should already understand vectorized linear predictions, batch-mean losses, independent gradient checks, SGD/momentum/Adam, scaling, validation versus test, experiment records and save/reload behavior. I can reuse those ideas and a versioned record format.

Here the new questions are: how does a linear score become a probability, why is the loss logarithmic, how do rare positives change evaluation, and how is a decision threshold chosen? I should not rerun every previous experiment without a classification-specific question.

## First session: score, probability and gradient

For an illustrative record, let `x = [1,2]`, `w = [0,0]`, `b = 0`, and purchase target `y = 1`.

```text
score z         = x · w + b = 0
probability p   = sigmoid(z) = 0.5
binary log loss = -log(0.5) ≈ 0.693147
gradient_w      = (p-y)x = [-0.5, -1.0]
gradient_b      = p-y = -0.5
```

With learning rate `0.1`, the next weights are `[0.05,0.10]`, bias is `0.05`, score is `0.30`, probability is about `0.574443` and log loss about `0.554355`. This is a single unweighted example, without regularization.

A decision threshold of `0.5` and one of `0.8` give different labels for that same probability. Changing the threshold does not retrain the model or change its probability estimate. The first stopping point is explaining that distinction and deriving the gradient sign for `y = 0`.

## Ordered implementation phases

We use the same explanation → assistant implementation/checks → code/result review cycle as project one. Every phase has concrete inputs, expected outputs and a changed-condition explanation.

| Phase | What I build and why | Evidence needed to finish |
| --- | --- | --- |
| 0. Task and first calculation | Define offline scoring, target semantics, allowed inputs and the first logistic update | Input/target contract, feature exclusions and checked arithmetic |
| 1. Data and split | Acquire/audit records, freeze labels/duplicate policy and partition data | Provenance/hashes, split identities and per-split class counts |
| 2. Baselines and metrics | Predict a training prevalence probability and a majority class | Log loss, average precision and confusion-matrix fixtures |
| 3. Logistic regression | Implement stable probabilities, logits-based binary log loss, gradients and regularization | Finite differences and a matching independent reference |
| 4. Optimizer experiments | Reuse gradient/update concepts on classification; compare rates, batches and schedules | Controlled convergence, validation quality and runtime report |
| 5. Imbalance and thresholds | Compare ordinary/weighted loss and choose a threshold using validation data | Precision-recall curves, costs and denominator-aware threshold report |
| 6. Classical model benchmarks | Compare linear SVM, trees, random forest and boosting | Fair feature/split/tuning protocol and distinction between margins/probabilities |
| 7. Calibration and errors | Check probability reliability and inspect mistakes by meaningful slices | Calibration evidence, slice counts and false-positive/negative examples |
| 8. Artifacts and application | Save complete preprocessing/model/threshold, build session replay and threshold explorer | Fresh-process prediction/decision parity and application checks |
| 9. Final review | Freeze choices, evaluate untouched test records and explain changed conditions | Final metrics/failures, reproducibility record and learner explanation |

## Models and training objectives

| Model | Purpose | Training/interpretation boundary |
| --- | --- | --- |
| Majority-class classifier | Expose how accuracy can hide missed purchases | Constant label from training data |
| Prevalence predictor | Baseline probability and log-loss comparison | Constant purchase probability estimated on training data |
| Logistic regression | Main from-scratch model | NumPy score/sigmoid/stable log loss/gradients with explicit optimizer state |
| L2 / optional L1 logistic model | Generalization and feature redundancy | Penalize weights, initially not the intercept; define normalization and L1 optimization |
| Linear SVM | Contrast margin-based classification with probabilistic fitting | Hinge objective and class labels `-1/+1`; margin is not a probability |
| Decision tree / random forest | Nonlinear references and interpretable error comparison | Library fitting procedures; tune depth and model size |
| Gradient boosting | Strong tabular comparison | Separate model family, with bounded validation tuning |

No neural-network classifier is included. SVM hinge loss is a separate model/objective comparison, not a drop-in equivalent of logistic log loss. Classifier probabilities from trees or forests still need calibration assessment.

## Losses, gradients and regularization

For batch size `B` and feature count `F`, use `X: [B,F]`, `w: [F]`, scalar bias `b`, logits `z: [B]` and binary targets `y: [B]`. Reject the accidental broadcast of `[B,1]` against `[B]`.

With `p = sigmoid(z)`, ordinary binary cross-entropy per example is `-y log(p) -(1-y)log(1-p)`. Implement it from logits as `max(z,0) - yz + log1p(exp(-abs(z)))` to remain finite for extreme scores. Its score derivative is `p-y`; average over the actual batch before adding the specified regularization derivative.

| Comparison | What changes | What I need to understand |
| --- | --- | --- |
| Ordinary log loss | Both classes receive equal per-record weight | Fit probability estimates under the training distribution; calibration remains empirical |
| Weighted log loss | Positive/negative records contribute different weights | May improve a decision tradeoff but changes the fitted objective and can alter probability calibration |
| Hinge loss with SVM | Penalize insufficient signed margin | Produces a decision score, not an immediate purchase probability |
| L2 / L1 penalties | Penalize coefficient size | Distinguish loss, regularizer and optimizer; L1 needs a stated kink/proximal policy |

For the initial weighted objective use `mean(weight_i × loss_i)` with training-derived class weights and document the denominator. Averaging by sum of weights changes the scale and effective regularization, so it is a separate convention. Do not silently compare these conventions at the same rate as if identical.

Weighted-logistic outputs are probability-like scores under a reweighted objective; do not advertise them as calibrated conversion probabilities without a held-out check/calibration procedure. Changing a threshold and changing the training loss are separate experiments. Focal loss is optional later, only if a concrete unresolved error pattern motivates it.

## Optimizer and learning-rate experiments

First hold the conservative feature set, split, unweighted log loss, model, initialization and mini-batch size fixed. Compare plain SGD, momentum and Adam with comparable rate-tuning allowances. Then change the batch policy or add one declared schedule.

Proposed starts are rates `0.0001, 0.001, 0.01, 0.1`, mini-batch size 32, and seeds `17,29,43` after a one-seed pilot. Pilot cap: 200 updates per trial. Main trial cap: 5 minutes or 10,000 updates, whichever comes first, adjusted only from recorded pilot evidence before the main comparison. Log updates, examples consumed and elapsed time; equal update counts across batch sizes are not equal work.

The classification-specific investigations are:

1. Does a rate that works for squared-error regression work for stable logistic loss on this feature scale?
2. Which optimizer reaches useful validation log loss fastest, and does its ranking quality agree?
3. Does the rare positive class create noisy updates at very small batch sizes?
4. Do class weights improve recall or decision cost while making log loss/calibration worse?
5. Does L2 help when one-hot categories are rare or correlated features compete?
6. Does a nonlinear library model improve held-out quality enough to justify its runtime and complexity?

Do these as separate studies, not one exhaustive sweep. Tune weighted and unweighted models fairly, freeze the ranking metric before interpreting results and retain failed trials. A nonfinite score/gradient, invalid probability or divergence is a diagnosed trial failure, not a result to discard silently.

## Evaluation and selection protocol

The dataset lacks exact session timestamps and exposed user IDs. A shuffled stratified split evaluates held-out records from this dataset; it does not establish future-store or chronological generalization. Do not fabricate a temporal ordering from CSV row position or month alone.

Proposed partition: 70% training, 15% validation and 15% final test with a fixed seed, adjusted at the data phase if duplicate grouping requires it. Store immutable row/group assignments and class counts. Within training, stratified cross-validation can tune model settings; preprocessing is fitted inside each fold. Validation records then support the declared threshold/calibration workflow. The final test remains untouched until all choices are frozen.

If calibration is fitted, reserve separate calibration and threshold-selection subsets inside validation, or use a predeclared cross-fitting scheme. Do not fit calibration and report its quality on those same records. Record the reduced positive counts and variability. Avoid a large search over tiny validation slices.

| Evidence | Meaning and reporting contract |
| --- | --- |
| Primary ranking metric | Average precision (AP); specify the implementation and positive class. Do not interchange AP with trapezoidal PR-AUC |
| Probability quality | Unweighted log loss and Brier score on the natural evaluation class distribution |
| Decisions | Precision, recall, F1 and TP/FP/TN/FN at a validation-selected fixed threshold |
| Supplementary | ROC-AUC, accuracy and balanced accuracy with class prevalence shown |
| Reliability | Calibration curve with bin counts; probability errors can remain despite good ranking |
| Operational scenario | Hypothetical decision cost and flagged-session volume under explicitly assumed costs |

Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)` and F1 is `2PR/(P+R)`. Define behavior when a denominator is zero; a model that predicts no purchases must remain visible in the report. A majority-negative model's high accuracy does not demonstrate useful purchase detection.

For a hypothetical false-positive cost `C_FP` and false-negative cost `C_FN`, compare total `C_FP×FP + C_FN×FN` or its declared per-session average. With calibrated probabilities and that particular two-error cost model, the theoretical threshold is `C_FP/(C_FP+C_FN)`. Actual campaign costs, intervention benefits and probabilities require evidence; this formula does not establish coupon profitability.

Inspect errors by visitor type, traffic category, month and session-length band with sample/positive counts. Feature attribution is predictive association, not a causal explanation. Repeated seeds show training variability; held-out resampling, if used, must respect grouping assumptions and be reported separately.

## Application behavior

The planned application has three views:

1. **Session replay:** select a held-out historical record, see features, baseline/model score, stored decision threshold and outcome revealed separately.
2. **Threshold explorer:** inspect the frozen model's decisions on validation data at different thresholds. Final test results use the frozen policy; exploratory test changes cannot become new unbiased acceptance claims.
3. **Benchmark and error review:** compare models, convergence, probability quality, runtime and false-positive/negative examples.

Show the model/source identity, feature policy and offline-data limitation. A margin-only model must display a margin. Unknown categories follow the saved encoder policy and a visible diagnostic; required missing fields cannot silently become zero unless the training contract explicitly permits that imputation.

## Correctness and changed-condition checks

- Check the worked update independently and compare float64 gradients against central differences at smooth points, with proposed `atol=1e-6, rtol=1e-5` verified against an independent implementation.
- Verify stable loss/probabilities at logits `-1000, 0, 1000`, both labels and unequal batch sizes. The implementation must not take `log(0)` through a naive probability path.
- Test target mapping, batch averaging, weight normalization, unpenalized intercept and Adam/momentum state over multiple updates.
- Compare logits/probabilities against an independent logistic reference with matching objective and regularization conventions; different solver defaults are not an equality contract.
- Use tiny known confusion matrices to check metrics and threshold ties, including zero predicted positives and a single-class evaluation error policy.
- Check that preprocessors, resampling and feature selection fit on training records only; no oversampling occurs before splitting.
- Check unseen categories, column-order changes, missing fields, forbidden `Revenue`/`PageValues` inputs, nonfinite durations and exact artifact schemas.
- Reload predictions and threshold decisions in a fresh process. A changed threshold should change labels, not scores.
- Explain a model that has good ranking but poor calibration, or high accuracy but zero useful positive recall.

## Source, artifacts and reproducibility

Proposed implementation layout; only this README currently exists:

```text
ecommerce-purchase-predictor/
├── README.md
├── pyproject.toml / uv.lock
├── prepare_data.py / train.py / compare.py / inference.py
├── configs/
├── src/ecommerce_purchase_predictor/
├── tests/
├── app/                     # session replay, threshold and benchmark views
├── notebooks/               # optional thin exploration/Colab entry
└── docs/                    # provenance, feature audit and implementation checkpoint
```

Scripts will declare inputs explicitly in their main blocks. No executable setup/launcher commands are promised before implementation. Any future shared TUI work belongs in `cli/`.

Each run records data/split/feature-policy hashes, source/dirty identity, dependencies, preprocessing, model/objective/reduction/penalties/class weights, optimizer/schedule/batch, seed, budget, hardware, timing and failure status. Keep raw experiment outputs separate from authored interpretation.

Inference exports include the feature schema/order, label mapping, encoders/scalers, model state, any calibrator, threshold-selection record and source/config identity. Save provenance with the calibrator's fitting subset. Load only trusted compatible artifacts.

Resumable training additionally saves optimizer moments, update/schedule index, sampler/next-batch position, RNG and monitoring state. Verify local interrupted-versus-uninterrupted parity. A changed feature vocabulary or class mapping must reject incompatible checkpoints.

Use local CPU first and measure before increasing the benchmark budget. Colab is an optional execution route for these classical models; if added, follow the [workspace persistent recovery contract](../../../docs/guide/COLAB.md) using the same trainer and verified persistent state. A notebook or a reload in the same runtime is not proof of fresh-runtime hosted recovery. None has been demonstrated here.

## Completion review

- [ ] Offline session-scoring task, positive class and feature availability/exclusions are enforced.
- [ ] Dataset attribution, hashes, split/group identities and class counts are reproducible.
- [ ] Baseline, stable logistic objective and optimizer implementations have independent numerical checks.
- [ ] Loss/weight/rate/batch/schedule/regularization studies explain their practical tradeoffs and failures.
- [ ] Classical model comparisons use consistent features and fair validation budgets.
- [ ] Threshold and any calibration are selected without using final test labels.
- [ ] Final AP, probability metrics, confusion matrix, slice counts and limitations are recorded.
- [ ] Complete inference artifacts reproduce scores and decisions in a fresh process and drive the application.
- [ ] I can derive a logistic update, distinguish probability from decision and diagnose a changed-condition failure.

Completion requires explaining observed quality rather than inventing a target accuracy. Real-time purchase prediction and measured marketing uplift remain separate future capabilities requiring suitable data and evaluation.

## References and next step

- [UCI dataset and feature definitions](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset): provenance, imbalance and feature audit; checked 2026-10-15.
- [Scikit-learn SGD guide](https://scikit-learn.org/stable/modules/sgd.html): linear classifier/objective reference; pin the installed implementation version at kickoff.
- [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): preprocessing and selection leakage.
- [Tutorial 0 outline](../../../docs/tutorials/projects/p00/transformer-foundations/v1/project/course-outline.md): probability, calculus, gradients, losses, optimization and evaluation.

Begin phase 0 only after Bike Demand Planner's completion review. This README completes the plan for the second classical ML application; it does not choose or implement the later deep learning or embedding project.
