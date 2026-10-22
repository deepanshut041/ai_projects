# Foundations

I want to turn my machine learning revision into applications I can build, benchmark and explain. I already know Python and have worked with machine learning before; this is a practical return after a year away, with maths refreshed when I need it.

My first stage has two classical machine learning projects. I will finish the regression application before beginning the classification application. Comparing optimizers, losses and learning rates is a central part of both, alongside the complete data-to-application workflow.

My second stage is [Waste Material Classifier](waste-material-classifier/README.md): one vision application built through fully connected neural networks, scratch CNNs, residual blocks and a transfer-learning comparison. This makes it the third application in the folder, after the two classical ML applications.

My third and final foundations stage is [Air Quality Forecaster](air-quality-forecaster/README.md): a real time-series application for RNNs, LSTMs, GRUs, gated memory, backpropagation through time and multi-step forecasting. It is the fourth application. The existing embedding/Word2Vec Support Ticket Router follows these three stages.

**Status — 2026-10-15:** the collection README and four separate project READMEs are prepared. Bike Demand Planner now separately reports a Python package/entry-point scaffold; its data, estimators, training and application remain pending. The new sequence project is README-only. Each project's own README owns its current implementation/setup status; preparation here does not establish model, experiment, learner or hosted completion. Inputs, predictions, budgets and acceptance thresholds in the plans are proposed or illustrative until measured.

## Start here

1. [Bike Demand Planner](bike-demand-planner/README.md): predict hourly bike rentals, study regression and understand optimization on a real prediction problem.
2. [E-commerce Purchase Predictor](ecommerce-purchase-predictor/README.md): estimate purchase outcomes from recorded shopping sessions, study classification and understand probability, imbalance and decision thresholds.
3. [Waste Material Classifier](waste-material-classifier/README.md): recognize a waste image's material category and study neural networks, convolutions, architecture, training, augmentation, residual connections and transfer learning.
4. [Air Quality Forecaster](air-quality-forecaster/README.md): forecast future PM2.5 from observed histories and compare recurrent cells, memory, gradient flow, state handling and multi-step prediction.

Start with Bike Demand Planner. I do not need to finish every revision video or all of Tutorial 0 before reviewing its first calculation and data contract.

| Project | Application | Main learning | Status |
| --- | --- | --- | --- |
| 1. Bike Demand Planner | A historical demand planning and model comparison tool | Regression, gradients, batch sizes, learning rates, losses, regularization, time-aware validation and residual analysis | Python package/entry-point scaffold prepared; data/models/training/application pending |
| 2. E-commerce Purchase Predictor | An offline session scoring and classifier evaluation tool | Logistic regression, stable log loss, class weighting, thresholds, calibration, PR metrics and classification errors | README prepared; follows project 1 |
| 3. Waste Material Classifier — vision DL stage | An image classification, model comparison and error-review assistant | Fully connected networks, CNNs, initialization/backpropagation, pooling/normalization, regularization, loss/optimizer/rate studies, residual paths, transfer learning and robustness | README prepared; follows both classical ML applications |
| 4. Air Quality Forecaster — sequence DL stage | A historical monitoring/forecast replay and comparison tool | RNN/LSTM/GRU, BPTT, gates, context/horizon, missingness, chronological evaluation, state/reset/TBPTT and recurrent encoder-decoder forecasts | README prepared; follows the vision application |

Applications 1 and 2 are the classical ML stage; application 3 is the vision deep learning stage; application 4 is the final sequence deep learning stage. Waste classification remains single-label image recognition; recurrence is covered separately by the forecaster. Detection, segmentation and Vision Transformer builds are outside this foundations route. Word2Vec and learned text embeddings follow in the existing [Support Ticket Router](../transformer-labs/support-ticket-router/README.md), without recreating it. The older P00 foundations lab and its saved `linear()` exercise are preserved; this preparation does not complete any learner gate.

## Why this sequence

Bike demand gives me a numerical target with meaningful large errors and seasonal changes. Purchase prediction gives me a binary target, unequal class frequencies and a decision threshold. Together they cover regression and classification without repeating two nearly identical regression projects.

Waste classification then adds learned nonlinear and spatial features. I can compare an image MLP with a CNN, understand shared filters/feature maps and investigate whether normalization, augmentation, residual paths or pretrained features actually help. It is one evolving application rather than several disconnected architecture demos.

Air-quality forecasting then adds time order, hidden/cell memory, causal input availability and forecast horizons. Simple lag-based models establish whether recurrence helps; gated cells and multi-step decoders have concrete failure cases to explain. This prepares sequence reasoning before the later embedding/text application.

I want to answer questions such as:

- Can I derive a gradient, check it independently and explain an unstable update?
- Does Adam reach a useful solution faster than momentum when each gets a fair tuning budget?
- Is a lower training objective also a better prediction on unseen examples?
- When does an extra feature improve a model, and when does it leak the answer?
- Which errors matter to the person using the application?
- Does a stronger library model justify its complexity compared with my simple implementation?

These questions guide the experiments. The projects are not a claim to cover every branch of ML or deep learning: clustering, anomaly detection and detection/segmentation remain separate topics. The selected four applications provide regression, classification, vision and recurrent forecasting practice.

## Vision deep learning stage

The [Waste Material Classifier README](waste-material-classifier/README.md) owns its full data contract, build phases, worked convolution/gradient example, proposed shape/parameter counts and experiment matrix. Its route is:

```text
simple image baseline -> image MLP -> small scratch CNN
    -> controlled architecture/training/generalization studies
    -> matched residual comparison -> frozen/fine-tuned pretrained CNN
    -> robustness/calibration/review -> saved predictor and application
```

The experiment matrix covers activation/initialization, depth/width, kernels/stride/padding, pooling/head choices, BatchNorm, SGD/momentum/Adam/AdamW, learning rates/schedules/batches, dropout/decay/early stopping, augmentation, CE/weighting/smoothing, data size/resolution, residual paths and transfer learning. Focal loss and hardware-specific performance work remain conditional extensions. These are bounded studies, not an exhaustive cross-product.

Image originals/groups are split before augmentation; pretrained weights and transforms have recorded identities. Scratch and transfer runs are reported separately. Softmax scores, saliency pictures and a low-score review threshold do not establish calibrated confidence, causal explanations or unknown-material detection. Local numerical checks, measured task quality and actual fresh-runtime Colab recovery have separate evidence gates.

## Final sequence deep learning stage

The [Air Quality Forecaster README](air-quality-forecaster/README.md) owns its real monitoring-data contract, ordered RNN/LSTM/GRU route, worked BPTT/gate examples, tensor/parameter counts and controlled experiment matrix:

```text
causal hourly data/windows -> persistence and lagged linear/MLP baselines
    -> explicit RNN cell/unroll -> LSTM and GRU
    -> context/horizon/training/loss/state/TBPTT studies
    -> direct/recursive/recurrent encoder-decoder forecasts
    -> historical replay, error review and persistent recovery
    -> existing Support Ticket Router for embeddings/Word2Vec
```

Forecasts use only observations available at issue time. Split by target time before constructing windows, retain missing-input versus missing-target masks, and report errors by horizon. Independent-window resets are the default; stateful streams and truncated gradients need explicit policies. Attention and text embeddings are outside the forecaster, and its PM2.5 estimates are not a measured AQI/health-warning service.

## How I will work through each phase

Each phase begins with its purpose, prerequisite, concrete inputs, expected outputs and stopping point. We explain the reasoning with a small example, the assistant writes the implementation and meaningful checks, and we review the code and results together. I predict a changed case and explain what the evidence supports before moving on. I do not need a Python syntax course or to memorize library method names.

The continuous route is:

```text
real user question -> available data -> split and baseline
    -> model and loss -> gradient check -> controlled experiments
    -> validation and error analysis -> saved predictor -> application
    -> independent explanation and changed-case review
```

Synthetic tiny arrays are useful for numerical checks. They remain clearly labelled checks; the application and reported task metrics use the recorded dataset.

## Shared experiment rules

| Comparison | What I hold fixed | What I measure |
| --- | --- | --- |
| Optimizer | Dataset/split, feature transform, model, loss, initialization, batch policy and comparable tuning allowance | Validation quality, updates, examples processed, elapsed time and failures |
| Batch policy | Model/objective/transform and declared training budget | Gradient noise, throughput, convergence and repeatability |
| Learning rate | Everything except rate or its declared schedule | Slow learning, useful progress, oscillation and divergence |
| Loss | Model family, data and a separately fair tuning allowance | Common task metrics and error patterns; raw losses across different objectives are not a ranking |
| Model family | Feature availability, split and evaluation policy | Quality versus runtime, artifact size and complexity |

Full-batch, stochastic and mini-batch describe how the gradient is estimated. Momentum and Adam describe the update rule; the benchmark records both choices. Decision trees and random forests have their own fitting procedures, so I will not attach Adam or SGD to them. Gradient boosting's learning rate is not interchangeable with a linear model's optimizer step size.

Use a bounded pilot, then a small predeclared comparison instead of every possible Cartesian product. Fit preprocessing inside each training fold, choose configurations on validation data, and reserve test labels for the frozen final comparison. Report failed trials and variability, not just the best seed. Each detailed README defines a metric and data contract for its task.

## Implementation and ownership

Documentation navigation is shown below; project implementation files are omitted from this tree.

```text
projects/foundations/
├── README.md
├── bike-demand-planner/
│   └── README.md
├── ecommerce-purchase-predictor/
│   └── README.md
├── waste-material-classifier/
│   └── README.md
└── air-quality-forecaster/
    └── README.md
```

These five README files document the selected route. Each project owns its source, tests, environment, configs, data provenance and application as implementation proceeds; consult its README for what actually exists. This sequence-project update creates no runnable model files. In the classical ML stage, NumPy implementations will expose linear/logistic models, losses and optimizer state; scikit-learn supplies independent references and practical comparison models. PyTorch is a reference option there, not a neural-network requirement. In the vision/sequence stages, PyTorch supplies autograd/trainable layers for our own models/trainer, with explicit numerical probes and independent checks before larger runs.

The folders are ordinary directories tracked by `ai_projects`, with no nested Git repositories or submodules. Preparation records are [classical ML READMEs](../../docs/implementation/2026-10/classical-ml-foundations-project-readmes/00-resume.md), [vision README](../../docs/implementation/2026-10/waste-material-classifier-vision-project-readme/00-resume.md) and [sequence README](../../docs/implementation/2026-10/air-quality-forecaster-recurrent-sequence-project-readme/00-resume.md). Each project will get an implementation checkpoint at kickoff. Data, environments, models and generated runs stay ignored; source, configs, provenance and selected authored reports can be tracked.

Creating a README does not register CLI discovery or implement a collapsible TUI group. Shared launcher/TUI work belongs in `cli/` when requested; project scripts should remain directly runnable with their inputs visible in their main blocks. Bike Demand Planner's own README documents its separate scaffold commands; the new sequence project has no runnable commands yet.

## Revision alongside implementation

The [private revision playlist](https://www.youtube.com/playlist?list=PLO4z6AplXMtc) contains the agreed videos, including repeated explanations. The [Tutorial 0 outline](../../docs/tutorials/projects/p00/transformer-foundations/v1/project/course-outline.md) is the maths/reference route. Its displayed topic order is authoritative; legacy lesson filenames retain older names.

| When I need it | Tutorial 0 topics | Application |
| --- | --- | --- |
| First predictor | 1: linear algebra and affine maps | Feature vectors, weights, bias and prediction units |
| First training loop | 3, 7, 8, 9, 10: calculus, gradients, losses, optimizers and updates | Derive, implement, check and inspect an update |
| Data and validation | 4, 11, 23: probability, datasets and evaluation | Error distributions, split design, baselines and generalization |
| Resuming work | 24: checkpointing | Reload optimizer and data-order state as well as weights |

Attention, neural-network architecture and embeddings are not prerequisites for the two classical ML applications. For the vision application, refresh Tutorial 0 topics 5–10 and 21–24 as needed; convolution, pooling and image-specific contracts are introduced in its own project route. Attention/embeddings remain outside its scope.

For the sequence application, reuse calculus/gradient/optimization and data/evaluation/recovery topics; recurrence, gates, state handling, forecast windows and teacher forcing are explained at their first use in that application. Do not require a completed text model to begin numerical forecasting.

## What completing these stages means

- Each application runs from documented inputs through the exact saved preprocessing/model artifacts, with separate completion reviews for the classical ML, vision and sequence stages.
- Numerical checks, data/split validation and failure behavior have independent evidence.
- Benchmarks explain their controls, budgets, uncertainty and practical tradeoffs.
- Each project has a baseline comparison, final held-out evaluation and an honest failure analysis. Improvement is measured, not promised.
- I can explain a prediction, derive an update and diagnose a changed-condition failure.
- Local implementation, local tests, learner explanation, hosted recovery and external publication are recorded separately.

**Next action:** open [Bike Demand Planner](bike-demand-planner/README.md), agree on the prediction-time contract and work through its first small prediction. Then complete purchase prediction, vision classification and recurrent forecasting in that order before the existing embedding router. README preparation does not start model training.
