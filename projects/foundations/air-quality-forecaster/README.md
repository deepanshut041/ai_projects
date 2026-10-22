# Air Quality Forecaster

I want to build a sequence-learning application that uses past pollution and weather measurements to predict future PM2.5 concentrations. I will compare useful non-recurrent baselines with a vanilla RNN, LSTM and GRU, then investigate memory length, multi-step forecasting and stateful training.

This is the last selected foundations stage, after [Waste Material Classifier](../waste-material-classifier/README.md). It gives recurrence a real numerical task before the later text/embedding application. The emphasis is understanding temporal state and training behavior, not replacing a simple baseline merely because an LSTM is available.

**Status — 2026-10-15:** project folder and detailed README prepared. Dataset acquisition, environment, source code, models, training, benchmarks, application and hosted recovery are pending. Proposed splits, architectures and budgets require implementation/data evidence. Worked numerical probes are illustrations, not air-quality results.

[Foundations overview](../README.md) · [Previous application: Waste Material Classifier](../waste-material-classifier/README.md)

## 1. The application and its limits

The initial user is an analyst replaying historical monitoring data. At a selected station and forecast issue time, the application shows the observations available so far, baseline/model forecasts for future hours, and actual future measurements revealed separately for evaluation. A comparison view shows error by forecast horizon, training behavior and failure examples.

The initial target is PM2.5 concentration in **µg/m³**, not a composite AQI. These historical data do not establish present-day local conditions, a public health recommendation or a deployed warning system. The app's useful output is a labelled model forecast and its measured historical limitations.

The first model uses one station and predicts the next hour. Then it uses multiple observed features and forecasts the next 6/24 hours. Pooling multiple stations is an extension after the single-station pipeline works; forecasting unseen stations is a separate generalization question.

| Contract | Initial design |
| --- | --- |
| Issue time | The end of the latest observed hour `t`, with an explicit assumed data-arrival policy |
| Input history | Observations from `t-L+1` through `t`, in increasing time order, at one station |
| First target | PM2.5 at `t+1` |
| Multi-step target | PM2.5 at `t+1 ... t+Q`, with horizon `Q` stored in the artifact |
| Main tensor | Float input `[B,L,F]`, explicit missingness metadata and fixed feature order |
| Forecast output | `[B,Q]` in original units after the saved inverse transform |
| Available future inputs | Known calendar fields only; future measured weather/pollution is excluded |
| Missing future target | Never fabricated; excluded using a saved validity mask and reported denominator |
| Invalid input | Reject wrong station/schema/order, nonfinite processed values, insufficient history or a missing-data condition outside the declared policy |

For hourly replay, actual observations become available as the issue clock advances. They may enter a later forecast's history only after their assumed arrival time. A fixed-origin 24-hour forecast cannot consume the actual intervening PM2.5 values while generating its remaining hours.

## 2. Dataset and provenance

Use UCI's [Beijing Multi-Site Air Quality dataset, ID 501](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data). It contains hourly readings from 12 monitoring sites spanning 2013-03-01 through 2017-02-28, with pollution, matched meteorological measurements and missing observations. The published record lists 420,768 observations. Preserve attribution to Song Chen and the [dataset DOI](https://doi.org/10.24432/C5RK5G); UCI publishes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The acquisition phase will use the official `PRSA2017_Data_20130301-20170228.zip` archive, record its hash/source/access date and enumerate station files. Do not substitute ancillary example CSV files listed on the dataset page for the full station archive. Actual file counts, timestamp coverage, units, duplicate records and missingness must be audited locally.

Start with a station selected and frozen from the audited archive before looking at test outcomes. Record why it was selected, such as a training-period coverage requirement. Later choose whether the same architecture shares weights across stations or trains separate models; do not concatenate stations into one apparent continuous sequence.

### Candidate features

- Univariate first: historical PM2.5 plus a declared missingness/age-of-observation representation.
- Multivariate extension: past PM10 and other pollutants, temperature, pressure, dew point, rain, wind speed/direction and known calendar features.
- Encode direction/category values deliberately. Numeric category IDs are not automatically meaningful magnitudes; transformations and unknown-category policy are learned/fixed from training data.
- Exclude source row counters from predictors. Retain station/timestamp identities in the manifest, not as accidental time-order shortcuts.

Future weather forecasts are not supplied by these measured-weather records. A forecast service integration would need a new input contract and historical forecast-vintage data before being evaluated fairly.

## 3. Missing data, windows and chronological validation

### Inputs and labels need different treatment

Preserve a missing-value indicator and elapsed time since a valid observation for each selected measurement where useful. Proposed initial input policy: forward-fill only from the past within a bounded gap, then use a training-only fallback statistic while retaining the missingness/age fields. Finalize gap limits and long-gap rejection after the training-period audit.

No backfill from a later observation, centered rolling statistic or interpolation using future endpoints is allowed in the headline forecast pipeline. A missing timestamp must be restored explicitly to an hourly grid, not collapsed so two non-adjacent readings appear consecutive. The original dataset's timestamps and measurement availability are separate facts to validate.

Never impute a target and present it as a measured label. Training/evaluation masks identify which future targets actually exist. Missing-input masks, target-validity masks and sequence-padding masks have different meanings. An example with no valid future targets contributes no loss and is handled explicitly, not through division by zero.

### Split by target time before constructing examples

Proposed periods, subject to audited coverage:

| Partition | Target timestamps |
| --- | --- |
| Training | 2013-03-01 through 2015-02-28 |
| Validation | 2015-03-01 through 2016-02-29 |
| Final test | 2016-03-01 through 2017-02-28 |

All stations use the same calendar boundaries in a pooled time-generalization experiment. A training window's complete label horizon must remain inside training; it cannot straddle into validation. Apply the same rule within each later partition. Freeze window IDs with station, issue time, input bounds and target bounds. Do not randomly split overlapping windows.

Validation/test inputs may use genuinely available earlier observations, including history before the target partition starts. This is a rolling-origin forecast contract, not a hidden reset of the chronological boundary. Labels in different partitions remain disjoint; fitting preprocessing or using future observations remains forbidden. Overlap among histories and repeated horizons makes errors correlated, which affects uncertainty estimates.

Fit scalers, encoders, imputation fallback statistics and any feature selection on training data only, separately within any expanding-window tuning folds. Validation/test use those fitted transforms. Audit feature timestamps and imputation boundaries after transformations, not just at raw CSV split time.

## 4. The ordered learning/build route

We use the established explanation → assistant implementation/checks → code/results review cycle. Each phase has concrete inputs/outputs, a stopping point and a changed-condition question. I already know Python; the focus is sequence mathematics, PyTorch contracts and evidence.

| Phase | What I build and why | Completion evidence |
| --- | --- | --- |
| 0. Forecast contract | Define issue/arrival time, target/horizon, station scope and one recurrence calculation | Input/target timeline and independently checked arithmetic |
| 1. Data and windows | Acquire/audit data, create chronological partitions, masks and causal preprocessing | Provenance/hashes, gap audit and no future/label-boundary leakage |
| 2. Baselines | Persistence, daily-seasonal persistence, lagged linear model, then MLP | Per-horizon metrics and defensible fallback behavior |
| 3. Vanilla RNN | Explicit cell/unroll/BPTT before the library sequence layer | Hand/finite-difference gradient parity, shared-weight and state-shape checks |
| 4. LSTM and GRU | Explain gates and implement tiny cells, then use optimized library modules | Gate/state parity with exact framework conventions |
| 5. Controlled training | Compare context, hidden size, batch/rate/optimizer, clipping, losses and regularization | Paired budgeted studies, diagnostics and retained failures |
| 6. Multi-step forecast | Compare direct multi-output, recursive and recurrent encoder-decoder forecasts | Target alignment, teacher-forcing separation and free-running evaluation |
| 7. Sequence mechanics | Investigate padding/lengths, stacked cells, state carry/reset and truncated BPTT | Mask invariants, contiguous-stream sampler and explicit detach/reset behavior |
| 8. Generalization | Inspect horizon/season/missingness errors; optionally pool stations | Comparable baselines, class-free regression metrics and error examples |
| 9. Recovery and app | Save/recover complete state; build historical replay and benchmark views | Fresh-process parity, actual deleted-runtime Colab recovery and end-to-end application |
| 10. Final review | Freeze choices and evaluate held-out timestamps | Reproducible final report and changed-condition explanation |

Start with independent fixed-length windows and reset state for each example. Stateful streaming and variable lengths come after that safe baseline. Synthetic memory probes are numerical/learning diagnostics; application claims come from recorded monitoring data.

## 5. Why sequence models: establish baselines first

| Model | Question it answers |
| --- | --- |
| Persistence | Is copying the latest available PM2.5 already hard to beat? |
| Daily-seasonal persistence | Is the value at the same hour of the previous day useful? For up to 24 hours, use only the corresponding historical observation available at issue time |
| Lagged linear/ridge model | How much can a fixed set of past values and known calendar fields explain? |
| Flattened-window MLP | Do nonlinear features help without recurrent hidden state? |
| Vanilla RNN | Does sequential shared-state processing improve useful prediction? |
| LSTM / GRU | Do gating and memory control help at longer contexts or horizons? |
| Optional temporal CNN | Is recurrence necessary compared with a bounded causal convolutional reference? |

Baseline fallbacks under missingness must be predeclared and receive the same target-time evaluation mask as learned models. Compare one-step forecasts first, then identical multi-step horizons. A recurrent model is not successful merely because training loss decreases; persistence and lagged models remain visible throughout.

## 6. First session: a two-step RNN and backpropagation through time

For an illustrative scalar sequence `x=[1,0]`, let input weight `w=0.5`, recurrent weight `u=0.8`, bias `b=0` and initial hidden state `h0=0`:

```text
h_t = tanh(w*x_t + u*h_(t-1) + b)
h1  = tanh(0.5)        ≈ 0.462117
h2  = tanh(0.8*h1)     ≈ 0.353724
J   = (h2 - 0.6)^2 / 2 ≈ 0.030326
```

The target `0.6` and values are invented unitless numbers, not standardized pollution readings from a run. This probe uses `h2` itself as the prediction, without an extra output head.

With `d2 = (h2-0.6)*(1-h2²)`, backpropagate through both uses of the shared cell:

```text
dJ/dw = d2 * u * (1-h1²)          ≈ -0.135560
dJ/du = d2 * h1                   ≈ -0.099569
dJ/db = d2 * (1 + u*(1-h1²))      ≈ -0.351022
```

The first stopping point is drawing the two copies of the cell, identifying shared parameters, and explaining why the bias receives contributions from both times. Verify derivatives using finite differences on the full unroll and an independent autograd implementation before larger training.

In longer sequences the gradient contains products of recurrent weights and activation derivatives. That can shrink or amplify the influence of an earlier input. Inspect gradients by lag; do not infer memory quality solely from the final loss. Gradient clipping limits an update's input gradient norm but does not restore information already lost through vanishing gradients.

## 7. LSTM and GRU: gates, state and exact conventions

### LSTM

The LSTM maintains hidden state `h` and cell state `c`. For a reference with separate affine inputs implicit in each gate:

```text
i_t = sigmoid(input_gate_affine(x_t, h_prev))
f_t = sigmoid(forget_gate_affine(x_t, h_prev))
g_t = tanh(candidate_affine(x_t, h_prev))
o_t = sigmoid(output_gate_affine(x_t, h_prev))
c_t = f_t*c_prev + i_t*g_t
h_t = o_t*tanh(c_t)
```

Products are elementwise. Work a scalar gate example first: `c_prev=1`, `f=.8`, `i=.5`, `g=.25`, `o=.5` gives `c=.925` and `h=.5*tanh(.925)`. Explain retaining, adding and exposing information separately. A forget gate below one can still attenuate memory; an LSTM does not guarantee unlimited useful context.

Inspect gate saturation and cell/hidden-state norms. If trying a positive forget-gate bias, record the effective bias: in the planned framework there are separate input and recurrent bias vectors, and setting both to one gives a different total than setting one to one.

### GRU

Use the exact chosen framework's reset/update convention, rather than assuming all published GRU equations are interchangeable. The intended PyTorch-style reference is:

```text
r = sigmoid(W_ir*x + b_ir + W_hr*h_prev + b_hr)
z = sigmoid(W_iz*x + b_iz + W_hz*h_prev + b_hz)
n = tanh(W_in*x + b_in + r*(W_hn*h_prev + b_hn))
h = (1-z)*n + z*h_prev
```

The reset gate multiplies the hidden candidate affine result here; this is not generally equivalent to multiplying `h_prev` before its matrix multiplication. A scalar example with `h_prev=.4`, `z=.75`, `r=.5`, input-candidate affine `.2` and hidden-candidate affine `.6` gives `n=tanh(.5)` and `h≈.415529`.

Compare a tiny manually unrolled cell and the library cell with identical weights, gate ordering, biases and inputs. Then use the optimized sequence module for task training. GRU has no separate LSTM-style cell state; do not invent one in the checkpoint schema.

## 8. Tensor shapes and a proposed small architecture

Use batch-first input/output, but keep the library's layer-first hidden-state convention. Let batch size be `B`, history length `L`, input features `F`, hidden width `H`, recurrent layers `N` and forecast horizon `Q`.

| Tensor | Unidirectional, no projected LSTM |
| --- | --- |
| Input | `[B,L,F]` |
| Sequence output | `[B,L,H]` |
| Final hidden state | `[N,B,H]` |
| LSTM final cell state | `[N,B,H]` |
| Last-valid-state head input | `[B,H]` |
| Direct output / target | `[B,Q]` |
| Valid-target mask | `[B,Q]` boolean |

The first proposal is one unidirectional layer with hidden width 32 and a linear forecast head, initially `Q=1`. Input width is set by the frozen feature manifest, not inferred from a fixed number in a README.

For an illustrative `F=8`, `H=32`, `Q=1`, and two bias vectors per gate group in the selected PyTorch convention:

| Cell | Recurrent parameters | Including a biased 32→1 output head |
| --- | ---: | ---: |
| Vanilla RNN | `H*F + H*H + 2H = 1,344` | 1,377 |
| GRU | `3H*(F+H+2) = 4,032` | 4,065 |
| LSTM | `4H*(F+H+2) = 5,376` | 5,409 |

These are analytic counts for the single-layer, unidirectional, non-projected examples, not measured implementation sizes. Equal hidden width is not equal parameter count. Compare equal-width and approximately parameter-matched alternatives separately, with runtime and activation/state memory shown.

Padding requires the final **valid** state, not the output at a padded timestep. Use packed sequences or a deliberately masked recurrence with restored batch order. A missing measurement is still a real timestamp; it is not padding to be discarded. `batch_first=True` does not change the layer-first shape of hidden/cell states.

## 9. One-step, multi-step and encoder-decoder forecasts

| Strategy | Training/output | Evaluation boundary |
| --- | --- | --- |
| One-step | Encode history and predict `t+1` | Evaluate the next observed target only |
| Direct multi-output | One history encoding/head predicts all `Q` future values | No actual future target enters the input at any horizon |
| Recursive | Repeatedly apply a one-step forecasting policy | Feed previous predictions, not true future PM2.5; other future covariates need a declared causal policy |
| Recurrent encoder-decoder | Encode past state; decoder emits future steps | Decoder uses predicted previous values and known future calendar inputs at inference |

Do not compare a teacher-forced decoder score with another model's free-running forecast as if they were the same task. Teacher forcing may supply previous true targets during training only. Evaluate free-running horizons; scheduled sampling is a conditional extension with its sampling schedule saved and separate evidence.

The univariate recursive model is the first rollout reference. A multivariate model cannot quietly consume actual future wind/rain or other pollutant values when the clock has not reached them. Choose and report a strategy such as past-only encoding with direct outputs; if holding unknown covariates constant during recursion, name that assumption and test its consequence.

The first encoder-decoder has no attention. Its fixed-state bottleneck is a useful limit to inspect before later Transformer/embedding work.

## 10. State reset, streaming and truncated BPTT

The initial supervised dataset contains independent windows. Reset hidden/cell states at the start of every window; batches may shuffle training windows only after their chronological partition is fixed. Never carry state from one shuffled example or station to another.

For the later stateful study, each stream's chunks must be contiguous, ordered and non-overlapping. Retain a stable mapping from stream/station to batch slot and state. Detach state at declared truncated-BPTT boundaries; reset when a stream changes, crosses a fold boundary, has an unsupported gap or begins a new independent evaluation episode. Hidden state for the last smaller batch must use its actual size.

Do not feed overlapping windows while carrying their final state, which would process the overlapping history twice. Compare stateful streaming with reset-window training as distinct input/training policies, not just a single detach switch.

Truncated BPTT preserves state values across chunks while limiting how far gradients flow. Detaching the state is different from zeroing it. Finite differences through a full unroll check the full derivative; they should not be misrepresented as a reference for an intentionally stop-gradient computation. Use an independent matching-detach reference for the latter.

Bidirectional encoding of a fully observed historical window is a possible offline ablation: both directions still see only data through `t`. It cannot consume future target timestamps or act as an equivalent one-pass streaming update. Keep it outside the primary unidirectional comparison until that distinction is clear.

## 11. Experiment matrix and fair comparisons

| ID | Study | Proposed comparison | Learning question |
| --- | --- | --- | --- |
| R1 | Baseline versus sequence models | Persistence, lagged linear/MLP, RNN, LSTM, GRU | Is recurrence/gating useful beyond strong history baselines? |
| R2 | History length | 24, 72, 168 observed hourly steps | Does longer history help or add optimization/data cost? |
| R3 | Forecast horizon | 1, 6 and 24 hours | How does error evolve with distance from the observations? |
| R4 | Inputs | PM2.5 only versus past multivariate readings/masks/calendar | Which information improves a forecast without future leakage? |
| R5 | Capacity | Width 16/32/64 and one versus two layers, in separate studies | Underfitting, generalization and state/compute cost |
| R6 | Optimizer/rate/schedule | SGD, momentum, Adam/AdamW; bounded rate choices and one schedule at a time | How do updates affect sequence optimization? |
| R7 | Initialization/gates | Recurrent initialization and explicit forget-bias variants | Gradient transport, gate saturation and effective memory |
| R8 | Clipping | Disabled versus validation-selected global-norm clipping | Can exploding gradients be diagnosed/controlled without hiding other failures? |
| R9 | Objectives | MSE, MAE, Huber; quantile loss later | Typical error, spikes, robustness and asymmetric error preferences |
| R10 | Regularization | Weight decay, head dropout, layer count and early stopping separately | Which mechanism improves held-out behavior? |
| R11 | Missingness | Declared causal fill baseline versus masks/age-aware inputs | How does the model behave after gaps and outages? |
| R12 | State/gradient span | Independent windows versus contiguous stateful streams; chosen TBPTT span | Difference between memory values and gradient history |
| R13 | Forecast strategy | Direct, recursive and no-attention encoder-decoder | Error accumulation and training/inference mismatch |
| R14 | Temporal diagnostics | Lag ablations or a fixed within-history permutation control | What evidence shows dependence on useful ordering/context? |
| R15 | Station scope, extension | One station, pooled seen stations, separately held-out station | Time generalization versus transfer to a new monitoring site |

All core mechanisms have a planned study, but not every setting is mandatory and no full Cartesian sweep is intended. Freeze one primary question and reference configuration at a time. A permutation control changes the task structure; it is diagnostic, not a deployable training recommendation.

In optimizer comparisons hold data/windows, feature transforms, model, objective, initialization, batch policy and comparable tuning allowance fixed. In cell comparisons report equal-width and parameter-aware results. When context changes, align eligible forecast origins across models or explicitly report the resulting data-coverage difference. Keep missing-target evaluation masks common when comparing alternatives on the same task.

Report updates, sequences, valid target elements, observed timesteps processed and elapsed time. Equal epochs do not establish equal compute for different history lengths/horizons. Seeds proposed for repeat trials are `17,29,43` after a one-seed pilot. Distinguish seed variability from uncertainty in correlated time-series evaluation.

Proposed pilot: a small chronological training subset, one station, FP32 and at most 100 updates. Investigate a bounded log-scale rate range, then declare main trial budgets and rates from measured behavior before test evaluation. Session caps are 60 minutes or 1,000 additional updates; the total trial/schedule continues across resumed sessions. Larger contexts/station counts require measured time/memory fit.

## 12. Losses, metrics and uncertainty

For a valid-target mask `m` and per-element loss `ell`, the initial reduction is `sum(m*ell)/sum(m)` over valid targets, with an explicit empty-mask policy. Avoid multiplying an already-NaN residual by zero; gather valid entries or replace invalid inputs before computing loss. If applying horizon weights, define the weighted valid denominator separately.

MSE emphasizes large misses, MAE prioritizes absolute error and Huber changes the treatment of large residuals. Define scales and Huber threshold in the declared training space; if standardizing the target, inverse-transform before application/task metrics. A lower Huber number is not comparable with a larger MSE number. Freeze negative-output postprocessing on validation data and use it in both serving and final metrics; report its effect rather than silently clipping only the UI.

Primary metric: **MAE at the one-hour horizon in µg/m³**. Report RMSE and mean signed error, plus a per-horizon MAE/RMSE table for multi-step models. An aggregate across horizons must state its weights and number of valid station/issue-time/horizon targets. Avoid MAPE as the headline metric with low/zero concentrations.

Inspect errors by season/hour, historical concentration bands, missing-input age and forecast horizon, with counts. Define bands from training data or a declared non-medical analytical rule; no invented AQI thresholds or health claims. Target spikes are observed concentration errors, not automatically sensor faults.

Overlapping histories, repeated predictions of a target timestamp at different horizons, serial dependence and shared weather make errors correlated. Use a defensible time-block/station resampling procedure if reporting uncertainty intervals, not an IID bootstrap of every window. Changing station scope also changes the population being evaluated.

A quantile extension evaluates pinball loss, empirical coverage and interval width on disjoint calibration/test periods. A pair of output quantiles is not automatically a calibrated prediction interval. Any peak-review threshold is an analyst-facing hypothetical policy until its purpose, units, validation selection and costs are established.

## 13. Correctness and failure checks

- Independently check the scalar recurrence and full-BPTT derivatives, then manual RNN/LSTM/GRU cells against matching library weights/gate/bias conventions. Proposed small-fixture float64 tolerances are `atol=1e-6, rtol=1e-5`, finalized with measured reference behavior.
- Check batch-first outputs versus layer-first states, one/two layers, unequal batch/sequence/feature widths, and a smaller final batch. Count actual parameters separately from state buffers/activation memory.
- Modify observations after an issue time and prove its forecast cannot change. Check horizon off-by-one errors, target partition crossings, station boundaries, causal fills and inverse transforms.
- Check missing targets, all-invalid batches, long input gaps, unknown wind categories, duplicated timestamps, chronology violations and nonfinite processed inputs.
- Verify padding/packing does not change a sequence's valid outputs or last-valid state; missing actual hours must not become discarded padding.
- Test state reset between independent sequences/stations and correct carry/detach in contiguous streams. Verify no evaluation state is inherited accidentally from training.
- Overfit a tiny valid training set before adding regularization; log gradients by timestep, recurrent weight/gate/state statistics and pre/post-clipping norms.
- In PyTorch recurrent modules, the dropout argument applies between stacked layers rather than automatically regularizing a one-layer cell. Test head dropout separately; do not name an ineffective setting as a successful ablation.
- Evaluate multi-step decoders without true future targets. An accidentally teacher-forced evaluation must be caught by an input/causality check.
- Verify fresh-process forecasts and interrupted/resumed stateful updates. Retain invalid-state/failed-save evidence rather than retrying as an unlabelled new run.

## 14. Application, artifacts and eventual layout

The planned local application has a timeline replay, forecast comparison and error/benchmark browser. Select station/issue time/horizon, inspect the past values/missingness, show baseline/model forecasts, then reveal later actual observations. Display artifact/data identity and a measured-versus-predicted distinction; keep unsupported live-data promises out of the user flow.

Run every model at the same forecast origin with its declared context and availability policy. A historical playback animation does not establish streaming deployment. Training diagnostics and technical benchmark records can live in an analyst view rather than appearing in the ordinary prediction path.

Proposed future files; only README.md is created now:

```text
air-quality-forecaster/
├── README.md
├── pyproject.toml / uv.lock
├── prepare_data.py / train.py / compare.py / inference.py
├── configs/                         # CPU, Colab and named study profiles
├── src/air_quality_forecaster/
│   ├── data.py / windows.py / transforms.py
│   ├── models/                      # lag baselines, MLP, RNN, LSTM, GRU, decoder
│   └── trainer.py / evaluation.py / checkpoint.py
├── tests/                           # numeric, timeline, masks, state and recovery
├── notebooks/colab.ipynb             # thin, output-cleared common-trainer entry
├── app/
└── docs/                            # provenance, experiment and implementation records
```

Direct runnable files will declare inputs in their main blocks. The folder is an ordinary ai_projects directory, not a nested repository. CLI/TUI integration and runnable commands are not implemented by this plan; shared terminal tooling remains in `cli/`.

Inference artifacts contain architecture/cell conventions, weights, feature order/scalers/imputation policy, station policy, history/horizon, units, output transforms and source/dependency/data identity. Stateful inference additionally needs a versioned stream state with station, last timestamp and compatible observation-processing identity; a new independent request must reset state or explicitly select a validated stream.

Each run records split/window/target-mask hashes, feature/missingness policy, model/gates, loss/reduction, optimizer/schedule/clipping, batch and TBPTT policies, seed/generators, update/sequence/timestep counts, validation choices and failures. Timing fixes device/precision/context/horizon/batch size, separates preprocessing from model time, uses warmup and synchronized accelerator boundaries, and reports repeated latency/throughput and memory rather than one dispatch time.

## 15. Compute and complete recovery

Start numerical checks and a short one-station pilot on local CPU; measure real forward/backward, step time, memory and checkpoint size before selecting the Colab profile. A small parameter count does not guarantee cheap long-sequence training. Keep the same importable model/trainer locally and in a thin Colab Free notebook; pin actual compatible dependencies at kickoff.

Follow the [workspace persistent recovery contract](../../../docs/guide/COLAB.md), which this project adopts explicitly. Save after completed updates with no pending gradient accumulation. In addition to parameters, optimizer moments, schedule/scaler, progress, RNG, sampler and monitoring state, record:

- Window manifest and chronological split identity, feature/target transforms and units.
- Stateful hidden/cell tensors where applicable, stream/station-to-batch mapping, next timestamps/chunks and detach/reset policy.
- Data-order cursor and missingness/imputation context needed to reproduce the next processed observation.

For the first parity drill use deterministic CPU settings, one worker and controlled generators. Stateful continuations require the same stream topology; a changed batch-to-station mapping cannot be silently accepted. Stateless-window checkpoints reset per-example recurrent state but still restore sampler/progress correctly.

Write immutable persistent generations, validate hashes/loadability and write completion last, retaining a valid fallback. Test incomplete writes, corruption and incompatible data/state. A pointer is advisory; select a verified compatible generation and report rollback.

Demonstrate uninterrupted updates versus stop/fresh-process/reload/continue locally. Then actually delete the Colab runtime, reacquire exact source/dependencies/data, restore persistent state and advance the same run. Save every 5 minutes or 100 completed updates and on normal stop, with shorter recovery drills and measured rollback. CPU/GPU/version-changing continuation is distinct from same-environment exact parity. No hardware fit, training, hosted recovery or paid resource use has been verified here.

## 16. Completion and the handoff to embeddings

- [ ] Forecast issue/arrival policy, units, station scope, windows, horizons and causal input rules are enforced.
- [ ] Data provenance, gap audit, chronological split/window IDs and target-validity masks are reproducible.
- [ ] Persistence/seasonal/lagged baselines and complete RNN/LSTM/GRU models run through one train/evaluate/save/load path.
- [ ] Manual cells, BPTT gradients, shapes/counts, gate/state behavior and causal/missingness invariants have independent checks.
- [ ] Context, horizon, model, optimizer/rate/clipping, objective, regularization, state/TBPTT and forecasting-strategy studies use declared controls/budgets and retain failures.
- [ ] Direct/recursive/encoder-decoder evaluation does not use actual future targets or measured future covariates.
- [ ] Final held-out per-horizon/slice errors, baseline gaps, limitations and measured resource costs are reported.
- [ ] Historical replay/error review works from the exact saved artifact; live forecasting/health claims remain separate capabilities.
- [ ] Local stateful/stateless recovery and fresh-runtime persistent Colab recovery have distinct evidence.
- [ ] I can trace hidden/cell states, derive a shared-weight gradient, explain a gate/detach decision and diagnose a changed timeline or missingness condition.

There is no invented accuracy target or promise that LSTM beats persistence. Agree practical quality/runtime requirements after the baselines/pilot, then freeze the final comparison. Completing an experiment, meeting application requirements, explaining the mechanism and proving hosted recovery remain separate outcomes.

After this final foundations stage, continue the existing [Support Ticket Router](../../transformer-labs/support-ticket-router/README.md) for learned word embeddings/Word2Vec, intent classification and tokenizer comparisons. This sequence project uses continuous measurements and explicit categorical encodings; it does not implement that router or claim its later text/tokenizer gates.

## References and first action

Primary sources inspected on 2026-10-15:

- [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data): real data, fields/units, period, missing observations and license. Actual archive audit is pending.
- [PyTorch RNN](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html): shared recurrence and sequence/state conventions.
- [PyTorch LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html): gate order, separate hidden/cell state, bias and shape semantics.
- [PyTorch GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html): reset/update/new conventions and framework-specific candidate calculation.
- [Tutorial 0 outline](../../../docs/tutorials/projects/p00/transformer-foundations/v1/project/course-outline.md): calculus, probability, gradients, losses, optimization, datasets/evaluation and recovery. Recurrent-specific mechanisms are taught here at first use.

The inspected PyTorch 2.14 references are documentation identities, not installed-runtime proof or a frozen dependency choice. Pin compatible versions and validate manual/reference parity at implementation kickoff.

**First action when this stage starts:** finish the vision application's review, agree the issue-time/one-station/one-hour contract and explain the two-step recurrence. Then acquire/audit the official data, freeze chronological windows and establish persistence before implementing recurrence.
