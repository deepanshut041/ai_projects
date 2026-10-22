# Waste Material Classifier

I want to build a vision application that takes a waste image, estimates its material category and helps me inspect the model's mistakes. I will begin with a fully connected neural network, build and train my own CNN, investigate the training choices, then compare residual and pretrained CNNs.

This is my vision deep learning stage after the two [classical ML applications](../README.md). The goal is to understand neural networks and CNNs through a complete application and controlled experiments. RNNs, LSTMs, text, Word2Vec and Transformers belong to later projects.

**Status — 2026-10-15:** project folder and in-depth README prepared. Dataset/weight acquisition, environment, source code, models, training, experiments, application and Colab recovery are pending. All architectures, split ratios, budgets and study settings here are proposed; numerical examples are independently checkable illustrations, not task results.

[Foundations overview](../README.md) · [Prior application: E-commerce Purchase Predictor](../ecommerce-purchase-predictor/README.md)

## 1. What I am building

The initial user is an analyst reviewing waste photographs and material-classification errors. The application will accept a supported image, show its predicted category and top alternatives, and expose the model/version used. A review view will compare original and transformed images, mistakes by class, model quality, size and latency.

The task is **one material label per image**, using the dataset's categories. It does not locate every object in a pile, segment individual materials, decide local recycling eligibility or operate sorting machinery. A material label is not automatically a disposal instruction. Detection, segmentation and hardware control would need different requirements and evidence.

The first usable application supports historical evaluation images and manual image input. Uploaded photographs are outside the measured benchmark unless independently labelled and evaluated; plausible-looking outputs are not evidence of generalization.

| Contract | Initial design |
| --- | --- |
| Input | One decodable image; validate format, pixel dimensions and decoded size |
| Internal image | RGB with an explicit orientation/resize policy; float tensor `[B,3,H,W]` |
| Target | Integer class ID in a saved nine-class mapping |
| Model output | Nine finite logits per image, `[B,9]` |
| Display | Predicted class and top-three model scores; scores are not advertised as calibrated confidence without evaluation |
| Review policy | A validation-selected low-score policy can request review; it is not a guarantee of unknown-class detection |
| Error input | Clear rejection of corrupt/unsupported input or an incompatible model/schema |

No application/runtime files or launch commands exist yet. This document defines what the later implementation must deliver.

## 2. Data source, provenance and split discipline

Use [UCI RealWaste, dataset 908](https://archive.ics.uci.edu/dataset/908/realwaste), introduced by Sam Single, Saeid Iranmanesh and Raad Raad. UCI describes 4,752 real landfill images in nine material classes, released at 524×524 pixels under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution and the [dataset DOI](https://doi.org/10.24432/C5SS4G).

| Published label | Published count |
| --- | ---: |
| Cardboard | 461 |
| Food Organics | 411 |
| Glass | 420 |
| Metal | 790 |
| Miscellaneous Trash | 495 |
| Paper | 500 |
| Plastic | 921 |
| Textile Trash | 318 |
| Vegetation | 436 |
| **Total** | **4,752** |

These counts are source metadata, not a local data audit. Acquisition will use the official image archive, not assume a tabular importer returns decoded image data. Record source/access date, archive and per-image hashes, actual dimensions, decode failures, labels, class counts and license before freezing the manifest. Inspect representative images and labels; do not silently discard ambiguous examples or relabel the test set after seeing model errors.

### Split before augmentation

1. Audit exact duplicates by content hash and review candidate near-duplicate images. Perceptual similarity alone is not proof that two images show the same object.
2. Use object/scene/capture groups when supplied or defensibly established; keep each group in one partition. If scene identity is unavailable, state that a file-level holdout does not prove new-site or new-object generalization.
3. Proposed proportions are 70% training, 15% validation and 15% final test, stratified by class where compatible with grouping. Freeze actual assignments, seed, counts and the grouping policy after the audit.
4. Split original images first. Every crop, resized cache and augmented copy inherits its original's partition. Augmentations never become independently split observations.
5. Compute normalization statistics and class weights from training originals only. No validation/test image influences a fitted feature transform, selection rule or sampler.
6. Use training folds/subsets for tuning studies; use validation for declared model/checkpoint choices. If fitting a calibrator or review threshold, partition validation or predeclare cross-fitting so fitted policies are not evaluated on their own fitting records.
7. Keep test labels untouched until configurations, transforms, checkpoints, calibrator and review policy are frozen.

The small dataset and uneven class counts make overfitting and split contamination practical concerns. Repeated training seeds measure training variability, not proof of performance at a different landfill or under a different camera.

## 3. How I will learn and implement it

I already know Python and will use the prior ML projects' gradient, optimizer, evaluation and artifact concepts. At each phase we define the problem and inputs/outputs, work one small calculation, the assistant writes code and meaningful checks, and we review the results together. I predict a changed condition and explain the evidence before moving on.

PyTorch supplies tensor operations, autograd and trainable layers. I will design the MLP/CNN and shared trainer rather than begin with a pretrained model as the whole project. A tiny explicit convolution and selected manual gradients expose the mechanism; reimplementing a full tensor/autograd engine is outside scope.

```text
image + saved label mapping -> decode/transform -> baseline or learned model
    -> nine logits -> training loss -> gradients -> optimizer update
    -> validation + error analysis -> frozen artifact -> application/review
```

### Required build route

| Phase | Problem and output | Completion evidence |
| --- | --- | --- |
| 0. Task and first patch | Establish single-label material recognition; work the convolution example below | Input/target/label contract and independent arithmetic |
| 1. Data and baselines | Audit and split originals; build prevalence/majority and simple image-feature baselines | Reproducible manifest, duplicate/group audit and baseline metrics |
| 2. Fully connected network | Train an image MLP; inspect nonlinear layers and gradients | Correct shapes/CE, gradient probes, one-batch overfit and learning curves |
| 3. Small CNN | Replace flattened dense connections with shared local filters | Convolution/pooling checks, parameter count and complete forward/backward/training |
| 4. Architecture and optimization | Investigate activations, initialization, depth/width, pooling, normalization and optimizer/rate behavior | Controlled comparisons and explanations of failures |
| 5. Generalization and losses | Compare augmentation, regularization and task-appropriate loss changes | Frozen split, separate ablations and common metrics |
| 6. Residual comparison | Compare a deeper plain network with a matched residual version | Shape-safe skip paths, gradient/activation diagnostics and matched-budget results |
| 7. Transfer comparison | Train a pretrained CNN head, then fine-tune selected blocks | Recorded weight/transform identity, frozen-state checks and clearly separated benchmark |
| 8. Robustness and review | Study mistakes, controlled corruptions, calibration and coverage versus error | Class/slice counts, example failures and an honest review policy |
| 9. Recovery and application | Save complete state, verify reload/resume and integrate image/model/error views | Fresh-process parity, actual Colab recovery proof and end-to-end application |
| 10. Final evaluation | Freeze choices and compare on held-out images | Reproducible final report plus a changed-condition explanation |

The first implementation milestone ends with a correct data pipeline, a baseline and an understood MLP. Each later model remains runnable; I should not lose the earlier baselines while adding a stronger model.

## 4. First session: what a convolution actually computes

For one channel, use this invented 3×3 input and 2×2 filter, with stride 1, padding 0 and bias 0:

```text
X = [[1,2,3],      K = [[1,0],
     [4,5,6],           [0,1]]
     [7,8,9]]

top-left output = 1×1 + 2×0 + 4×0 + 5×1 = 6
output          = [[6,8],
                   [12,14]]
```

This is cross-correlation: the filter is not flipped, matching the operation called convolution in the planned PyTorch layer. One filter is reused at all four positions. For this illustration, the objective is the sum of outputs, `J=40`; it is a numerical probe, not our classification loss.

```text
gradient of J with respect to K = [[12,16],
                                   [24,28]]
gradient with respect to bias   = 4
```

Those gradients add contributions from every position where the shared parameter was used. If just `K[0,0]` increases by 0.1, `J` increases by 1.2. The first stopping point is explaining the filter reuse, output shape and gradient sum before checking against loops/finite differences/autograd.

For a spatial input size `H`, kernel `K`, padding `P`, dilation `D` and stride `S`:

```text
H_out = floor((H + 2P - D(K-1) - 1)/S + 1)
```

Apply it independently to height and width, including rectangular checks. With ordinary groups=1, a biased convolution has `C_out × (C_in × K_h × K_w + 1)` trainable parameters. Parameter count, multiply-add work and activation memory are different quantities.

## 5. Fully connected neural network foundations

The first image network is a proposed `64×64` RGB MLP:

```text
[B,3,64,64] -> flatten [B,12288]
    -> Linear(12288,128) -> ReLU -> Linear(128,9) -> logits [B,9]
```

With biases in both linear layers, this has **1,574,153 parameters**. This is an analytic count for the proposed architecture, not an implemented model measurement. I will inspect registered parameters and compare the actual count when code exists.

The MLP phase covers:

- Affine layers, learned hidden features and why stacked affine layers without nonlinearities collapse into one affine map.
- ReLU first, then controlled sigmoid/tanh or GELU comparisons; saturating activations and inactive units need inspection rather than slogans about one universal best activation.
- Forward values, computation graphs, chain rule and batch-averaged gradients. Check a small complete network against manual derivatives, finite differences and autograd.
- Initialization, symmetry, fan-in/fan-out and activation-dependent signal scale. Derive the required assumptions before applying Xavier/He-style rules.
- Training order: clear gradients, forward, loss, backward, inspect/update. Track model mode separately from gradient recording.
- Why flattening uses fixed pixel coordinates and many weights; its limitations become a question to test, not a guaranteed CNN victory.

Use a deterministic tiny set of training images with augmentation/dropout disabled for the first overfit-a-batch check. If a sufficiently capable network cannot fit it, investigate labels, transforms, loss, gradients and updates before scaling training.

## 6. Small CNN: shapes, sharing and capacity

A proposed scratch reference at the same image resolution is:

| Step | Output shape | Trainable parameters |
| --- | --- | ---: |
| RGB input | `[B,3,64,64]` | 0 |
| 3×3 convolution, 3→16, stride 1/padding 1; ReLU | `[B,16,64,64]` | 448 |
| 2×2 max pool, stride 2 | `[B,16,32,32]` | 0 |
| 3×3 convolution, 16→32, stride 1/padding 1; ReLU | `[B,32,32,32]` | 4,640 |
| 2×2 max pool, stride 2 | `[B,32,16,16]` | 0 |
| 3×3 convolution, 32→64, stride 1/padding 1; ReLU | `[B,64,16,16]` | 18,496 |
| Global average pooling | `[B,64]` | 0 |
| Linear, 64→9 | `[B,9]` | 585 |
| **Total** | | **24,169** |

This reference has biases and no BatchNorm/dropout. Adding normalization, changing biases or replacing the head changes its count. It is not a reproduction of a named paper architecture.

Teach channels, feature maps, locality and shared weights, then stride/padding/dilation, receptive field, pooling and the classification head. Downsampling can discard useful detail; global pooling changes the way position information reaches the head. A convolution is not automatically invariant to every image transformation, especially after padding and striding.

Compare MLP and CNN at equal input resolution and disclose their parameter/runtime difference. Add a secondary approximately parameter-matched comparison if we want evidence about architecture beyond model size; one unmatched accuracy result cannot isolate locality as the only cause.

Use smaller `16×16` fixtures for CPU correctness before full-image trials. Full-resolution performance is not established by tiny checks.

## 7. Classification loss and target contracts

For the initial single-label task, logits are `[B,9]` and labels are integer IDs `[B]`. The model returns logits; stable cross-entropy consumes them directly. Softmax is for interpreting scores, not an extra training layer before a logits-based CE implementation.

With logits `z`, target ID `y` and stable log-sum-exp:

```text
loss = logsumexp(z) - z[y]
gradient_z[j] = softmax(z)[j] - indicator(j == y)
```

Nine zero logits give uniform score `1/9` and unweighted per-image loss `log(9) ≈ 2.197225`. For that image the target-logit gradient is `-8/9` and each other gradient is `1/9`. For a batch mean, divide each image's contribution by the actual batch size. Test a smaller final batch and extreme logits independently.

| Loss study | Question | Controls |
| --- | --- | --- |
| Ordinary cross-entropy | Can the network learn useful nine-class features? | Required starting objective |
| Class-weighted CE | Do minority-class errors improve, and what happens to calibration? | Weights from training originals; record normalization and common evaluation metrics |
| Label smoothing | Does reducing target certainty improve validation behavior? | Small validation-selected epsilon; explicit target distribution |
| Focal loss, conditional extension | Are persistent easy/hard-example effects unresolved by the simpler baselines? | Only after diagnosing that issue; tune gamma/weights separately |

For class-index targets, PyTorch's weighted-mean CE normalizes by the sum of selected class weights; a manually implemented `mean(weight×loss)` is a different convention. Keep the chosen reduction fixed and verify gradient scale before comparing rates. First study weighting and smoothing separately; combined semantics need their own explicit check.

With smoothing epsilon `ε` and nine classes, use `q = (1-ε) one_hot(y) + ε/9` for the declared reference. The target mass is `1-ε+ε/9`, not simply `1-ε`. Compare losses using common macro-F1, ordinary evaluation NLL and other fixed task metrics, not numeric training objectives on different scales.

Binary BCE and multilabel sigmoid objectives are not the default for these mutually exclusive labels. A later binary warmup would be a separately defined subset task, not an interchangeable nine-class loss.

## 8. Experiments: questions, controls and priorities

The required route visits all core areas below, but not every possible combination of settings. Each study has one primary question, a declared reference model and a bounded tuning budget. Run a one-seed pilot, then predeclare the main comparison and failure/stopping rules before seeing final test results.

| ID | Study | Starting comparison | What I should learn |
| --- | --- | --- | --- |
| V1 | Classical image baseline → MLP → CNN | Majority/prevalence; simple color/texture features with a linear classifier; same-resolution learned models | Whether learned spatial features add useful quality over simple cues |
| V2 | Activation and initialization | ReLU with appropriate initialization; then one activation or initialization change at a time | Saturation, inactive units, activation/gradient scale and learning failures |
| V3 | Depth and width | Small reference versus one deeper or wider variant | Capacity, underfitting, overfitting, parameter and activation cost |
| V4 | Kernels and downsampling | 3×3 versus 5×5; max versus average pooling; a strided-convolution variant in a separate trial | Receptive field, detail loss and computation; these alternatives are not exactly equivalent |
| V5 | Classification head | Flattened dense head versus global average pooling | Parameter count, position sensitivity and generalization |
| V6 | Normalization | No normalization versus BatchNorm under a fixed block-order convention | Distribution/scale behavior and the effect of train/eval running statistics |
| V7 | Optimizer | SGD, momentum-SGD, Adam and AdamW with explicit decay policy | Convergence, history, regularization semantics and quality/runtime tradeoff |
| V8 | Rates and schedules | Small rate sweep; constant versus cosine decay or validation-driven reduction in separate studies | Slow learning, divergence and later-stage refinement |
| V9 | Batch size | Proposed 16, 32 and 64 where measured memory permits | Throughput, update noise and BatchNorm interaction |
| V10 | Regularization | Weight decay, dropout and early stopping separately | Which mechanism reduces validation error and which merely changes training loss |
| V11 | Image augmentation | Deterministic resize baseline, then justified transforms in separate ablations | Label-preserving variation versus destructive augmentation |
| V12 | Loss and imbalance | CE, training-only class weighting, smoothing; separate weighted sampling study if needed | Minority-class recall, overall quality and probability tradeoffs |
| V13 | Data amount and resolution | Nested training subsets; proposed 64/128-pixel scratch inputs | Learning curves, detail, memory and runtime limits |
| V14 | Residual path | Matched deeper plain CNN versus residual CNN | Shape alignment, gradient transport and whether a skip path helps this task |
| V15 | Transfer learning | Pretrained backbone frozen, selected blocks fine-tuned, and an architecture-matched random-init control when affordable | Benefits of prior training versus architecture and resolution changes |
| V16 | Robustness and review | Declared blur/brightness/compression/occlusion levels and low-score review policy | How quality degrades and what abstention can and cannot establish |

### Keep optimizer and architecture conclusions honest

- In an optimizer study, hold model, objective/reduction, initialization, transform distribution, batch size, data split and comparable tuning allowance fixed. Tune each optimizer's learning rate; one shared arbitrary rate is not a fair verdict.
- Separate weight decay from optimizer effects: compare SGD/Adam under a declared no-decay condition first, then tune the regularized variants. Adam's coupled L2 and AdamW's decoupled decay are not the same experiment.
- In an architecture study, use the same input resolution/data protocol initially. Record differences in parameter count, compute and memory; give architecture alternatives comparable tuning budgets rather than forcing a configuration known to fail one model.
- Pair seeds and initialize shared modules consistently where possible. Identical seeds alone do not ensure different architectures consume the same random augmentation stream; record or precompute paired transform draws when that control is needed.
- Report validation quality versus updates, examples seen and elapsed time. Equal epoch counts or equal update counts are not necessarily equal compute.
- Gradient accumulation matches a larger batch only under matching loss reduction and compatible stochastic/state behavior. BatchNorm and dropout can break exact equivalence; do not claim accumulation is a drop-in large-batch proof.
- A feature/augmentation change is a new trial. Resume cannot silently change its data/transform identity.

### Proposed bounded trial policy

Start from one device, FP32, one-seed correctness/overfit checks and a small subset. Proposed repeated seeds are `17,29,43`; confirm them in the experiment manifest. Use a small log-scale rate search such as `1e-4,1e-3,1e-2,1e-1` as a pilot range, not a promise that every optimizer uses those rates well. Narrow the main studies from pilot evidence.

Initial session caps follow the [workspace Colab contract](../../../docs/guide/COLAB.md): 60 minutes or 1,000 additional optimizer updates, whichever comes first. A pilot can stop after 100 updates. These are session limits, not a fresh optimizer/schedule on each restart. Finalize total trial budgets, validation frequency and early-stop patience after measuring throughput; save the declaration before the main comparison. Record best and final checkpoints without selecting by test score.

Checkpoint after 5 minutes or 100 completed updates, whichever comes first, and at normal stop; use shorter intervals for recovery drills. Retain failures, aborted trials and budget exhaustion. Nonfinite loss/weights, invalid gradients, incompatible state or failed persistent saves require an explicit failure/stop rather than silently restarting a run.

Mixed precision, channels-last layouts, compilation and quantization are optional measured performance extensions after FP32 correctness. Compare quality and resource cost under supported hardware; do not require a particular GPU or equate smaller storage with faster inference.

## 9. Augmentation and transform contracts

The first transform is deterministic decode → orientation → RGB → declared resize → tensor → normalization. Keep validation/test transforms deterministic. Never randomly crop held-out images during the headline evaluation; any test-time augmentation is a separately named and costed policy.

Possible training augmentations are small rotations, horizontal flips, modest brightness/contrast changes and a crop policy checked against actual images. Decide ranges through a training-only visual audit: cropping can remove the labelled material, and strong color changes can remove a useful physical cue. Inspect transformed examples before enabling a sweep. An augmentation's plausibility is not established by an API accepting its settings.

For scratch models, estimate any channel mean/std from training originals under the declared deterministic preprocessing. For pretrained networks use the selected weights' documented preprocessing unless an explicit ablation changes it. Record antialiasing, resize/crop geometry, interpolation, channel order, input range and normalization. Caches are keyed by image and transform identity; augmented images cannot leak through a shared cache into validation.

## 10. Residual and transfer-learning comparisons

A residual block computes `y = x + F(x)` when shapes match. A block changing width or spatial size needs a declared projection/downsampling path. Inspect identity/projection outputs and gradients before stacking blocks. Compare a matched plain network so a residual connection is not confounded with several simultaneous architecture changes. A skip connection does not guarantee better generalization.

After the scratch route is understood, use one established CNN backbone, provisionally ResNet-18, as the transfer reference. Pin the exact architecture, pretrained weight identifier/checksum, source/license, training provenance where documented and preprocessing in the run manifest. No weights have been downloaded for this plan.

The three distinct conditions are:

1. **Frozen features:** replace the classification head with nine outputs and train the head. Frozen parameters alone do not freeze BatchNorm running buffers; explicitly set and test the backbone's mode/state policy while the head trains.
2. **Fine-tuning:** unfreeze selected blocks with declared parameter groups/rates, then assess whether more adaptation is justified. Record the BatchNorm-buffer policy and optimizer-state transition when parameters are added.
3. **Architecture-matched random initialization:** when affordable, train the same backbone at the same input resolution to distinguish pretrained features from changing architectures. If that control is outside budget, state that the comparison combines those changes.

A pretrained model may require a larger input than the small scratch baseline. Report that resolution and compute difference. Prior external training is additional information, so a transfer result is a practical comparison, not an equal-data scratch learning claim. Assess any known dataset overlap; unknown overlap remains a limitation rather than an asserted clean-room benchmark.

## 11. Evaluation, diagnostics and robustness

Primary selection metric: **macro-F1 over the fixed nine classes**, with class support shown. Report top-1 accuracy, per-class precision/recall/F1, a labelled confusion matrix, ordinary evaluation NLL and model size/latency. Define zero-denominator behavior and do not average away classes a model never predicts. Top-three accuracy is supplementary for a review assistant, not a replacement for top-1 quality.

Use the training-majority and prevalence predictors plus simple image-feature baselines. Evaluate NLL on the natural held-out class distribution even when training used class weighting or a rebalanced sampler. Weighted sampling and class weighting are separate studies; combining them changes the effective objective again.

For performance comparisons, fix device, precision, image size and batch size. Measure batch-one application latency separately from batched throughput, and separate decode/transform time from model-only forward time. Warm up the model, synchronize accelerator timing boundaries, and report repeated median/p95 latency with the number of measurements. Record cold-load time and model/buffer storage separately. A CPU result, an asynchronous dispatch time or a different input resolution cannot silently stand in for a GPU or end-to-end comparison.

Track training/validation loss, task metrics, learning rate, activation statistics, gradient norms, update norms, inactive-ReLU fraction where meaningful, throughput and peak memory. Train/eval mode must be explicit: `eval()` changes dropout/BatchNorm behavior but does not disable gradients. Inference uses an appropriate no-gradient context as well.

### Error review and model reliance

- Review false predictions across every class, including high-score mistakes and rare classes. Annotate observed blur/clutter/lighting without inventing labels absent from the dataset.
- Use simple occlusion or saliency probes as diagnostic evidence of sensitivity. A heatmap is not a causal explanation or a verified object mask.
- A background-only or crop-perturbation probe can expose shortcuts, but absent object masks it cannot isolate background use perfectly.
- Controlled corruption sets use predeclared severity levels on held-out images. They measure robustness to those transformations, not all deployment shifts.
- Keep manually collected external photos separate; obtain reliable labels before reporting task metrics on them. Unknown materials and mixed piles need explicit limitations or review, not forced confident claims.

### Probability quality and abstention

Softmax scores can be overconfident. Inspect reliability diagrams with bin counts, Brier score and NLL. If fitting temperature scaling, use separate calibration records and evaluate reliability on disjoint records; do not fit the temperature on final test labels. Compare uncalibrated and calibrated artifacts explicitly.

A validation-selected score threshold can flag uncertain images. Report **coverage** (fraction receiving a prediction) and error on accepted images together, including class-dependent rejection. High confidence is not proof that an input belongs to a known class, so this is not a validated open-set detector.

Repeated seeds quantify optimization variability. If using uncertainty intervals over held-out results, use defensible image-group sampling where available and distinguish that uncertainty from seed variability. Small slice counts require visible denominators; a tiny high-scoring class is not broad deployment proof.

## 12. Independent checks before larger training

- Verify the patch calculation, shared-filter gradients, output-size formula and analytic parameter counts against loops/finite differences and actual PyTorch layers when implemented. Include multiple channels, unequal height/width, nonunit stride and padding cases.
- Check a tiny full MLP forward/backward with hand-known values; use float64 numerical fixtures and compare non-kink derivatives with proposed `atol=1e-6, rtol=1e-5`, finalized against observed reference behavior.
- Verify stable CE, class mapping, target dtype/range and weighted/smoothed reductions. Check large logits, a smaller final batch and an invalid label.
- Overfit a tiny clean batch before adding augmentation/regularization. A reference training check is separate from explaining its mechanism.
- Verify parameters are registered and receive expected gradients; frozen backbone tensors/buffers and intentionally unused modules have explicit checks.
- Check pooling/skip shapes, BatchNorm train/eval behavior, dropout mode, gradient clearing and parameter-group membership.
- Confirm no split overlap, no augmented-source leakage, deterministic evaluation transforms and no fitting of statistics on held-out records.
- Check corrupt/oversized files, orientation, grayscale/RGBA handling, unknown schema/model/class mapping and nonfinite outputs. Resource limits are validated on decoded images rather than trusting a file extension.
- Verify saved preprocessing and model produce matching fresh-process logits/scores/labels. Serve the same policy used for the final metrics.
- Test interrupted/resumed training, incomplete writes, corrupted newest generation and incompatible data/architecture identity before a long hosted run.

## 13. Application milestones

| View | Useful behavior | Evidence |
| --- | --- | --- |
| Image prediction | Supported input → exact preprocessing/model → category and alternatives | Known fixture, invalid-input behavior and artifact identity |
| Model comparison | Same held-out image through selected frozen model artifacts | Model/data/transform differences clearly visible |
| Error browser | Filter by true/predicted class and inspect failures | Correct confusion-matrix links and class/sample counts |
| Training benchmark | Compare learning curves, budgets, quality, size and latency | Every plotted result maps to a run manifest; missing/failed results remain labelled |
| Review queue | Flag images under the frozen review policy | Coverage/error report; no unsupported unknown-class guarantee |

The first app milestone is inference plus an error browser; training controls do not need to be exposed to the application user. A later upload interface must apply the same decoding/size policy as evaluation. External publication is separate from building the local application.

## 14. Proposed source and artifact layout

Only this README is created now. Proposed implementation files are:

```text
waste-material-classifier/
├── README.md
├── pyproject.toml / uv.lock
├── prepare_data.py / train.py / compare.py / inference.py
├── configs/                         # CPU smoke, bounded Colab and named studies
├── src/waste_material_classifier/
│   ├── data.py / transforms.py
│   ├── models/                      # MLP, small CNN, residual and transfer variants
│   ├── losses.py / trainer.py
│   └── evaluation.py / checkpoint.py
├── tests/                           # numerical, data, modes, artifacts and recovery
├── notebooks/colab.ipynb             # thin output-cleared common-trainer notebook
├── app/
└── docs/                            # sources, manifests and implementation checkpoint
```

Keep direct script inputs visible in their main blocks. The project is an ordinary ai_projects folder with no nested Git repository. No launcher command or collapsible TUI group is implemented by this README; shared terminal tooling belongs in `cli/`.

Inference artifacts store model architecture/config, tensor state including buffers, class mapping, exact transforms, score/calibration/review policy and source/dependency/data identity. Reject incompatible class order or preprocessing. Weights alone are insufficient to reproduce an image prediction; load only trusted compatible artifacts.

Each training run records data/split/transform hashes, dirty-source identity, dependency/runtime versions, weights provenance, seeds/generator state, optimizer/schedule/loss/reduction, batch size, examples/updates, budgets, resource identity, validation decisions, failures and timing. Reports link to immutable raw runs rather than replacing their records with a best-score summary.

## 15. CPU, Colab Free and full-state recovery

Develop numerical/data checks locally on CPU, then measure an actual forward/backward and bounded training pilot on the selected device. Hardware discovery alone is not readiness. The proposed 64-pixel model and batch sizes are starting designs, not verified memory/time guarantees. A CPU smoke must remain available when a GPU is absent.

Use the same trainer locally and from a thin Colab notebook. Follow the [workspace Colab and persistence contract](../../../docs/guide/COLAB.md); this vision project adopts it explicitly. No paid tier or hosted run is started by this plan. If an intended trial cannot fit the measured Free/available budget, stop and record a smaller declared design rather than silently enlarging resources or bypassing limits.

Save after a completed optimizer update with no pending accumulated gradients. A complete checkpoint includes:

- Parameters, persistent buffers, architecture, class mapping and transforms.
- Optimizer moments/velocity/parameter groups, scheduler, global update and any AMP scaler.
- Epoch, shuffled sampling order/next-batch position, Python/NumPy/PyTorch and applicable device RNG states, and augmentation-generator state.
- Best-validation/early-stop state, dataset/split/weight-source identity, config, dependency/hardware identity and metric/event position.

Start exact local parity with one data worker and a controlled augmentation generator. Multiple workers, prefetching and resumed random transforms require additional consumed-versus-queued state accounting. Same-environment deterministic continuation is distinct from CPU/GPU or version-changing continuation.

Write immutable persistent generations, validate their manifests and re-read checksums/loadability, write completion last, and retain an older verified fallback. An incomplete or corrupted newest generation must be rejected with an explicit rollback report. Checksum validity is not proof that files survive runtime deletion.

Acceptance is uninterrupted training versus stop/reload/continue locally, followed by actual Colab **Disconnect and delete runtime**, fresh source/dependency/data setup, persistent-state restoration and further updates. Log restored/advanced counters and resource identity. A notebook file, same-runtime reload or weight-only export does not establish this gate. All such execution remains pending.

## 16. Completion and handoff

- [ ] Dataset/license/hash/class mapping, duplicate/group audit and frozen partitions are reproducible.
- [ ] Baselines, complete MLP and scratch CNN train/evaluate/save/load through the shared pipeline.
- [ ] Numerical shapes, gradients, loss/reduction, parameter counts and mode/state behavior have independent evidence.
- [ ] Core architecture, optimizer/rate/batch, regularization, augmentation and loss studies have controlled budgets and retained failures.
- [ ] Matched residual and clearly labelled transfer/fine-tuning comparisons explain their extra information/compute.
- [ ] Frozen final test metrics include macro-F1, class supports, errors, probability quality and measured resource costs.
- [ ] Review/robustness behavior states its limits; external images do not become an unlabelled accuracy claim.
- [ ] The local application runs from an image through the exact saved artifact and shows useful error analysis.
- [ ] Local full-state resume and fresh-runtime persistent Colab recovery are separately demonstrated.
- [ ] I can trace an image through every layer, calculate a convolution/gradient/count and explain a failed or changed-condition experiment.

No accuracy target or improvement is invented in advance. Agree the application's quality/runtime requirements after baselines and a pilot, before the final comparison. Implementation correctness, task quality, learner explanation, hosted recovery and publication remain separate outcomes.

After review, continue [Air Quality Forecaster](../air-quality-forecaster/README.md), the selected final foundations project for RNN/LSTM/GRU and temporal forecasting. The existing [Support Ticket Router](../../transformer-labs/support-ticket-router/README.md) follows that sequence stage for text embeddings/tokenizer work. This README does not implement either next project.

## References and first action

Primary references inspected on 2026-10-15:

- [RealWaste metadata, images and license](https://archive.ics.uci.edu/dataset/908/realwaste): source facts and acquisition entry point; local data audit pending.
- [PyTorch Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html): cross-correlation, shapes and parameter semantics.
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html): logits, class indices, weighting/smoothing and reductions.
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): frozen feature extraction versus fine-tuning.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385): residual architecture reference, not an asserted paper reproduction.
- [Tutorial 0 outline](../../../docs/tutorials/projects/p00/transformer-foundations/v1/project/course-outline.md): linear algebra, calculus, network/initialization/backpropagation, losses, optimizers, evaluation and recovery. Convolution-specific teaching belongs here at first use.

The PyTorch 2.14 documentation was visible during source inspection; this is not an installed-runtime claim or a dependency choice. Pin compatible Python/PyTorch/torchvision versions and pretrained identifiers at kickoff.

**First action when this stage begins:** finish the classical ML applications' reviews, agree the nine-label image contract and work the first patch/gradient example. Then acquire and audit the pinned dataset before building the shared pipeline and MLP.
