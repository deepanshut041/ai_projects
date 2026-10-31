# Projects roadmap

## Standalone project execution — 2026-10-31

Standalone project execution (2026-10-31): all eight runnable projects support native commands and the optional workspace CLI/TUI. Clean standalone source checks and all eight workspace launch checks pass. Inference Labs also passes a fresh independent-clone check without its parent umbrella or CLI. Collection guidance and future scaffolds require both routes; Bike uses its native unittest runner through the adapter. Documentation-only projects, learner checkpoints and model results remain unchanged.

See the [native and workspace command table](standalone-projects.md) for current commands and verification limits.

## TinyServe boilerplate — 2026-10-30

TinyServe preparation (2026-10-30): Inference Labs is now an independent collection at `ai_projects/projects/inference_labs/`, with a learner-owned `tinyserve` boilerplate and local bare origin. Python 3.12.13, frozen tooling, status/setup/configuration launchers and Projects → Inference Labs TUI integration are verified. All 10 setup checks and 183 CLI tests pass; isolated wheel execution, a fresh source rebuild and six real TUI jobs at 80×24/120×36 pass. The learner will implement the model, cache, scheduler, kernels and API; no upstream engine is copied and no model/GPU/performance or learner gate is complete. Existing learner checkpoints remain unchanged. Next for TinyServe: choose the first small model and runtime when coding begins.

See the [TinyServe setup record](../../../cli/docs/implementation/2026-10/tinyserve-boilerplate-and-inference-labs-tui/00-resume.md) and [project README](../../projects/inference_labs/tinyserve/README.md). This prepares a later implementation; the existing learning route is preserved.

## Attention Labs worked visual completion — 2026-10-30

Attention Labs worked visuals (2026-10-30): 27 original editable figures now fill the missing numerical and state transitions across all fourteen continuation tutorials. They explain retrieval weights, group wiring, positional changes, latent paths, softmax merges, sparse reach, recurrent writes, gate order, paged ownership and balanced ring rounds beside the matching math/code. Existing source visuals, Foundations, MHA and every reference program are preserved. There remain 22 lesson pages in eight categories. Learner/model/hosted checkpoints are unchanged.

Follow the [ordered reading route](attention-labs-route.md). The [visual completion record](../implementation/2026-10/attention-labs-playlist-tutorial/13-original-visual-completion.md) owns the per-topic coverage and checks. Review one lesson and predict its changed case before continuing. The existing P00 learner checkpoint stays in force.

## Attention Labs continuation — 2026-10-30

Attention Labs continuation (2026-10-30): fourteen complete tutorials now extend the existing seven Foundations lessons and single MHA lesson, giving 22 lesson pages in eight categories. The ordered route covers KV caching, MQA, GQA, RoPE, ALiBi, MLA, FlashAttention, sparse/local/global/windows, linear attention, Performer, DeltaNet, Gated DeltaNet, PagedAttention and Ring Attention. Each new mechanism has worked math, source visuals, a complete locally checked reference, Colab instructions and transfer practice. Public Vizuara lessons and primary sources supply the material; source diagram errors are explained alongside corrected calculations. Original Lecture 9 remains excluded. Learner/model/hosted checkpoints are unchanged.

Follow the [complete reading order](attention-labs-route.md); the [continuation record](../implementation/2026-10/attention-labs-playlist-tutorial/12-next-attention-topics.md) separates component checks from learner and hosted results. The earlier eight-page preparation entries below are historical.

## Attention Labs tutorial correction — 2026-10-30

Attention Labs sequential revision (2026-10-30): Lectures 1–8 are revised in order across eight shared tutorial pages (seven Foundations lessons and one MHA lesson in Attention Heads), following the teacher's problem, diagrams, worked calculations and code progression. The lessons include 131 presentation/lab captures with explained bars, matrices and axes, restored six-word/five-word examples and runnable Colab-compatible references. Repeated provenance captions and four extra provenance JSON files are removed; attribution stays in references. Lecture 9 is excluded, MHA stays in Attention Heads, and model/learner/hosted checkpoints are unchanged.

The tutorial correction changes preparation quality and delivery, not the current P00 learner checkpoint or the Attention Foundations implementation scaffold. MQA/GQA/MLA were subsequently added as separate Attention Heads lessons; see the current continuation above.

## Attention approved text and new section videos — 2026-10-24

The learner explicitly approved the rewritten text before video production on 24 October 2026. Eleven new captioned videos contain 86 scenes and 47:02 of narration and worked visuals. They follow the same fixed example from the need for context through query/key comparisons, all positions, projections, heads, padding, causality, cross-attention, a gradient update, the complete actual implementation and its place in the Transformer. Every called project helper is shown before use. Each verified player appears before its matching notes section, with the same media in eleven corresponding tutorial chapters. Finished-file audio/speech/captions and root/prefixed desktop/mobile play, pause, seeking, transcript/download and print checks pass. The approved teaching body is preserved; only delivery-status footers reflect approval. Old rejected media remains archived. Attention source, Colab behavior and Bike results are unchanged; learner mastery, later model/training and hosted execution remain separate.

Start at the Attention project understanding notes, section 1, then continue in order through section 11. The video-delivery record is implementation/2026-10/attention-is-all-you-need-source-backed-tutorial/16-approved-text-video-delivery.md.

## Attention code understanding and section videos — 2026-10-23

The project understanding notes now contain seven focused code/value walkthroughs (about 2,400 words), replacing the 8,240-word course concatenation. The “What the attention computation means” learning points now explain roles, axes, heads, masks, paths and gradients in connected prose, both locally and in the shared tutorial. Seven captioned code companions with 42 editable scenes total about 24 minutes. Each verified video appears immediately after its matching notes heading, before prose; seven corresponding course chapters reuse the same media. Actual root and prefixed browser checks exercise all seven players at desktop/mobile widths, seeking through beginning/middle/end, caption cues, transcripts/downloads and print behavior. Source, Colab runner and Bike results are preserved. No learner explanation, translation training or hosted recovery is inferred. Next: watch/read the Q/K/V and score/value sections, then explain one full query trace.

## Attention runner and source layout — current, 2026-10-23

Six runnable scripts now live directly under src; core/data/training/utils remain grouped below it. src/cli is removed. The concept notebook is replaced by notebooks/attention_colab.ipynb, which uploads verified source, installs the locked Python environment, runs the demo and all 35 checks, and downloads logs. The Attention Lab tutorial retains explanations, worked maths and original figures. Current docs, source downloads and launcher paths follow this structure. Local operational-cell verification is distinct from an actual Google-hosted run. Later translation implementation, learner review and hosted recovery remain pending. This supersedes the earlier grouped-source notebook/cli layout below.

## Attention — grouped source and separate notebooks, 2026-10-23

The user explicitly requested grouped responsibility folders under src, superseding the initial flat-file suggestion. Attention now uses core (attention/model/config), data (tokenizer/dataset), training (trainer/checkpoint) and utils (graphs/controls), with runnable scripts directly in src. The Colab execution notebook lives outside src in notebooks/. The outer attention_is_all_you_need package directory and old root scripts were removed; setup, imports, launcher resources, packaging, current documentation and [the tutorial](../tutorials/projects/attention-is-all-you-need/attention-lab/v1/lessons/02-project-setup.md) follow the new layout.

The core numerical module is byte-identical after relocation, and fixed-seed demo outputs/weights match the pre-move baseline. All 35 existing tests pass; the six operational Colab runner cells execute locally with simulated file upload/download, real locked setup, demo and all tests. Runtime data ignores explicitly preserve src/data source files. Source downloads and the documented Colab ZIP are refreshed. Later tokenizer/data/model/training work remains unimplemented; learner review and hosted recovery remain separate. Next study: review the grouped source and the existing attention trace.

## Attention Is All You Need — setup and attention tutorial, 2026-10-23

The requested [twelve-lesson reading route](../tutorials/projects/attention-is-all-you-need/attention-lab/v1/lessons/01-weighted-information.md) documents existing setup and implemented attention with worked tensors, Q/K/V, softmax, heads, padding/causal/cross-attention, public API and gradients. Separate learning points/comparisons, reference runs, source downloads and readiness records support review. The local grouped project reader now exposes phase overviews and separate setup/attention implementation records, including a continuous view of the same authored explanations.

The user explicitly selected setup plus implemented attention now; data/model/training lessons are added as those phases are implemented. No model code, original notebook, captions, Bike source/results or learner gates are advanced. The current 35 CPU checks and demo pass; later tokenizer/data/model/training/BLEU and actual hosted recovery remain pending. Next for this request: read the weighted retrieval and setup, then review one complete attention trace. Later implementation starts with the Part 2 data/tokenizer/batch contract.

## Bike project tutorial, 2026-10-20

[Foundation Lab: Bike Demand Planner](../tutorials/projects/bike-demand-planner/foundation-lab/v1/lessons/01-prediction-contract.md) is the consolidated course for the completed Bike project. Its ten lessons retain the complete understanding notes, worked calculations, code/results, diagrams and 51 short captioned videos, including four new opening-lesson companions embedded before their matching prose. Learning points and comparisons are separate review pages beside that route; setup, source downloads, reference results and completion/readiness records share the tutorial reader.

The original private implementation records remain historical evidence with unchanged video source hashes and old URLs. The source stays at `0b49c0a`, the existing final application stays at port 8504, and no model is refitted or reselected. Tutorial authoring, native reader checks and browser verification are separate from learner mastery or hosted recovery. Next: read/watch the consolidated route from its first hourly prediction.

## Bike Demand Planner — local implementation complete, 2026-10-19

The complete local project is signed as `0b49c0a`: `foundations(bike-demand-planner): finish evaluation and local application`. Exactly 15 Bike files were committed after 128 source checks, full isolated reconstruction and installed/browser verification. The Bike source tree is clean; 105 unrelated staged files remain unchanged. No push or external publication occurred.

The [completion record](../../projects/foundations/bike-demand-planner/docs/implementation/10_frozen-held-out-evaluation/00-resume.md) retains the [worked notes and five embedded videos](../../projects/foundations/bike-demand-planner/docs/implementation/10_frozen-held-out-evaluation/02-notes-to-understanding.md). All ten planned local implementation increments are complete: prediction contract, data/schema/splits, baselines, linear regression, controlled optimization/objective/feature studies, practical models, error review, saved inference/recovery/application and frozen final assessment.

The isolated source checkout executes all 17 finite README project commands after setup. Every prepared data byte, all seven representative model files and all final metric sets reproduce. Recovery and fresh-process validation prediction parity rerun there as part of preparation. Both original saved assets and newly rebuilt assets load through the final installed wheel with source-project opens denied. Actual browser checks cover final assessment, historical validation replay, separate outcome reveal, missing observations, rejected/stale requests, seven models, counted months and widths 1440/768/390/320. The final page has been freshly restarted at [127.0.0.1:8504](http://127.0.0.1:8504/).

Completion fixes remove timestamp-specific input-run names, verify retained assessment bytes on repeat, reject incomplete/latest failed evidence, clarify separate preparation/server commands, remove unused scaffold metadata and correct stale validation-page wording. Evaluation/fitting/inference/recovery modules, original model/result/data artifacts and all video-pinned files remain unchanged. The isolated later run is explicitly historical reproduction, not another independent generalization claim. Groups/raw final MAE remains 125.4167 versus the retained forest's 103.6474; no selection or quality threshold changes.

Next: review the completed application's results and limitations. Learner mastery and hosted recovery remain unassessed, and E-commerce implementation has not started. Optional count/quantile/weather/hosted extensions are outside this completed local scope; earlier P00 learner evidence is unchanged.


## Bike Demand Planner Phase 09 — frozen final assessment and complete project explanation, 2026-10-19

Saved replay/recovery source is signed as `b140d58`, subject `foundations(bike-demand-planner): add saved predictions and local recovery replay`. Exactly 21 Bike files were committed after 109 checks, preserving unrelated staged entries. The user authorized the planned final evaluation; linear regression remains an implemented comparison, while groups/raw remains the prior selected candidate.

The [current record](../../projects/foundations/bike-demand-planner/docs/implementation/10_frozen-held-out-evaluation/00-resume.md) begins with 2 July 08:00, issued 07:00: forecast 349.5040 before target access, later observed 568, residual −218.4960. The seven-model protocol froze at 03:10:37 UTC before the run at 03:10:44; every target-free forecast was durably saved before the guarded raw target join. The development loader still rejects test access. No model is refitted, no seed/policy is newly searched and no test-driven reselection occurs.

Reviewed run: `final-20261020T031044371200Z`. Across 4375 later hours, groups/raw scores MAE 125.4167, RMSE 174.4702 and signed error −111.3563. Working-day rush scores 265.4320 over 741 hours. Forest final MAE is lower at 103.6474 despite worse validation MAE; preserve this negative transfer finding and the original selection. Forty missing slots remain unscored. All 122 source checks, independent audit of 30,625 aligned forecasts/seven metric sets/slice masses and all training-derived group outputs, source/data/artifact hashes, package privacy and source-denied installed-wheel execution pass.

The [final application](http://127.0.0.1:8504/) shows the fixed comparison, monthly counts, predeclared examples, costs, provenance and unchanged artifact identity. Historical prediction and observed-outcome reveal remain separate. Its January–June request authority does not silently expand because later batch features exist. Actual desktop/mobile checks at 1440/768/390/320 pass, including missing observations and rejected/stale requests.

Implementation/results and all six complete written records precede five captioned companion videos (4:20–4:45), embedded before matching prose. Finished-file and actual root/prefixed/managed-reader playback, pause, seeking, captions, desktop/mobile and private-export checks pass. Next: explain what can be claimed if the forest is chosen now. Technical local implementation is checked; substantial underprediction and absent operational acceptance criteria remain visible. New source is uncommitted. Learner review, hosted recovery, external publication and E-commerce implementation remain pending; earlier P00 learner evidence is unchanged.


## Bike Demand Planner Phase 08 — saved forecasts, full-state recovery and local replay, 2026-10-19

Error-review source is signed as `2318f70`, subject `foundations(bike-demand-planner): review hourly errors and freeze prediction candidate`. Exactly thirteen Bike files were committed after 96 prior checks passed; unrelated staged entries were preserved. The user authorized the saved application/recovery continuation, with linear regression retained for the separate training-state demonstration. No new assessed learner explanation was supplied.

The [current record](../../projects/foundations/bike-demand-planner/docs/implementation/09_saved-prediction-recovery-and-replay/00-resume.md) starts with the real Monday 08:00 saved forecast 349.5040. A fresh process reproduces all 4357 validation responses exactly. The immutable JSON bundle carries original candidate/model bytes, schema/order/domain/output/fallback and target-free recorded holiday authority. All 109 checks, independent forecast/state/Adam/source audits, distribution privacy and source-denied installed-wheel execution pass.

A separate training-only linear Adam demonstration uses batch 32, seed 17, rate .01, inverse-time decay and three passes. Generations preserve transform, moments/index, order/cursor/RNG, schedule, monitoring and context. Deliberate corruption of generation 3 and a partial pending publication trigger an explicit fallback to generation 2 at update 273/examples 8709. Its next hr_08 update is 1.1941539882→1.1978901611; final state at 813 updates/25935 examples exactly matches uninterrupted numerical state. A zeroed-moment control changes the result. Wall-clock parity, arbitrary hardware/power-loss and hosted recovery are not inferred.

The [local application](http://127.0.0.1:8503/) exposes historical validation replay and seven compatible benchmark artifacts. Forecast/baseline/identity precede a separate recorded-outcome action; the real missing Jan 2 03:00 hour remains unlabeled, unsupported dates clear old output, and June sorting changes table order without changing the candidate. Actual UI checks pass at 1440/768/390/320 widths with no document overflow or JavaScript errors. Groups/raw and its overall/June/rush tradeoffs remain unchanged.

Implementation/results and all six written records preceded five captioned videos (4:24–4:49), embedded before matching prose. Finished-file, actual root/prefixed/managed-reader play/pause/seeking/captions, desktop/mobile and private-export checks pass. Next: review why exact recovery can coexist with poor short-run fit and unchanged June weaknesses. New source remains uncommitted. Learner review, final-test assessment and any hosted recovery remain pending.


## Bike Demand Planner Phase 07 — hourly errors and candidate boundaries, 2026-10-19

Practical-estimator source is signed as `6226dc1`, subject `foundations(bike-demand-planner): compare calendar trees and ensembles`. Exactly twelve Bike files were committed after 83 prior checks passed; unrelated staged entries were preserved. The saved error-review scope follows those models, with completed linear regression retained as a reference. User authorization supplies no new assessed learner explanation.

The [current record](../../projects/foundations/bike-demand-planner/docs/implementation/08_residuals-slices-and-candidate-choice/00-resume.md) starts with the real Monday 08:00 forecast 349.5040 against target 420, residual −70.4960. All 96 source checks, independent audit of 14 prediction files/686 slice summaries/time groups/level comparisons and isolated distribution checks pass. Four scientific figures retain counts, missing hours and undefined empty categories. No model is refitted and final-test targets remain reserved.

Groups/raw remains selected at overall MAE 91.5729; working-day rush has 750 rows and MAE 178.1954. June MAE is 142.9952, while the retained forest achieves 90.8782 there but is worse overall at 99.8324. The existing overall-MAE objective is retained with this tradeoff explicit. Ten missing slots are not imputed; 4346 consecutive-hour pairs have residual correlation 0.8677. Calendar-matched historical/later levels are diagnostic, not causal or a new candidate. Four valid sensitivity cases and fourteen rejections define a supported historical replay contract. Candidate JSON freezes exact original model bytes with feature/schema/data/protocol identities, output/fallback policies and known limits.

Implementation/results and all six written records preceded five captioned videos (4:18–4:41), embedded before matching prose. Finished-file, actual root/prefixed/managed-reader playback, pause, seeking, captions, desktop/mobile and private-export checks pass. Next: review why a valid June request receives the same group mean as January while another model improves June without winning the current overall objective. New source remains uncommitted. Application/recovery, final-test evaluation and learner review remain pending.


## Bike Demand Planner Phase 06 — tree leaves and ensembles, 2026-10-19

Objectives and calendar-structure source is signed as `c0d7f8c`, subject `foundations(bike-demand-planner): compare losses and calendar feature models`. Exactly eight Bike files were committed after all 72 prior checks passed; unrelated staged entries were preserved. The user authorized the saved practical-model scope, with linear regression retained as a completed reference. No new learner explanation was supplied.

The [current record](../../projects/foundations/bike-demand-planner/docs/implementation/07_tree-forest-and-boosting-comparisons/00-resume.md) follows four real hourly counts into tree means 1.5/187 and boosted predictions 103.525/111.8725, then traces later January forecasts through fitted library-backed models. All 83 checks, independent prediction/metric/leaf/source/hash audits, packaging privacy and isolated installed-package checks pass. Frozen non-test data and all repository indexes are preserved.

The reviewed calendar-only study retains three training-only timing pilots, three configurations and three seeds per family, 27 main tree-family fits, four deterministic references and 26 seed-aggregated configuration/policy summaries. Best tree/forest mean validation MAE is 99.9993/99.7549; clipped 300-stage boosting reaches 102.2215. Groups/raw remains selected at 91.5729. The busy tree leaf averages eight training targets to 180.75 against later target 420, exposing a concrete historical-fit/transfer gap. Seeds describe algorithm variation, not independent future periods.

Code/results and all six written records preceded five captioned videos (3:45–4:25), embedded before matching prose. Finished media and actual root/prefixed/managed-reader playback, pause, seeking, captions, desktop/mobile and private-export checks pass. Next: review why the finer historical leaf forecasts the later commute worse than the broad group. New source remains uncommitted. No reserved-test result, deeper error-review implementation, learner mastery or application/hosted outcome is inferred.


## Bike Demand Planner Phase 05 — objectives and calendar structure, 2026-10-19

Optimizer source is signed and committed as `b7f73b1`, subject `foundations(bike-demand-planner): compare gradient optimizers and batch policies`. Nine Bike files were isolated from unrelated staged work; all 60 prior checks passed. The user authorized this continuation without providing a new learner explanation.

The [current record](../../projects/foundations/bike-demand-planner/docs/implementation/06_losses-regularization-and-calendar-structure/00-resume.md) implements explicit half-MSE, MAE and Huber slopes, intercept-free L2, training-only feature transformations, separate conditioning/structure studies and raw/nonnegative output policies. Twelve within-loss training pilots precede twelve comparison fits and two baselines. All 72 checks, independent prediction/metric/objective/stationarity/hash audits and source/wheel/isolated-package checks pass. Frozen data, source/video pins and all repository indexes are preserved.

Hour/working-day interactions improve raw/clipped validation MAE to 108.6937/100.5164, while the established group baseline remains stronger at 91.5729. The validation-led selection freezes groups/raw and its model/protocol identity. Target transforms remain deferred. No final-test result, application serving or learner mastery is inferred.

Implementation and all six written records preceded six captioned videos (3:59–4:17), now embedded before matching prose. Finished-file and actual root/prefixed/managed-reader playback, seeking, captions, desktop/mobile and public-export checks pass. Next: review the distinction between objective, penalty, conditioning and capacity, and why groups/raw remains selected. New objective/feature source remains uncommitted; Phase 06 and later implementation remains pending.


## Bike Demand Planner Phase 04 — batches, momentum and Adam, 2026-10-19

Linear regression is signed and committed as `7064b7b` with subject `foundations(bike-demand-planner): fit and validate calendar linear regression`; exactly 11 Bike files were isolated from unrelated staged work. The user authorized continuation without supplying a new learner explanation.

The [current working record](../../projects/foundations/bike-demand-planner/docs/implementation/05_batches-momentum-and-adam/00-resume.md) now implements float64 SGD, a declared momentum convention and bias-corrected Adam, seeded without-replacement batches with actual tail averaging, example-based inverse-time decay and versioned continuation state. All 60 source checks, independent scalar probe/metric/identity/hash/order checks, package privacy and isolated installed-wheel optimizer/continuation checks pass.

A training-only twelve-run pilot selects rates 0.01/0.001/1.0 for SGD/momentum/Adam, retaining two excessive-rate failures. Before main trials, the protocol freezes paired seeds 17/29/43, ten passes (86450 examples), a revised 100000-update cap and 300-second limit. Separate optimizer, batch and schedule families reuse exact configurations, giving 18 unique main trials. Batch-32 mean validation MAEs are 116.1661, 116.6027 and 114.3608; Adam has more seed variation and none beats the group baseline's 91.5729. Decay improves training squared error while mean validation MAE slightly worsens.

Implementation and all six written records preceded six captioned companions (4:11–4:58), embedded before matching prose. Finished media and actual root/prefixed/managed-reader playback, seeking, captions, desktop/mobile layouts and private-export exclusion pass. Next: review the optimization/work/validation distinctions. The new source is uncommitted; frozen data, repository indexes, prior source/video pins, reserved-test and learner/hosted boundaries are preserved. Phase 05 remains pending.

## Bike Demand Planner Phase 03 — full-batch linear model, 2026-10-19

Baseline code is signed and committed as `e35ed93` with subject `foundations(bike-demand-planner): add calendar baselines and aligned evaluation`; only ten Bike files were committed and all unrelated staged entries were preserved. The user authorized the next implementation without supplying a new learner explanation.

The [current record](../../projects/foundations/bike-demand-planner/docs/implementation/04_linear-regression-and-full-batch-learning/00-resume.md) now implements float64 forward/loss/gradients, training-only population scaling, two checked updates and a bounded full-batch fit. Record 54 moves 143.7944 → 147.0363 → 149.9337, then 174.7785 at convergence. All 48 source checks, independent Decimal/scalar metric/identity/hash checks, packaging privacy and an isolated installed-wheel fit pass. The fit converges in 200 updates, with gradient norm 9.7244e-8; the rank-42 stable reference agrees within 1e-6 rentals/hour on observed inputs.

On the same 4357 validation rows, linear MAE is 116.7312 and RMSE 157.9791, worse than group MAE 91.5729 and RMSE 137.3900. The notes explain additive versus grouped curves, unchanged-input predictions and 724 retained negative forecasts. Implementation and all six written records were finished before producing five captioned section companions (3:50–4:44). Each is embedded before its matching prose. Finished audio/video, actual root/prefixed playback and middle/later seeking, captions, desktop/mobile layouts and private-export exclusion pass; all 26 provider requests completed with none pending. [Media evidence](../../projects/foundations/bike-demand-planner/docs/implementation/04_linear-regression-and-full-batch-learning/linear-media-verification.json) retains measured checks. Next: review the shared-gradient/validation limitations. The new source is uncommitted, frozen data and indexes are preserved, no test targets were opened and later phases remain pending. Earlier P00 and learner/hosted evidence are unchanged.

## Inline section videos with deeper worked examples — 2026-10-19

The [hourly-data understanding notes](../../projects/foundations/bike-demand-planner/docs/implementation/02_hourly-data-audit-and-splits/02-notes-to-understanding.md) are the watch/read entry: six captioned players sit immediately above their matching prose. They are now expanded to 4:42–6:34, with 18 additional scenes and about 11 minutes covering another request, missing-hour arithmetic, exact vector/dot-product changes, circular distance and coefficient dependence, delay-sensitive readiness, and observation/schema replay. The notes were expanded first. Existing film/cuts and source snapshots remain preserved.

The [project-learning-video skill](../../../ai_research/research_docs/.agents/skills/project-learning-video/SKILL.md) reuses project-learning and the synchronized owner video workflow. CLI and reader evidence live in the [section-video package](../../../cli/docs/implementation/2026-10/section-videos-embedded-in-understanding-notes/00-resume.md). Local media remains ignored. Bike Phase 02 still resumes with the hour/working-day baseline; no model training, learner mastery or hosted gate is advanced.

## Bike data walkthrough video companion — 2026-10-19

The requested complete explanation is delivered as a [23:03, 1080p narrated video](../../../.ai/videos/projects/bike-hourly-data-calendar-split/out/video.mp4), with 37 scenes, six chapters, animated vectors/timelines, code highlights, runnable commands and SRT/VTT captions. The [private companion record](../../projects/foundations/bike-demand-planner/docs/implementation/02_hourly-data-audit-and-splits/video-production-2026-10-19.md) links the transcript and measured verification. Visual/timing checks and the finished audio/video checks pass. Original source and staging are preserved; media stays local and ignored, with no remote publication.

This is a companion to the completed data handoff. Current modelling remains Phase 02; the separate training-only hour/working-day baseline and fallback comparison are still next. No new learner explanation, model result or hosted gate is inferred from video production.

## Bike Demand Planner Phase 02 — the first predictor and its errors, 2026-10-18

Phase 01 data preparation is signed and committed as `ad1b870` (`foundations(bike-demand-planner): prepare calendar data and chronological splits`). All 22 checks and setup passed before committing its 11 Bike files; all 104 unrelated staged entries were preserved. The frozen 45-column calendar schema and chronological membership are unchanged.

The requested next phase has started with a bounded metric/constant-predictor chunk. Fitting on 8,645 training targets gives 1,243,103 / 8,645 = 143.79444765760556 rentals/hour. Real validation records at 2012-01-09 02:00 and 08:00 have targets 5 and 420, making the constant model's opposing errors concrete. MAE/RMSE, signed error and explicit overall/conditional underprediction definitions now share an identity-checked scoring path. The first validation result on 4,357 hours is MAE 154.2452, RMSE 208.1723 and signed error −71.4057.

All 33 unit checks pass, and independent arithmetic verifies the mean, metrics, prediction identities and artifact hashes. A versioned run retains source/dependency/data/split identities, fit/prediction policies, metric definitions, timings and failure status. Source/wheel privacy, isolated wheel imports/known-answer execution and setup pass. This new source remains uncommitted, separate from the completed data commit. No reserved test data is loaded for fitting/scoring.

Current learning is in [simple predictions and hourly error metrics](../../projects/foundations/bike-demand-planner/docs/implementation/03_simple-predictions-and-hourly-error-metrics/00-resume.md), under ignored project docs. Natural-language learning points, the complete code/maths walkthrough, comparisons and the fit/predict/score diagram retain the explanation. Next is a separate training-only hour/working-day mean with an explicit missing-group fallback, measured on the same validation hours. Phase 02 remains in progress; later phases have not started. The user requested the handoff from the completed data implementation, but no new learner explanation or mastery is inferred. Earlier P00 evidence and hosted gates are unchanged.

## Bike Demand Planner Phase 01 — calendar encoding and frozen membership, 2026-10-16

Readability update (2026-10-18): the current private learning record now uses connected explanations across the understanding notes, learning points and comparisons. Resume, next steps, tracker and entry pages use plain wording and a smaller progress table. The real Monday example links the pages; detailed source/check records remain available separately. Existing implementation, split identities, the recorded 22-check result and learner status are unchanged; no code tests were rerun for these writing changes.

The requested continuation is implemented after the official acquisition/audit chunk. Record 54, 2011-01-03 08:00, is a real Monday training observation: issue at 07:00, calendar hour 8/weekday 1/month 1/flags 0 and 1, and separate target 154. Fixed one-hot hour/weekday/month plus two flags give 45 ordered binary inputs. Raw year is omitted because it is constant in 2011 training; season is omitted from the initial schema while month is retained. No fitted scaler or data-derived category discovery is used; full-one-hot coefficient dependencies are documented.

Chronological membership is frozen as 2011 training, Jan–Jun 2012 validation and Jul–Dec 2012 reserved test, with the midnight target at each evaluation boundary excluded to respect the one-hour issue lead. Counts are 8,645 train / 4,357 validation / 4,375 test / 2 embargo rows. These are distinct from the source's 165 unrecorded slots. Availability guards check last preceding-stage interval end against first evaluation issue time. The offline protocol assumes interval-start labels and zero reporting/model-build delay on naive source timestamps; actual timezone/DST semantics remain unspecified.

The schema, full membership, labelled train/validation CSVs, test features without targets and frozen manifest are retained under ignored data/processed with source/schema/membership/export identities. Training-only constant-feature counts do not fit a scaler; columns retain fixed positions. Twenty-two checks pass, including independent active indices, strict forbidden/domain failures, boundary cases, disjoint complete membership, constant-column behavior, held-out perturbations leaving training/schema/membership unchanged, and reproducible/tamper-aware exports. Actual preparation and example outputs are separate evidence; no model or score has been produced.

Next: review the representation and boundary reasoning, then begin baselines only when requested. The learner's earlier changed-target gradient direction remains observed; the new encoding/split explanation is not inferred from passing checks. Phase 01 remains at this collaborative stopping point; Phase 02 and later implementation remain pending. The earlier signed numerical commit is `4794ad4`, and current acquisition/encoding/split source remains uncommitted with staging preserved. The private current implementation record owns full worked notes and evidence. Earlier P00/hosted state is unchanged.

## Bike Demand Planner Phase 00 — first worked contract and update, 2026-10-15

Active practical work is now Phase 00, in progress. The session uses an illustrative Monday 2011-10-17: issue at 07:00, predict system-wide rentals in [08:00,09:00), then use a later hypothetical target only for learning. The working-day flag, toy hour/4 encoding, initial weights and alternative counts 45/35 are demonstration choices, not acquired data or a final feature schema. The complete readable explanation lives in the private current implementation record; phase folders remain overview-only.

Implemented standard-library linear prediction, half squared error, analytic gradients and one non-mutating SGD update, plus a direct numerical comparison. Targets 45 and 35 run independently from initial weights [10,5]/bias 20 and prediction 40. Target 45 raises the prediction to 40.30; target 35 lowers it to 39.70 with weights [9.95,4.90]/bias 19.95. Both losses decrease from 12.5 to approximately 11.045. Seven checks pass, including the opposite-target known answer and central finite differences on a different mixed-sign case. No runtime numerical dependency was added. Dataset/production request parsing, batch fitting, model quality, application and hosted work remain pending.

The learner requested fuller human-readable reasoning and more natural learning points with Mermaid diagrams. Current learning points and the worked notes now explain the time boundary, features/state, loss, chain rule and update in connected prose; the canonical project-learning skill and template carry that style forward. Browser rendering of all three added diagrams succeeds; the two current diagrams were visually inspected. Prepared text and passing tests do not establish learner mastery.

Changed-target review: the learner correctly explained that prediction 40 against target 35 gives positive residual/gradients and decreasing weights for this positive feature vector. The code and notes now trace that case and compare equal loss magnitudes with opposite gradient directions. This records a narrow understanding observation; independent arithmetic and the broader information contract remain unassessed. Next: review the changed-target code/math comparison and any feature-scale question before a Phase 01 handoff. Phase 00 remains in progress, Phases 01–09 remain planned, and earlier P00/hosted evidence is unchanged. No new commit or publication.

## Project learning workflow and Bike docs — 2026-10-15

Phase-arc follow-up (2026-10-15): all ten Bike overview READMEs are reviewed against the current public README and retained detailed requirements. They now name concrete inputs/outcomes, common result/schema contracts, controlled study bounds, full saved-state/application behavior and frozen final-test handoffs. Private source-coverage/preservation evidence is retained in Bike implementation records. Phases remain overview-only and all phase/learner implementation remains pending; next is the existing Phase 00 request trace.

The canonical [project-learning skill](../../../ai_research/research_docs/.agents/skills/project-learning/SKILL.md) now routes incremental project implementation through understanding notes, concise learning points, plain next steps, comparisons and a progress tracker. The shared helper supports the learning domain and current-phase records; delivery/Research formats remain available for their own work.

Bike Demand Planner's private docs now contain the full local entry, ten phase overviews and separate implementation evidence. Phases always belong in `docs/phases/`, outside `docs/implementation/`. Every phase folder contains only its overview README, describing scope, outcome, completion and the long project arc. The six detailed records for current work live separately under `docs/implementation/01_prediction-contract-and-one-update/`. The existing human plan has been replaced with notes to understanding and separated learning points/tasks/comparisons; the tracker retains actual source/check/commit evidence without ticket-style bookkeeping. No study files are force-added or published.

The learner has already watched tutorials and read important PDFs. This practical route consolidates that knowledge with concrete inputs, maths, selected code, actual outputs and alternatives. Use brief recaps where useful and revisit exposed gaps; do not restart a general prerequisite course or infer mastery from prior reading. Next: start Bike Phase 00's information-availability contract and one update after this documentation preparation. Phase implementation and understanding assessment remain pending. Public Bike source and signed initial commit are unchanged; no model/training/application or hosted gate advances.

## Bike Demand Planner setup — 2026-10-15

[Bike Demand Planner](../../projects/foundations/bike-demand-planner/README.md) is the first practical application. Its Python 3.12.13 package, lock, direct setup/demo commands and unfinished data/train/compare/inference entry points are prepared. Runtime dependencies remain empty until numerical work needs them. Personal study/work records stay local; the public README describes the application and actual availability.

Current scope is runnable setup only. Frozen lock/setup, seven direct/module entry-point checks, isolated source/wheel runs, private-file exclusion, 69 local links and the illustrative arithmetic pass. Project-scoped whitespace checks pass; the broader parent check has an unrelated existing trailing blank line in the router pyproject, left untouched. No dataset, fitted preprocessing, model, training, benchmark results, application or CLI/TUI integration was created. The next learning action is the prediction-time/available-information contract and one worked numerical update. Work continues in small reviewed chunks with maths, code walkthroughs, alternatives and interpretation as the main focus; routine implementation is handled by the assistant. Later work remains pending. Bike remains first; subsequent projects retain the order in the Foundations catalog, including the separately prepared Air Quality Forecaster. Earlier P00 setup, pending `linear()` and all learner/hosted gates are preserved.

Earlier dated preparation entries below describe their recorded state and are superseded by this setup update where they still say Bike has no environment or source files.

## Final recurrent foundations project preparation — 2026-10-15

[Air Quality Forecaster](../../projects/foundations/air-quality-forecaster/README.md) is the selected final foundations application for recurrent sequence learning. Its separate README covers real historical PM2.5 forecasting, persistence/lagged baselines, manual RNN/LSTM/GRU cells, BPTT/gates, state/masks/reset/TBPTT, direct/recursive/encoder-decoder horizons, controlled experiments, historical replay and complete local/Colab recovery. Chronological target-time partitions and causal missingness/input policies are explicit; future measured weather/targets are forbidden as advance-forecast inputs. Attention, text embeddings and a public health/AQI service are outside scope.

The [Foundations overview](../../projects/foundations/README.md) now lists four applications across three stages: the two classical ML applications -> Waste Material Classifier vision stage -> Air Quality Forecaster sequence stage -> existing Support Ticket Router embedding/Word2Vec work. The vision README handoff now links to the sequence project. This supersedes earlier sequence-topic-undecided statements. Implementation next action remains Bike phase 0; each later application follows the preceding completion review.

Delivery for the recurrent project is README preparation only: no data, environment, source/model implementation, training, application, hosted recovery or CLI/TUI integration. Separately observed Bike Demand Planner scaffold files are preserved, and the overview now defers to each project README for its live setup state. Earlier P00 setup/pending linear() and all learner evidence remain intact. [Preparation checkpoint](../implementation/2026-10/air-quality-forecaster-recurrent-sequence-project-readme/00-resume.md).

## Waste Material Classifier vision preparation — 2026-10-15

The user selected [Waste Material Classifier](../../projects/foundations/waste-material-classifier/README.md) as the vision deep learning stage. Its separate in-depth README and the updated [Foundations overview](../../projects/foundations/README.md) define simple image baselines -> fully connected MLP -> scratch CNN -> architecture/training/generalization studies -> matched residual comparison -> frozen/fine-tuned pretrained CNN -> robustness, review and application. The README includes RealWaste provenance/split discipline, convolution/gradient and parameter-count examples, bounded experiments, evaluation, artifacts and local/Colab recovery gates. RNN/LSTM/text/embedding/detection/segmentation work is outside this project.

The application order is Bike Demand Planner -> E-commerce Purchase Predictor -> Waste Material Classifier. The first two make up the classical ML stage; the third application is the user's second, vision DL stage. Later sequence work will be chosen separately, and the existing Support Ticket Router remains the later embedding/tokenizer application. This supersedes earlier statements that the vision project's topic was undecided; it does not advance implementation or learner gates.

Prepared only: the new folder currently contains its README. No dataset/pretrained weights, dependencies, training, model/application source, hosted recovery or CLI/TUI integration was created. Immediate implementation next action remains Bike phase 0; begin the vision project after both classical ML completion reviews. Earlier P00 setup and pending linear() evidence is preserved. [Preparation checkpoint](../implementation/2026-10/waste-material-classifier-vision-project-readme/00-resume.md).

## Classical ML foundations route — 2026-10-15

The user selected a practical classical ML stage before later deep learning and embedding work: [Bike Demand Planner](../../projects/foundations/bike-demand-planner/README.md) first, then [E-commerce Purchase Predictor](../../projects/foundations/ecommerce-purchase-predictor/README.md). The new [Foundations collection](../../projects/foundations/README.md) sits beside Transformer Labs and Language Models as ordinary ai_projects directories. Each application has a distinct target, data contract, phased build route, controlled optimizer/loss/rate/batch/regularization studies, independent correctness checks, common task metrics and an application completion review.

This is preparation only: three README files exist; data, model code, environments, training, applications and launcher/TUI integration are pending. Begin Bike phase 0's prediction-time contract and numerical example at implementation kickoff. Finish its application/review before starting purchase classification. Later deep learning/complementary foundations projects are undecided; the existing Support Ticket Router remains the later embedding/tokenizer application. Existing Tutorial 0/P00 files and saved learner evidence remain valid prerequisites/reference material; no prior learner gate is declared completed or reset by this new route. Earlier P00-first next-action statements below record the prior route and are superseded by this user-selected practical sequence.

## Tutorial 1 topic preparation — 2026-10-13

Tutorial 1 preparation (2026-10-13): [Course 1](../tutorials/projects/p00/transformer-course/v1/lessons/01-start.md) now follows a 32-topic mathematical implementation route through translation, RoBERTa MLM and extractive QA, with worked gradients, source contracts, 64 changed-condition practices and editable topic notes. The outline was saved first and bounded agents authored/reviewed the topics. Existing chapter IDs, Tutorial 0/Small GPT and learner source files are preserved. Current learning remains P00-M1 setup complete, with linear() next in the relocated foundations lab; learner mastery and hosted recovery remain pending. No commit or publication. [Outline](../tutorials/projects/p00/transformer-course/v1/project/course-outline.md), [study notes](../tutorials/projects/p00/transformer-course/v1/project/study-notes.md) and [verification package](../implementation/2026-10/transformer-foundations-translation-and-roberta-course/00-resume.md).


## 2026-10-13 — ordinary grouped Projects folders

All four existing project folders are now under `projects/transformer-labs/`, owned by ai_projects rather than nested repositories/submodules. `projects/language-models/` is prepared for later Small GPT, GPT-2 and other language-model builds. CLI discovery retains the project IDs, and future Projects creation defaults to language-models with an explicit transformer-labs option. Documentation and Research keep their existing ownership. Recovery bundles, working-file archives and a verified move manifest are retained at workspace `.ai/project-layout-migration/2026-10-13/`. [Migration checkpoint](../implementation/2026-10/group-transformer-labs-as-ordinary-projects-folders/00-resume.md).

Actual learning stays P00-M1 setup complete, with linear() next in the relocated foundations lab. Semantic Notes Search and Tiny Vision Transformer remain planned; Language Models has no implementation yet. No training, hosted gate, commit or publication is implied.

Updated: 2026-10-13
Canonical plan for implementing, training and understanding known AI models.

Start each work session with [resume.md](resume.md), then [progress.md](progress.md). This file defines direction and completion gates; progress.md owns current status/evidence; resume.md owns the immediate next action; [decisions.md](decisions.md) records durable choices. Update all affected records before ending a substantial work session.

## Purpose and ownership

Learn model mechanisms deeply by building and training complete small implementations. The longer-term motivation is larger-model inference on constrained machines, but a Projects exercise implements or studies established techniques. Novel optimization questions, focused reproductions and improvement experiments belong in [Research](../../../ai_research/roadmap.md).

Keep the full architecture path. Study models through **understand → implement → train → inspect/debug → explain → profile → verify**. A known optimization can be an educational extension; a claimed improvement needs a separately scoped Research protocol. Do not replace hands-on learning with reading or an already-finished generated solution.

Implementations are ordinary directories tracked by ai_projects, grouped under projects/foundations/ for foundational applications, projects/transformer-labs/ for mechanism/task labs and projects/language-models/ for complete language-model builds. Do not create nested model repositories or model submodules. The workspace CLI lives separately in ../../cli/. All shared Projects documentation lives here under docs/. Use C++ and Python for implementation and tutorial examples. Large datasets, weights, environments, checkpoints and generated runs remain ignored, with provenance/acquisition instructions committed.

## Learning path and prerequisites

IDs are stable even when the study order changes. Dependencies mean learning prerequisites, not source-code coupling. Closely related builds share plumbing but remain separately runnable with a coverage/difference checklist.

| ID | Complete small-model project | Prerequisites | Mechanisms to implement and learn |
| --- | --- | --- | --- |
| P00 | Tokenization and Transformer Foundations | None; topic-based mathematical/deep-learning prerequisite course included | Tokenization, attention, masks, post-norm translation encoder–decoder, RoBERTa MLM/extractive QA, training and recovery. |
| P01 | Small GPT Language Model | P00 correctness, training/recovery and explanation gates | Decoder-only causal MHA, learned positions, pre-norm/GELU, next-token loss, text generation and full training state. Historical GPT-1 is an optional reference extension. |
| P02 | GPT-2 | P01 | GPT-2-specific config/initialization/weight mapping, tokenizer compatibility and verified KV cache; strengthen the existing pipeline. |
| P03 | Llama 2 → Llama 3 | P02 | RMSNorm, RoPE, SwiGLU, GQA and position/cache semantics. |
| P04 | Qwen3 Dense | P03 | Q/K normalization, model-specific configuration and reference parity. |
| P05 | Switch-style routing → Qwen3 MoE | P04 | Top-1/top-k routing, expert dispatch/combine, gradients and balance. Label decoder adaptations explicitly. |
| P06 | Qwen3-Next | P05 | Gated DeltaNet, gated attention, MoE and hybrid state. |
| P07 | PaliGemma | P03 | Vision encoder/projector, patch/token assembly, masks and image-dependent training. |
| P08 | Qwen3.5 | P06, P07 for vision | Complete hybrid text model, then model-specific vision integration. |
| P09 | Qwen3.8 dense VLM | P08 | Version-specific hybrid/vision paths and MTP where specification supports it. |
| P10 | Gemma 4 E2B text model | P03 | Local/global attention, per-layer embeddings and cache-sharing ownership. |
| P11 | DeepSeek V3 → V3.2 | P05 | MLA, shared/routed experts, MTP and causal sparse indexing. |
| P12 | DeepSeek V4-Flash-style model | P11 | Compressed attention and constrained residual-stream mixing. |
| P13 | Qwen3.8-Flash-Next-style model | P06, P09, P11 | Sparse selection, n-gram lookup, gated residuals and integrated hybrid MoE. |
| P14 | Kimi Linear → Kimi K3-style model | P06, P11; P07 for vision | KDA/gated MLA, Attention Residuals, latent experts and later vision path. |
| P15 | DeepSeek V4.1-Flash-style model | P12 | Causal encoder/decoder, shared compressed caches, replay and conditional memory. |
| P16 | Mamba 2 → Mamba 3 | P02; recurrence primer | State-space recurrence, sequential/chunked equivalence and richer state variants. |
| P17 | RWKV-7 | P02; recurrence primer | Time/channel mixing, recurrent updates and persistent-state behavior. |
| P18 | LLaDA → iLLaDA; optional editing extension | P02 | Bidirectional masked prediction, corruption/loss and iterative conditional generation. |
| P19 | Gemma 4 unified multimodal | P07, P10 | Direct modality projection, mixed sequences and end-to-end training checks. |

Default order: **P00 → P01–P05 → P10/P11 → P06–P09 → P12–P15**. P16–P19 are distinct branches entered when their prerequisites are understood. Complete the separately named incremental builds within a project; record partial coverage honestly. Source readiness is checked at each kickoff, not assumed from the model name.

Begin with **Tutorial 0 — Deep learning foundations for Attention Is All You Need**:25 topic lessons follow the saved outline from linear algebra/tensors/calculus/probability through networks/loss/optimization, data/encoding/embeddings, positions/attention/heads/masks and every original core sublayer, then seq2seq/evaluation/recovery/paper assembly. **Course 1 — Transformer Foundations: Translation and RoBERTa** applies them to complete model builds. **Course 2 — Small GPT** follows actual P00 gates. Reader numbers express study order while IDs/URLs/child checkpoints remain stable. Actual learner work is still linear() next, now specified in Tutorial 0 Lesson1 with its old Lesson3 continuation retained.

Detailed model plans, source links, sizing and CPU verification notes are consolidated below in [this roadmap](#detailed-project-plans-and-retained-source-notes). Current status lives in progress.md; setup/report conventions live in [getting-started.md](getting-started.md#workspace-setup). Published reference sizes are metadata, not instructions to allocate/train large models.

## Training compute and recovery

Develop locally and use Colab for bounded training. **P00–P04 must target Colab Free first**; P05 onward gets an evidence-based compute choice at kickoff, with Pro later when requested. Keep complete miniature models and local CPU correctness; published dimensions are reference metadata. This is a requirement, not a verified hosted capability.

Follow [COLAB.md](COLAB.md) for child-owned notebooks, dependency/source delivery, profiles, full training state, verified Drive checkpoint generations, retention and fresh-runtime recovery. A thin notebook calls the same trainer as local runs. Proposed session and save budgets are measured before expansion. Checkpoints include optimizer/scheduler/precision/RNG/data-position state; weights alone do not resume a training run. Model completion for the early group includes evidence after **Disconnect and delete runtime** and resuming the same run.

## Required cycle for each project

1. Read primary paper/source and pin revisions. Draw the full model and annotate tensor/state shapes.
2. Declare local correctness/published-reference configurations, tokenizer/data/objective contracts, simplifications and compute bounds; P00–P04 also require a Colab Free profile under the [Colab contract](COLAB.md).
3. Implement meaningful components and assemble a complete model. The learner predicts small numerical examples before running them; the assistant supports scaffolding, checks and review.
4. Train a small model/task; inspect actual loss, activations and gradients; diagnose at least one failure or controlled change.
5. Verify forward/backward, causal or conditional behavior, generation/state reset, complete training checkpoint/resume and relevant independent numerical references. For P00–P04 also verify persistent saves, interrupted-write recovery and resume after deleting the Colab runtime.
6. Explain model-specific mechanisms from the implementation and observations. Passing tests and learner understanding are separate gates.
7. Profile weights, cache/state, intermediates, workspace and tensor lifetimes/access. Measure prefill/decode separately where relevant.
8. Implement one established execution technique when it reinforces the mechanism being studied. Verify semantics and errors before interpreting timings.
9. Create or extend the corresponding tutorial under tutorials/ with editable diagrams, C++/Python worked examples, practice and whiteboard preparation; record tutorial availability separately from understanding.
10. Save coverage, results/limits and the next action. Update progress.md, resume.md and any new decision.

A negative or inconclusive performance result can complete an educational investigation. A speedup is not required, but evidence and explanation are.

## Project 0 Tokenization and Transformer Foundations

### Separate P00 video companion — 2026-10-10

[attention-is-all-you-need](../../projects/transformer-labs/attention-is-all-you-need/README.md) is the separate P00 build following three Priyam Mazumdar videos: attention → English/French tokenizer/dataset/collator → translation Transformer and training. Video 1 attention is now implemented and checked locally, with the original final Attention interface, an explicit path and efficient SDPA. A basic-concepts notebook imports the Python implementation; the latest request authorizes this notebook, superseding the earlier no-notebook setup constraint. The data/model/training files retain their names for videos 2–3 and are not implemented yet. Tokenizers and translation contracts follow the videos rather than the foundations byte/BPE/copy-task route. Full-state persistent recovery and hosted/learner gates remain pending. The foundations checkpoint and P01 sequence are unchanged.

**Recommended first project:** build the text-to-tensor pipeline, then a complete miniature encoder–decoder Transformer based on [Attention Is All You Need](https://arxiv.org/html/1706.03762v7). Proposed independent repository: ../../projects/transformer-labs/transformer-foundations-lab/; no repository/model/training exists yet. [START_HERE.md](getting-started.md#first-project) owns kickoff; the [current planning package](../implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/02-implementation-plan.md) records source evidence, contracts and documentation verification.

This is one bounded build, not a new curriculum area. Use Python/PyTorch for complete training and selected C++/Python numerical traces. Implement the mechanisms directly using tensor operations/autograd; use high-level Transformer modules only as independent checks, not as the implementation. A new tensor engine, CUDA kernels, full WMT translation and full C++ training port are optional later work.

| Milestone | Build and learn | Evidence before advancing |
| --- | --- | --- |
| P00-M1 | Tensors, broadcasting, matrix products, stable softmax/cross-entropy, chain rule/autograd and optimizer updates; fit one tiny MLP on synthetic labels. Pin sources/runtime and record CPU/Free budgets. | Hand-computed shapes/loss, an analytic or finite-difference gradient check, finite gradients and a explained loss decrease. |
| P00-M2 | Compare character and UTF-8 byte baselines; implement byte-BPE training, deterministic merges, encode/decode and artifact save/load. | Unicode/empty/repeated-input round trips, merge-order example, deterministic tie handling, split-before-tokenizer-training and artifact/hash checks. |
| P00-M3 | Embeddings, sinusoidal positions, scaled dot-product/single-head then multi-head attention, padding/causal masks, LayerNorm, residuals and ReLU FFN. | Hand-worked attention, independent numerical/gradient checks, shape/mask invariants and future-target perturbation checks with dropout off. |
| P00-M4 | Assemble encoder self-attention + decoder masked self-attention + cross-attention with post-residual LayerNorm; teacher forcing and padding-aware target loss. | Full forward/loss/backward and autoregressive conditional decode; prove source conditioning, target causality, PAD exclusion, EOS/bounds and fresh-request reset. |
| P00-M5 | Train on generated copy sequences; use reversal as one controlled extension. Add common trainer, thin Free notebook and full persistent recovery. | Separate fitting and disjoint held-out generation metrics; train/validation split checks; interrupted-save fallback, deterministic local resume and fresh Free runtime recovery under COLAB.md. |
| P00-M6 | Inspect gradients/attention/errors, count parameters and map memory; explain the encoder–decoder-to-GPT transition and save a tokenizer/trainer handoff. | Evidence-backed explanation, one diagnosed failure/change, bounded local profile, source/departure checklist and a versioned handoff. Learner mastery is assessed separately. |

### Tokenizer and data contract

Begin with a character exercise and a lossless byte baseline, then implement a deliberately simple byte-BPE. [Sennrich et al. §3.2](https://arxiv.org/html/1508.07909v5) motivates merge learning; [minbpe](https://github.com/karpathy/minbpe) is a readable byte implementation reference. This educational tokenizer is not claimed compatible with GPT-1/GPT-2 token IDs or preprocessing.

Proposed IDs: bytes 0–255, PAD/BOS/EOS/SEP 256–259, merge IDs from 260. Train merges on training records only, never across document/special-token boundaries. Break equal-frequency ties lexicographically by token-ID pair; stop at the vocabulary cap or no repeated eligible pair. Encoding applies frozen merge ranks, not new test-input frequencies. Treat literal special-token spellings as ordinary text unless explicitly supplied structurally. Preserve exact UTF-8 text without implicit normalization; distinguish byte round trips from rendering arbitrary generated bytes, whose UTF-8 error policy is recorded.

Persist schema, byte vocabulary/merge ranks, special-ID map, boundary/normalization policy, training split/source hashes and tokenizer hash. Freeze that identity in each run/checkpoint; a changed tokenizer requires an explicit new run. Report tokens per byte/record and vocabulary size on the same held-out text, without claiming that compression proves model quality. A character baseline may reject unseen characters explicitly; the byte path must encode valid unseen Unicode.

The mandatory model task starts with synthetic digit/text copy pairs using the byte baseline; a second bounded run can use BPE. Split/deduplicate source sequences before fitting the tokenizer or creating overlapping batches. Declare seeds, generator version, length distribution and split hashes; shared alphabet is acceptable, duplicate examples across splits are not. Train with decoder input `[BOS, y0, ...]` and labels `[y0, ..., EOS]`; ignore PAD in loss and metric denominators. Evaluate free generation without teacher forcing. Reversal adds a distinct transformation with the same pipeline; it does not replace the required copy baseline.

### Small architecture and paper fidelity

| Profile | Proposed complete model | Purpose |
| --- | --- | --- |
| Local correctness | 1 encoder + 1 decoder layer, width 32, 4 heads, FFN 128, source/target limit 32, byte vocabulary 260, FP32/dropout 0 | Fast CPU shape/gradient/mask and overfit checks. |
| Colab Free learning | Up to 2 encoder + 2 decoder layers, width 64, 4 heads, FFN 256, source/target limit 64, vocabulary at most 512 | Bounded task training/recovery; choose only after a measured smoke run. |
| Architecture reference | Paper dimensions/training choices recorded as metadata | Explicit reduced-size/task/training departures; never allocate implicitly. |

Keep bidirectional encoder attention, causal target self-attention and decoder queries attending to encoder keys/values. Encoder and cross-attention hide source padding; decoder self-attention also hides target padding. Valid query rows must have a visible key; reject empty/all-padding source records and define safe padded-query behavior. Do not impose GPT causality on the encoder. Use paper post-norm, sinusoidal positions, ReLU and embedding scaling. Record weight tying, dropout, label smoothing and scheduler choices; initial unsmoothed loss/plain bounded optimizer choices may differ from the paper. The [Annotated Transformer source](https://github.com/harvardnlp/annotated-transformer) is a component/copy-task guide whose convenience pre-norm code differs from the paper; inspect before reusing.

Finalize task pass thresholds/budgets before training. A proposed gate is at least 90% exact generated copy matches on 100+ disjoint held-out examples in the trained length range, plus reported token accuracy and source-shuffle/control degradation. Small-set overfit is a separate gate. Diagnose failures before expanding; record new protocols without relabeling a failed run as success. Synthetic success demonstrates the chosen sequence task, not translation or general language competence.

### Handoff to the first language model

P00 supplies a frozen tokenizer artifact/API, proven mask/block contracts, data/split/provenance conventions and trainer/checkpoint patterns. P01 remains independently runnable: use an explicit versioned package or source snapshot with license/revision, not live imports through workspace-relative paths. Reuse verified plumbing selectively; do not transfer encoder–decoder weights as a GPT checkpoint. Copy the [COLAB.md](COLAB.md) notebook/full-state/persistent-recovery gates into the child package and verify them again for the language-model task.

## Project 1: Small GPT Language Model

[Start the integrated P01 tutorial](../tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md) after the P00 exit gates. Its [project readiness record](../tutorials/projects/p01/small-gpt/v1/project/readiness.md) maps the required milestones; local reference evidence is preparation.

Proposed repository: ../projects/gpt-lab/ (not created yet), entered after P00 gates. Build a complete small decoder-only text model using causal multi-head attention, learned positions, pre-norm/final LayerNorm and GELU. [build-nanoGPT](https://github.com/karpathy/build-nanogpt) is a code-reading reference; use bounded local/Free settings instead of its large/distributed defaults. This is an educational GPT-style model, with original GPT-1 details retained as optional historical study.

| Milestone | Outcome |
| --- | --- |
| P01-M1 | P00 handoff/revision checklist; decoder-only architecture map; tiny config, frozen byte/BPE tokenizer, licensed text split/shifted-target and Free setup/storage/budget contracts. |
| P01-M2 | Complete pre-norm decoder; independent attention/gradient and causality checks. Explain removal of encoder/cross-attention and changes from P00's positions, activation and normalization. |
| P01-M3 | Train/backpropagate a small text LM, measure held-out loss, sample text, implement common-trainer notebook/full persistent resume and verify interrupted saves/fresh Free recovery. Inspect/explain behavior. |
| P01-M4 | Optional historical GPT-1 post-norm/classification adaptation with explicit pooling and combined objective. Keep this separate from the required text LM. |
| P01-M5 | Required parameter/memory ledger, reproducible CPU profile, diagnosed failure/change and learner explanation. |
| P01-M6 | Optional early KV-cache extension; required GPT-2 cache coverage follows in P02. Match uncached/cached positions and outputs before comparing storage strategies. |

P01 completion requires M1–M3 and M5 plus the shared learning/recovery gates; M4/M6 are optional extensions. Initial correctness config remains 2 layers, width 64, 4 heads, context 64, FFN 256 and byte vocabulary 260; use the actual frozen BPE vocabulary in the BPE profile. Increase toward 1–3M parameters only after measured fit. The [older GPT-1 plan](../implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/02-implementation-plan.md) retains historical source/architecture material; its mandatory post-norm/classifier sequence is superseded by this route. The old GPT-1 tutorial has been removed; use the integrated P01 course above.

Begin at **P00-M1**; create only the selected child's repository when implementation starts. No foundation/model milestone is complete from updating this plan.

## Resuming and advancing

Before editing: read resume/progress, inspect the owner AGENTS.md and Git state, then resume the child's existing implementation package. Create a child-owned package before deep implementation. Use task argv arrays and registered paths.

Advance only with linked implementation, correctness, training/inspection, learner-explanation and resource evidence appropriate to the model. Do not mark a whole family done after one variant. State implementation, verification, learning, research and external publication separately.

For a new Research question, write a short handoff identifying the mechanism, observed limitation, baseline and question. Put the authoritative protocol and results in Research. The learning roadmap continues independently.

## Change rules

Keep IDs stable; append decisions rather than erasing them. Record a changed study order/prerequisite and its reason. Extend source cutoff dates explicitly. Historical review/setup packages remain evidence; their old checkpoints do not override resume.md.


## Detailed project plans and retained source notes

The model plans, sizing guidance, exercises and resource links below were consolidated from the original October 9, 2026 planning document. They are reference notes within this single roadmap. The current P00-first order and P00/P01 gates above govern execution; the retained GPT-1 description is an optional historical branch. Later inherited model/release claims and resource availability were not freshly verified by this opening-route revision; recheck primary sources and pin revisions at each project kickoff. Proposed million-parameter budgets are optional later configurations, not allocations required for the first tiny model.

Original scope: from GPT-1 through dense, sparse, recurrent, diffusion and multimodal models. Prepared and release/resource cutoff: October 9, 2026.

### Objective and working assumptions

Return to LLM research by writing complete miniature model implementations, understanding their mathematics, verifying their behavior, and optimizing their execution.

The learner wants ground-up implementation before a small language model; current understanding is unassessed. P00 supplies bounded prerequisites, then the main work is implementation and experimentation. Blogs such as Sebastian Raschka's are examples of the desired teaching format; the resources below also include Karpathy, Umar Jamil, LabML, Stanford, model authors, and other open implementations.

The available PC does not have a suitable GPU for substantial training. Therefore:

- Use tiny, configurable models for local forward passes, backward passes, generation, state management, and profiling.
- Use short learning checks on synthetic or very small datasets when practical.
- Train early complete miniature models on Colab Free through the same code, with persistent checkpoints and fresh-runtime recovery under [the compute contract](COLAB.md). Local checks still come first; choose Pro later from measured needs.
- Keep full pretraining and large-scale post-training as later experiments.
- Write readable PyTorch reference implementations before adding optimized execution paths.
- Make each project end in a complete model, not just an isolated architectural modification.

The published model names identify the architectures being studied. The small parameter budgets identify the educational replicas. These replicas are not official checkpoints and are not expected to reproduce the original models' capabilities.

This is a selected roadmap of distinct, publicly documented architectures available at the cutoff date. It is not an exhaustive release catalogue or a benchmark ranking.


### How to size the implementations

The following are proposed starting budgets. Count the exact parameters in each implementation; embedding tables, expert sizes, and projection choices can change the totals substantially.

| Level | Starting configuration | Use |
|---|---|---|
| Component test | Width 32–64; sequence length 8–32; batch size 1 | Verify an equation, mask, gradient, routing decision, or recurrence. |
| Complete tiny dense model | 4–6 layers; width 128; 4 attention heads where appropriate; vocabulary 1,024–2,048; context 64–256 | Default GPT, Llama, Qwen, and dense-model projects. Target roughly 1–3M parameters. |
| Complete tiny sparse/hybrid model | 4–8 layers; width 128–192; 4 routed experts; top-1 or top-2 routing; small latent ranks | MoE, hybrid attention, and advanced cache projects. Target roughly 3–8M total parameters. |
| Tiny multimodal model | Small images with few patches; a shallow vision component; short audio clips where applicable | Verify image/audio token construction, modality fusion, and gradients. |
| Larger configuration | Increase dimensions through the same configuration interface | Resource estimates and future experiments when compute is available. |

Use FP32 for the initial CPU implementation. Use small FP64 inputs for selected numerical gradient checks. Preserve valid dimensional relationships, head divisibility, rotary dimensions, and state shapes when reducing a model.

For MoE and lookup-memory models, report total parameters, selected experts, state/cache memory, and lookup-table parameters separately. Sparse activation does not remove the storage cost of inactive experts or tables.

Keep explicit configuration profiles per model:

1. **Local correctness:** runs tiny CPU numerical checks and short experiments.
2. **Colab Free (required for P00–P04):** bounds the small-model training data/token, memory and session budgets; uses the same trainer and tested persistent resume. Later projects decide their tier at kickoff.
3. **Architecture reference:** records published dimensions and deliberate differences; never allocates implicitly.
4. **Colab Pro (later, optional):** expands a measured budget when requested without replacing the checkpoint/recovery workflow.


### Roadmap at a glance

The order follows learning dependencies rather than the historical genealogy of every model. Projects with arrows contain incremental builds, each of which should remain runnable.

| Project | Complete model implementation | Suggested local budget | Main implementation objective |
|---|---|---|---|
| 0 | Tokenization and Transformer Foundations | Width 32–64; count at kickoff | Tokenizer, attention and complete tiny conditional encoder–decoder. |
| 1 | Small GPT Language Model | Tiny correctness first; 1–3M later | Causal pre-norm decoder, learned positions and complete text LM; GPT-1 history optional. |
| 2 | GPT-2 | 1–3M | Pre-normalization decoder and complete language-model pipeline. |
| 3 | Llama 2 → Llama 3 | 1–3M | RMSNorm, RoPE, SwiGLU, GQA, and KV caching. |
| 4 | Qwen3 Dense | 1–3M | Qwen-specific attention normalization and model configuration. |
| 5 | Switch Transformer → Qwen3 MoE | 2–5M | Routing, dispatch, expert combination, and load balancing. |
| 6 | Qwen3-Next | 3–6M | Gated DeltaNet, gated full attention, hybrid state, and MoE. |
| 7 | PaliGemma | 2–4M | Vision encoder, projector, multimodal sequence, and decoder. |
| 8 | Qwen3.5 | 3–6M | Complete hybrid text model, followed by its vision path. |
| 9 | Qwen3.8-27B-style model | 3–8M | Integrated dense hybrid vision-language model and MTP. |
| 10 | Gemma 4 E2B-style model | 1–3M | Local/global attention, cross-layer KV sharing, and per-layer embeddings. |
| 11 | DeepSeek V3 → V3.2 | 2–5M | MLA, shared/routed experts, and learned sparse indexing. |
| 12 | DeepSeek V4-Flash-style model | 3–6M | Compressed attention and constrained residual-stream mixing. |
| 13 | Qwen3.8-Flash-Next-style model | 3–6M | QSA, n-gram embeddings, gated residuals, hybrid MoE, and MTP. |
| 14 | Kimi Linear → Kimi K3-style model | 3–8M | KDA, gated MLA, Attention Residuals, and latent-space experts. |
| 15 | DeepSeek V4.1-Flash-style model | 4–8M | Causal encoder–decoder, shared compressed caches, and conditional memory. |
| 16 | Mamba 2 → Mamba 3 | 1–3M | State-space recurrence, richer state tracking, and MIMO. |
| 17 | RWKV-7 | 1–3M | Recurrent time/channel mixing and generalized delta-rule updates. |
| 18 | LLaDA → iLLaDA; optional LLaDA2.1 | 1–3M | Masked diffusion, iterative generation, and token editing. |
| 19 | Gemma 4 12B-style unified model | 2–4M | Direct image/audio projection into a shared language backbone. |

#### Suggested study order

- **Foundations:** project 0, then projects 1–5.
- **Modern dense and sparse implementations:** projects 10–11, then 6–9.
- **Advanced research capstones:** projects 12–15.
- **Distinct research branches:** projects 16–19. These can be explored independently once their prerequisites are understood.

Gemma 4 fits naturally after Llama. DeepSeek V3 fits after the first MoE implementation. Kimi K3 depends on understanding both hybrid recurrence and MLA. DeepSeek V4.1 benefits from completing the earlier DeepSeek projects.


### Detailed implementation projects

#### 1. GPT-1 historical reference (optional P01 extension)

The required P01 is the Small GPT Language Model defined above. This retained historical architecture/task-head exercise is optional; it does not override the P00-first sequence or require exact GPT-1 replication.

**Build:** A small decoder-only language model with causal multi-head attention, learned positional embeddings, GELU, residual connections, and GPT-1's post-residual LayerNorm arrangement. Add a classification head as a separate downstream use.

**Resources:**

- [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer): annotated component implementation and training examples.
- [Original GPT-1 implementation](https://github.com/openai/finetune-transformer-lm/blob/master/train.py): historical architecture authority.

The annotated Transformer is a component-learning resource, not a faithful GPT-1 implementation. Use the original GPT-1 source to resolve model-specific details; write your own modern PyTorch version of its older TensorFlow code.

**Local exercise:** Overfit a short sequence, then check that the task head receives gradients and can fit a tiny classification dataset.

**Completion:** Complete model, causal-mask checks, correct next-token target shift, and working task-head code.

#### 2. GPT-2

**Build:** A complete GPT-2-style model and pipeline, including pre-normalization blocks, final LayerNorm, token embeddings, positional embeddings, optimizer integration, checkpointing, and generation.

**Resources:**

- [Karpathy: build-nanoGPT](https://github.com/karpathy/build-nanogpt): step-by-step code history and linked code-along lecture.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): visual explanation.
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html): GPT and tokenizer lessons.
- [Raschka: LLMs From Scratch](https://github.com/rasbt/LLMs-from-scratch): complementary implementation, pretraining, and fine-tuning material.

**Local exercise:** Reuse P00's byte-BPE understanding, then document GPT-2 tokenizer/preprocessing differences and any compatibility claims. Save/reload complete checkpoints and compare generation with/without a KV cache. Extend P01's pipeline with GPT-2-specific coverage rather than repeating the foundation build.

**Completion:** Reproducible forward/loss/backward execution, checkpoint round trip, and cache parity.

**Optional GPT-3-style research exercise:** Sweep several small model configurations and token budgets. Measure learning curves and resource use. Treat this as a scaling study, not an exact GPT-3 reproduction. Stanford's [CS336 scaling material](https://cs336.stanford.edu/) provides the research framework; use local experiments if its assignment API is unavailable.

#### 3. Llama 2 → Llama 3

**Build:** A complete Llama-style model. Implement RMSNorm, RoPE, SwiGLU, bias choices, grouped-query attention, and KV caching. Keep the Llama 2 and Llama 3 configurations separately identifiable.

**Resources:**

- [Umar Jamil: Llama from scratch](https://github.com/hkproj/pytorch-llama): code and linked explanation.
- [Raschka: converting GPT to Llama](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama): incremental Llama notebooks.

**Local exercise:** Compare normalization, positional encoding, and feed-forward alternatives one at a time. Check cached generation positions carefully.

**Completion:** Complete model, correct GQA head mapping, rotary-position tests, and cache parity.

#### 4. Qwen3 Dense

**Build:** Your own complete Qwen3 implementation, including its query/key normalization, rotary attention, grouped-query attention, feed-forward blocks, and configuration details.

**Resources:**

- [Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch).
- [Public Qwen3 dense and MoE notebooks](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3).

The article is subscriber-only; the companion code is public. Use the tiny configuration for local work. Matching a released checkpoint is an optional later verification exercise.

**Local exercise:** Copy identical small weights into your implementation and a trusted reference configured equivalently; compare logits and selected gradients.

**Completion:** Complete model, normalization/head-shape checks, and reference agreement within a documented numerical tolerance.

#### 5. Switch Transformer → Qwen3 MoE

**Build:** First a complete Switch-style language model with top-1 routing, then a Qwen3-style MoE model with multiple selected experts and weighted output combination.

**Resources:**

- [LabML: annotated miniature Switch Transformer](https://nn.labml.ai/transformers/switch/index.html).
- [LabML: training walkthrough](https://nn.labml.ai/transformers/switch/experiment.html).
- [Qwen3 MoE notebooks](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3).

**Local exercise:** Use a slow token-by-token expert dispatcher as the correctness reference. Implement a batched dispatcher and compare outputs and gradients. Log expert selection and, where used, capacity overflow and dropped tokens.

**Completion:** Routing and combination are correct; selected experts and router weights receive the expected gradients. An expert need not receive gradients in a batch where no tokens select it.

#### 6. Qwen3-Next

**Build:** A complete hybrid MoE model with Gated DeltaNet and periodic gated full-attention blocks. Implement recurrence, convolutional state where required, and attention-cache management.

**Resources:**

- [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms): hybrid architecture and recurrence explanation.
- [Gated DeltaNet implementation material](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet).

**Local exercise:** Start with a readable sequential recurrence. Compare full-sequence and step-wise execution before implementing chunked execution. Measure recurrent-state storage separately from the growing full-attention cache.

**Completion:** Complete hybrid model, correct state reset, and agreement between execution modes under matching semantics.

#### 7. PaliGemma

**Build:** A complete small vision-language model: vision encoder, modality projector, multimodal token assembly, attention masks, and language decoder.

**Resource:** [Umar Jamil: PaliGemma from scratch](https://github.com/hkproj/pytorch-paligemma), with a linked explanatory video and component code.

**Local exercise:** Use generated images containing simple shapes. Verify patch ordering, image-token insertion, and gradients through the vision path. Add a small training loop; the linked implementation primarily demonstrates model construction and inference.

**Completion:** End-to-end multimodal forward/backward execution. A tiny learning check should establish that image information can influence the task result; changing random inputs alone is insufficient evidence of useful visual understanding.

#### 8. Qwen3.5

**Build:** The complete dense hybrid text model first, then its vision-language path. Preserve the model's own normalization, gating, position handling, and state requirements rather than assuming the preceding hybrid implementation is identical.

**Resources:**

- [Raschka: Qwen3.5 implementation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/16_qwen3.5).
- [Tiny Qwen](https://github.com/Emericen/tiny-qwen): readable implementation covering the Qwen3.5/3.6/3.8 generation.

Raschka's example reuses some Hugging Face linear-attention helpers. Reimplement those helpers as an explicit exercise if the goal is to understand every component.

**Local exercise:** Trace text-only prefill, multimodal prefill, and incremental decoding. Check that a fresh request resets every relevant state.

**Completion:** Complete model with documented text/vision scope and working state management.

#### 9. Qwen3.8-27B-style model

**Build:** A complete miniature dense hybrid vision-language model following the released architecture, including its multi-token prediction component.

**Resources:**

- [Official model card](https://huggingface.co/Qwen/Qwen3.8-27B).
- [Released configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
- [Tiny Qwen reference implementation](https://github.com/Emericen/tiny-qwen).

**Local exercise:** Check ordinary next-token prediction, the additional MTP training path, vision integration, and cached generation independently before combining them. Implement a minimal tool-call parser/harness as a separate system exercise.

**Completion:** Complete model and verified integrated behavior. Record which released mechanisms are implemented and which remain deferred. A random or lightly trained replica is not evidence of frontier reasoning or agent capability.

#### 10. Gemma 4 E2B-style model

**Build:** A complete text model with local/global attention, cross-layer KV sharing, and per-layer embeddings. Use a small auxiliary embedding dimension.

**Resources:**

- [Per-layer embeddings explanation](https://sebastianraschka.com/llm-architecture-gallery/per-layer-embeddings/).
- [Raschka: standalone Gemma 4 notebook](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/17_gemma4).
- [Official E2B model](https://huggingface.co/google/gemma-4-E2B).

**Local exercise:** Compare cache ownership and storage with sharing enabled and disabled. Check the layer-specific embedding slice and gate for each block. The linked notebook is text-only; multimodal work is a separate extension.

**Completion:** Correct local/global masks, cache-sharing relationships, and per-layer embedding paths.

#### 11. DeepSeek V3 → V3.2

**Build:** A complete MLA/MoE language model with shared and routed experts. Include V3's load-balancing strategy and MTP as independent components, then add V3.2-style sparse indexing.

**Resources:**

- [Official DeepSeek V3 implementation](https://github.com/deepseek-ai/DeepSeek-V3).
- [MLA walkthrough/code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla).
- [DeepSeek Sparse Attention walkthrough/code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa).

**Local exercise:** Verify latent-cache reconstruction and position handling. For sparse attention, compare against a simple gather-based reference using the same selected indices. Test that the indexer cannot select future positions.

**Completion:** Runnable V3-style and V3.2-style configurations, correct cache behavior, and indexer checks.

#### 12. DeepSeek V4-Flash-style model

**Build:** A complete model with Compressed Sparse Attention, Heavily Compressed Attention, and manifold-constrained hyper-connections. Prototype residual mixing with two streams; complete the exercise with the released design's four-stream arrangement at small width.

**Resources:**

- [Official V4-Flash architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).
- [KV sharing, mHC, and compressed-attention walkthrough](https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures).
- [CSA/HCA explanation](https://sebastianraschka.com/llm-architecture-gallery/csa-hca/).
- [mHC explanation](https://sebastianraschka.com/llm-architecture-gallery/mhc/).

**Local exercise:** Check compression boundaries and causal visibility. Verify non-negativity and approximate row/column sums for the constrained residual mixing matrix. Test residual-stream shapes and gradients.

**Completion:** Complete generating model with correct compressed state and constrained mixing. Muon is a separate optimizer exercise after model correctness.

#### 13. Qwen3.8-Flash-Next-style model

**Build:** A complete hybrid MoE model with Qwen Sparse Attention, small n-gram embedding tables, gated residual machinery, and MTP.

**Resources:**

- [Official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- [Qwen Sparse Attention explanation and pseudocode](https://sebastianraschka.com/llm-architecture-gallery/qwen-sparse-attention/).

**Local exercise:** Implement selection over complete four-token micro-blocks and correct handling of the visible incomplete tail. Use tiny selection budgets and hash tables. Check n-gram construction at sequence boundaries and verify lookup gradients.

**Completion:** Complete model with correct block selection, lookup memory, residual behavior, and hybrid state. Report lookup-table storage separately.

#### 14. Kimi Linear → Kimi K3-style model

**Build:** A complete Kimi Linear-style model with Kimi Delta Attention and periodic MLA. Extend it toward K3 with gated MLA, Attention Residuals, latent-space experts, and its specified activation and other model-specific components. Add the vision path after text correctness.

**Resources:**

- [Kimi Linear material](https://github.com/MoonshotAI/Kimi-Linear).
- [Attention Residuals explanation and code](https://github.com/MoonshotAI/Attention-Residuals).
- [Kimi K3 specification and report](https://github.com/MoonshotAI/Kimi-K3).
- [Released K3 model card](https://huggingface.co/moonshotai/Kimi-K3).

**Local exercise:** Verify channel-wise recurrence gating, attention over earlier depth representations, and projection into/out of the expert latent space. Keep individual feature switches for numerical checks and later ablations.

**Completion:** Runnable configurations with an explicit coverage checklist against the published K3 specification. Label partial coverage accurately.

#### 15. DeepSeek V4.1-Flash-style model

**Build:** A complete small causal encoder–decoder. Implement encoder-derived shared KV state, CSA2 cache/index sharing, bounded replay, and small Engram conditional-memory tables. Begin with four encoder and four decoder layers.

**Resources:**

- [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
- [Official reference model code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).
- [Inference instructions](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md).
- [Engram implementation](https://github.com/deepseek-ai/Engram).

**Local exercise:** Trace cache ownership through encoder and decoder. Check causal masks on every path, including decoder access to encoder states. Compare cached execution and bounded replay against a simple reference. Add cache quantization only after float-state correctness; treat speculative decoding as a subsequent exercise.

**Completion:** Complete encoder–decoder model with verified cache ownership, causal behavior, and a documented coverage checklist.

Projects 14 and 15 are paper-driven capstones. Public components and reference implementations are available, but no complete, verified training code-along comparable to the early GPT tutorials was established during source review.


### Distinct research branches

#### 16. Mamba 2 → Mamba 3

**Build:** A complete state-space language model. Begin with the simpler Mamba 2/SSD reference, then implement Mamba 3's recurrence, complex-valued state tracking, and SISO/MIMO variants.

**Resources:**

- [Official Mamba repository](https://github.com/state-spaces/mamba): includes simplified Mamba 2 and minimal SSD components.
- [Tri Dao: Mamba-3 methodological deep dive](https://tridao.me/blog/2026/mamba3-part2/).
- [Community pure-PyTorch Mamba-3 implementation](https://github.com/VikramLex/mamba3-minimal).

**Target:** 1–3M parameters. Use a small state dimension and MIMO rank.

**Local exercise:** Compare sequential and chunked execution, state resets, and SISO/MIMO shapes. Use synthetic state-tracking tasks for a short learning check. Verify community code against the official mathematics/reference before using it as an authority.

#### 17. RWKV-7

**Build:** A complete recurrent language model with time/channel mixing and generalized delta-rule state updates. Implement the recurrence in ordinary PyTorch first.

**Resources:**

- [Official architecture documentation](https://github.com/RWKV/RWKV-wiki/blob/main/docs/basic/architecture.md).
- [RWKV-LM code](https://github.com/BlinkDL/RWKV-LM).
- [Pretraining tutorial](https://wiki.rwkv.com/advance/pretrain.html).

**Target:** 1–3M parameters.

**Local exercise:** Check step-wise generation against equivalent sequence execution. Measure persistent-state size and compare it with attention-cache growth. CUDA-dependent training examples are references for later execution; your initial implementation must have a CPU path.

#### 18. LLaDA → iLLaDA; optional LLaDA2.1

**Build:** A complete masked diffusion language model with bidirectional attention, corruption/masking, the appropriate masked prediction objective, and iterative generation. Add token editing as a separate LLaDA2.1 exercise.

**Resources:**

- [Official LLaDA/iLLaDA material](https://github.com/ML-GSAI/LLaDA): model/inference code and training pseudocode.
- [dLLM](https://github.com/ZHZisZZ/dllm): runnable adaptation, training, inference, and evaluation recipes.
- [LLaDA2.1 paper](https://arxiv.org/abs/2602.08676).

**Target:** 1–3M parameters, short sequences, and few denoising steps.

**Local exercise:** Check mask sampling, loss selection/weighting, immutable prompt positions during conditional generation, and token-editing behavior. Compare convergence on tiny tasks with an autoregressive baseline. Raw training losses across different objectives are not directly comparable.

The original LLaDA repository does not release the original complete training framework and data. Its pseudocode and the public dLLM recipes support an educational implementation, not an exact training reproduction.

#### 19. Gemma 4 12B-style unified model

**Build:** A complete miniature multimodal model that projects image patches and audio frames directly into the language backbone, without separate deep modality encoders.

**Resource:** [Google's Gemma 4 12B architecture guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/).

**Target:** 2–4M parameters with small images and short clips.

**Local exercise:** Verify patch/frame ordering, modality positions, mixed-sequence masks, and gradients through the projection layers. Compare interface and resource costs with your encoder-plus-projector model from project 7. Use a tiny learning check for modality-dependent behavior.

This is an independent multimodal implementation project guided by the published architecture, not an established full from-scratch training tutorial.


### Required CPU verification protocol

Every model should support the following checks at a tiny configuration.

#### Forward and backward correctness

- Assert shapes at embeddings, projections, heads, expert dispatch, residual streams, and output logits.
- Check that outputs, losses, and expected gradients are finite.
- Compare selected components with a simple reference using identical weights and inputs.
- Use FP64 finite-difference/gradient checks on very small differentiable inputs where appropriate.
- Record the numerical tolerance and maximum error. Choose tolerances based on dtype, operation, and sequence length rather than a universal threshold.

#### Causal behavior

For causal models, alter future tokens while retaining the prefix. Earlier outputs should remain unchanged within tolerance. Disable dropout and use identical evaluation settings.

Apply this check to sparse indexers, compression, convolution, n-gram construction, and encoder–decoder access as well as ordinary attention. Diffusion models intentionally use bidirectional attention; test their conditional-generation restrictions instead.

#### State and cache behavior

- Compare full-sequence, chunked, and token-by-token execution where the formulations should be equivalent.
- Test empty/new requests, state reset, sequence boundaries, and multiple batch elements.
- Match masks, position policies, and sliding windows between reference and cached versions.
- Remember that dropping history from a full-attention cache changes the available context; compare against a reference with the same semantics.

#### Short learning check

When feasible, run a small number of optimization steps to overfit a handful of examples or solve a synthetic task. Record the loss curve and held-out/task behavior where meaningful.

This checks that the model and optimization path work. It does not establish general language capability, useful reasoning, or reproduction of the original model's performance.

#### Checkpoint and configuration checks

Save/reload the state and configuration, then compare outputs. Count parameters from the actual model. Record every simplification relative to the published architecture.


### Optimization track

Keep both a clear reference implementation and an optimized implementation. Every optimization needs numerical checks before performance interpretation.

| Exercise | What to measure |
|---|---|
| Preallocated KV cache versus repeated concatenation | Allocation count, memory, prefill/decode time. |
| Sequential versus chunked recurrence | State/output/gradient agreement and runtime. |
| Token-wise versus batched expert dispatch | Routing agreement, selected-expert gradients, and runtime. |
| Separate versus combined projections | Numerical error, allocations, and runtime. |
| Tensor layout and unnecessary copies | Copy/contiguous operations and memory use. |
| Shared versus independently stored caches | Ownership correctness and bytes per token. |
| Dense versus sparse selection references | Correct selected positions, masking, and resource cost. |
| Float versus simulated quantized state | Reconstruction error and model-output changes. |

Use the [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). Record thread count, dtype, dimensions, warmup, and repeated timings. Measure prefill and incremental decoding separately.

CPU profiling establishes results for the local CPU implementation. GPU kernel performance and distributed scaling require their own later measurements. Quantized storage alone does not demonstrate faster execution.


### Training, adaptation, and reasoning projects

These complement the model implementations. Develop their algorithms and tiny local checks now; substantial runs can wait for suitable compute.

| Project | Implementation work | Resources |
|---|---|---|
| Tokenization and data | Core work now belongs to P00: byte/BPE training, encode/decode, filtering, deduplication and reproducible splits. Later models add tokenizer-specific compatibility. | [minbpe](https://github.com/karpathy/minbpe), [BPE paper](https://arxiv.org/html/1508.07909v5), [Stanford CS336](https://cs336.stanford.edu/); retained supplementary [BPE walkthrough](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html). |
| Pretraining systems | Data loading, optimizer state, gradient accumulation, checkpoints, resource accounting, and profiling. | [build-nanoGPT](https://github.com/karpathy/build-nanogpt), [LLMs From Scratch](https://github.com/rasbt/LLMs-from-scratch), CS336 assignments 1–2. |
| Instruction tuning | Chat formatting, assistant-token loss masking, and an SFT loop. | [Raschka chapter 7 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07), [Hugging Face Smol Course](https://github.com/huggingface/smol-course). |
| LoRA and DoRA | Low-rank updates, frozen/base parameters, adapter merging, and gradient checks. | [LoRA/DoRA walkthrough](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch). |
| Preference optimization | Preference pairs, policy/reference log probabilities, and DPO loss. | [DPO from scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb). |
| Reasoning and RL | Verifiers, rollout handling, GRPO, distillation, and inference-time selection. | [Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch), CS336 assignment 5. |

For a tiny untrained model, test loss equations and use simple synthetic rewards. Meaningful reasoning experiments usually need a competent pretrained starting policy. Implementing a reasoning objective does not by itself create a reasoning model.


### Recent releases: implementation priority

The strongest final architecture projects in this roadmap are the miniature Kimi K3 and DeepSeek V4.1 builds. Their selection is a learning recommendation, not a claim that they are the newest releases in every category.

- **GLM-5.3:** Its [official card](https://huggingface.co/zai-org/GLM-5.3) states that it uses the same base model as GLM-5.2 and attributes gains to post-training. Use it as inspiration for a controlled post-training study.
- **Reflection Beam:** The [October 5 release](https://reflection.ai/blog/introducing-beam) describes interleaved local/global attention and fine-grained MoE. Use its training and balancing choices as a comparison once those mechanisms are implemented.
- **Mistral Large 4:** At the cutoff date, [official documentation](https://docs.mistral.ai/resources/changelogs) lists a public preview and says open weights are coming soon. Add an implementation project after sufficient public artifacts become available.

Public code, open weights, and complete training reproducibility are different properties. Some projects provide public model/inference code without the original data or complete training pipeline. Resource descriptions above identify the relevant gaps. Model and code licenses also vary; Kimi K3, for example, releases weights under its own license.


### Source-review and verification status

The linked resources and selected release claims were researched during preparation of this roadmap on October 9, 2026. Resource availability, model cards, and public repository material were inspected. Some long-form articles are paywalled, and their public companion code is provided where available.

No model implementations, training runs, numerical checks, or performance benchmarks were executed as part of producing this document. The parameter budgets, study ordering, proposed exercises, and completion gates are recommendations for the upcoming work.

For durable research, record the exact repository commit or model revision you use when starting each project. Update the release cutoff explicitly when extending this roadmap later.
