# Projects progress

## Separate Project Docs and Tutorials — 2026-10-30

Projects Docs navigation (2026-10-30): the homepage and sidebar now separate Project Docs from Tutorials, with Foundations, Transformer Labs, Language Models, Quantization Lab and Attention Labs collections in both sections. Tutorial collection pages preserve all existing course/chapter URLs and authored ordering; record search has its own page. All 808 tutorial files are byte-preserved. Project, learner and hosted checkpoints are unchanged.

Project documents continue under /projects/; tutorials now browse through /tutorials/{collection}/ before opening their existing lesson URLs. Search is at /search/. The [delivery record](../implementation/2026-10/separate-project-docs-and-tutorials-by-collection/00-resume.md) owns route, compatibility and browser evidence. This is reader organization; implementation and learner progress are preserved.


## Attention Labs inline presentation visuals — 2026-10-30

Attention Labs visual revision (2026-10-30): all ten Attention Foundations/Heads lessons now include 27 attributed diagram screenshots captured in Chrome from the presentation websites linked by the original lectures. Each appears beside its matching explanation with a full-size link and exact slide provenance; complete previews avoid cropped blank margins on mobile. The checked numerical examples and authored figures are retained, source-example differences and simplified claims are explained, and Lecture 9 remains excluded. Project implementations and learner/hosted checkpoints are unchanged.

Continue at the [Attention Foundations tutorial](../tutorials/projects/attention-labs/attention-foundations/v1/lessons/01-context-and-map.md) or [Attention Heads](../tutorials/projects/attention-labs/attention-heads/v1/lessons/01-independent-heads.md). The [visual revision record](../implementation/2026-10/attention-labs-playlist-tutorial/08-visual-revision.md) owns source capture, placement and browser verification. This addresses tutorial presentation; attention implementation remains at its prepared boilerplate checkpoint.


## Attention Foundations build/test project — 2026-10-30

Attention Foundations project preparation (2026-10-30): the existing attention-labs/attention-foundations child now has an explicit goal to build and independently test six basic single-head attention mechanisms, a six-phase route, locked Python 3.12.13 source boilerplate, workspace runners and a source-upload Colab notebook. All 18 scaffold checks and the six operational notebook cells pass locally; only the Colab file interface is simulated. Attention implementations, actual Google-hosted execution and learner review remain pending. MHA/GQA stay in attention-heads, shared tutorials stay in Docs, and existing learner/model checkpoints are preserved.

Start from the [project goal](../../projects/attention-labs/attention-foundations/README.md) and [Colab instructions](../../projects/attention-labs/attention-foundations/COLAB.md). Next implementation action: specify one additive query/key/value retrieval, then implement and independently verify it. The [current setup record](../../projects/attention-labs/attention-foundations/docs/implementation/00_setup-build-goals-and-colab-runner/00-resume.md) separates preparation from learning and hosted evidence.


## Attention Labs shared tutorials — 2026-10-30

Attention Labs preparation (2026-10-30): Attention Labs is an independent collection repository/submodule with one attention-foundations folder for Lectures 1–7 and a separate attention-heads folder for MHA and later GQA. Ten detailed lessons live in the shared Docs tutorials/projects/attention-labs tree. All eight English automatic subtitle/transcript texts and all 181 original slides were reviewed; each included lecture retains its original presentation viewer/link. The explanations, numerical fixtures, diagrams and Python/C++ references are original. Lecture 9 is excluded. Existing model source and learner/hosted checkpoints are unchanged.

Start with [Attention Foundations](../tutorials/projects/attention-labs/attention-foundations/v1/lessons/01-context-and-map.md), then continue to [Attention Heads](../tutorials/projects/attention-labs/attention-heads/v1/lessons/01-independent-heads.md). The [owning record](../implementation/2026-10/attention-labs-playlist-tutorial/00-resume.md) separates source review, author checks and reader delivery. Next: review the worked route and its explanation prompts; no new learner milestone is recorded.

## Local source handoff and generated-file cleanup — 2026-10-23

The user requested clean local commits across the workspace. The Attention source/Colab organization, source-backed tutorial and both Docs highlighting fixes are reviewed and verified. Generated offline builds, browser-test captures and Python caches are archived with member hashes before removal; authored study records, media, source, dependencies and scientific results are retained. Both Docs readers stay available. The coordinated CLI record is `cli/docs/implementation/2026-10/attention-and-docs-source-cleanup/`, with final owner heads and parent-pointer audit in workspace `.ai/source-clean-commit/20261024T010936Z/final.json`. No push, history rewrite, training or learner milestone follows from this source handoff.

## Code highlighting across both Docs readers — 2026-10-23

The common Projects/Research Markdown renderers now produce syntax colors for declared languages, including workspace shell commands and Research JSON provenance. The shared theme supports readable dark/print output and accessible scrolling without changing copied source. All 21 Projects and 30 Research tests, builds and full browser suites pass, with additional root/prefix desktop/mobile syntax checks and actual managed-reader verification. Both readers are running. [Evidence and scope](../implementation/2026-10/syntax-highlighting-across-projects-and-research-docs/00-resume.md); no model, media pin, learner or scientific result changed.

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

## Foundation Lab: Bike Demand Planner tutorial conversion, 2026-10-20

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

## Bike Demand Planner Phase 02 — implementation ready for review, 2026-10-19

Implemented training-only hour/working-day means, a global-training-mean fallback and a same-contract comparison with the preserved constant predictor. The 48 observed groups retain values/support counts; the shared runner retains aligned train/validation predictions and versioned provenance. Validation MAE is 91.5729 versus 154.2452, RMSE 137.3900 versus 208.1723, with remaining signed underprediction of −72.3422. Full comparison and failure analysis are in the [private current record](../../projects/foundations/bike-demand-planner/docs/implementation/03_simple-predictions-and-hourly-error-metrics/00-resume.md).

All 38 code checks, independent predictions/metrics/identity/hash checks, installed-wheel group/fallback checks, export privacy and setup pass. Written learning records and Phase 02 implementation were finished before authoring four short companions (3:29–3:54), now embedded before the matching prose. All four native players pass root/prefixed playback, pause, middle/later seeking, captions, desktop/mobile layouts and private-export checks; 20 narration receipts completed and the teaching text/source pins are preserved. The reserved test is closed, model work remains uncommitted and staged work is preserved. Learner review is pending; Phase 03/P00/hosted gates are unchanged.

## Inline section videos with deeper worked examples — 2026-10-19

The [hourly-data understanding notes](../../projects/foundations/bike-demand-planner/docs/implementation/02_hourly-data-audit-and-splits/02-notes-to-understanding.md) are the watch/read entry: six captioned players sit immediately above their matching prose. They are now expanded to 4:42–6:34, with 18 additional scenes and about 11 minutes covering another request, missing-hour arithmetic, exact vector/dot-product changes, circular distance and coefficient dependence, delay-sensitive readiness, and observation/schema replay. The notes were expanded first. Existing film/cuts and source snapshots remain preserved.

The [project-learning-video skill](../../../ai_research/research_docs/.agents/skills/project-learning-video/SKILL.md) reuses project-learning and the synchronized owner video workflow. CLI and reader evidence live in the [section-video package](../../../cli/docs/implementation/2026-10/section-videos-embedded-in-understanding-notes/00-resume.md). Local media remains ignored. Bike Phase 02 still resumes with the hour/working-day baseline; no model training, learner mastery or hosted gate is advanced.

## Grouped project documentation reader — 2026-10-19

The local reader now has collection navigation (Foundations, Transformer Labs and Language Models), project pages, separate Phases and Implementation folders, and numbered learning records with filenames and document titles. It discovers existing project READMEs and ignored project docs without moving source files. Relative note/code links, Mermaid, equations, downloads and the existing data video are available locally. Default static exports contain project READMEs only.

The [reader implementation package](../implementation/2026-10/grouped-project-documentation-reader/00-resume.md) owns the checks and any remaining reader work. Bike remains in Phase 02 with its hour/working-day baseline next; model code, staged work, learner understanding and hosted evidence are unchanged.

## Bike data walkthrough video companion — 2026-10-19

The requested complete explanation is delivered as a [23:03, 1080p narrated video](../../../.ai/videos/projects/bike-hourly-data-calendar-split/out/video.mp4), with 37 scenes, six chapters, animated vectors/timelines, code highlights, runnable commands and SRT/VTT captions. The [private companion record](../../projects/foundations/bike-demand-planner/docs/implementation/02_hourly-data-audit-and-splits/video-production-2026-10-19.md) links the transcript and measured verification. Visual/timing checks and the finished audio/video checks pass. Original source and staging are preserved; media stays local and ignored, with no remote publication.

This is a companion to the completed data handoff. Current modelling remains Phase 02; the separate training-only hour/working-day baseline and fallback comparison are still next. No new learner explanation, model result or hosted gate is inferred from video production.

2026-10-19 shared tooling: added the CLI-owned tutorial-video capability and synchronized owner skills, with local protocol/rendering checks recorded in `../../../cli/docs/implementation/2026-10/shared-tutorial-video-cli-and-synchronized-ai-skills/`. No learner, model-training or hosted gate was advanced.

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

## Classical ML foundations README preparation — 2026-10-15

The user selected two sequential classical ML applications: [Bike Demand Planner](../../projects/foundations/bike-demand-planner/README.md), then [E-commerce Purchase Predictor](../../projects/foundations/ecommerce-purchase-predictor/README.md), under the new [Foundations collection](../../projects/foundations/README.md). The collection README explains the shared study route; each project README defines its real-world application, data/provenance and leakage contract, phases, worked gradient calculation, optimizer/loss/rate/batch/regularization experiments, evaluation, artifacts, recovery and completion evidence.

Current delivery is documentation preparation only. No datasets, environments, model implementations, training, applications or CLI/TUI integration have been created for these projects. The selected next practical project is Bike Demand Planner, phase 0: define the forecast issue time/horizon and allowed inputs, then explain/check one linear update. E-commerce follows the bike application's completion review. Later deep learning and embedding work follows separately; the existing router is preserved. Earlier P00 setup completion and pending `linear()` are retained as evidence/reference, not advanced or erased. Earlier next-action statements below describe the prior route.

[Preparation checkpoint](../implementation/2026-10/classical-ml-foundations-project-readmes/00-resume.md). All phases and learner/hosted completion gates remain pending.

## 2026-10-13 — Support Ticket Router standalone entry points and central presentation

Corrected the boilerplate at the user's request: restored the visible Run label; moved presentation metadata and the existing tooling package into the independent cli repository; removed the project-local command dispatcher; added standalone demo/setup/configuration/data/tokenizer/embedding/classifier/training/comparison/inference/retrieval files with explicit inputs in their main blocks. The model project contains no terminal UI code, metadata or references. Its learning and application stages remain unimplemented.

The shared registry supports centrally owned profiles while preserving legacy project-local profiles and all source-path/argv validation. Run launches the selected source; Checks runs infrastructure tests. Verification: 37 project tests, 102 shared CLI tests, direct source files and managed demo from unrelated cwd, shared TUI smoke and successful source package/wheel build. [Tooling checkpoint and tracker](../../../cli/docs/implementation/2026-10/support-ticket-router-boilerplate-and-tailored-tui/00-resume.md) record passing compact/wide interaction, isolated wheel execution, clean artifacts and documentation/whitespace checks.

This completes infrastructure correction only. Data acquisition, tokenizer/model implementation, training, predictions and hosted recovery remain pending. P00-M1 and linear() next are unchanged. The router's later goal remains: train embeddings and an intent classifier; compare word tokenization, BPE and subwords.

### Phase planning follow-up — 2026-10-13

Created the requested [phase overview](../../projects/transformer-labs/support-ticket-router/phase.md) and ignored local `plan/` folder with ten phase-specific Markdown requirements and a checkpoint. Plans cover data/labels, splits/baseline, word tokens/batches, joint model, hosted recovery, tokenizer comparisons, Word2Vec, character representations, inference/retrieval and the final application/evaluation. Phase 1 now names its source/schema, explicit inputs, saved artifacts, validation failures and deterministic label-map checks. Requirements and whiteboard explanations precede assistant-written code/tests and joint review, with chunks adapted to each phase. This is planning only; no dataset acquisition or phase code/training was performed.

## Tutorial 1 topic preparation — 2026-10-13

Tutorial 1 preparation (2026-10-13): [Course 1](../tutorials/projects/p00/transformer-course/v1/lessons/01-start.md) now follows a 32-topic mathematical implementation route through translation, RoBERTa MLM and extractive QA, with worked gradients, source contracts, 64 changed-condition practices and editable topic notes. The outline was saved first and bounded agents authored/reviewed the topics. Existing chapter IDs, Tutorial 0/Small GPT and learner source files are preserved. Current learning remains P00-M1 setup complete, with linear() next in the relocated foundations lab; learner mastery and hosted recovery remain pending. No commit or publication. [Outline](../tutorials/projects/p00/transformer-course/v1/project/course-outline.md), [study notes](../tutorials/projects/p00/transformer-course/v1/project/study-notes.md) and [verification package](../implementation/2026-10/transformer-foundations-translation-and-roberta-course/00-resume.md).


## 2026-10-13 — ordinary grouped Projects folders

All four existing project folders are now under `projects/transformer-labs/`, owned by ai_projects rather than nested repositories/submodules. `projects/language-models/` is prepared for later Small GPT, GPT-2 and other language-model builds. CLI discovery retains the project IDs, and future Projects creation defaults to language-models with an explicit transformer-labs option. The TUI shows collapsible collection headers with their child projects, an empty Language Models state, temporary search expansion and stable fold/file selection across refreshes and resize. Documentation and Research keep their existing ownership. Recovery bundles, working-file archives and a verified move manifest are retained at workspace `.ai/project-layout-migration/2026-10-13/`. [Migration checkpoint](../implementation/2026-10/group-transformer-labs-as-ordinary-projects-folders/00-resume.md).

Actual learning stays P00-M1 setup complete, with linear() next in the relocated foundations lab. Semantic Notes Search and Tiny Vision Transformer remain planned; Language Models has no implementation yet. No training, hosted gate, commit or publication is implied.

## 2026-10-12 — Tutorial 0 completely rewritten by topic

The user rejects the eleven-page narrative and requests all foundations for Attention Is All You Need, with an outline first and lessons written one by one. The 25-topic outline precedes the complete rewrite. Core maths, probability/information, networks/initialization/backprop/loss/optimizers, data/encoding/embedding/position, attention/masks/heads, FFN/residual/LayerNorm, regularization/encoder–decoder learning, evaluation/recovery and paper assembly have dedicated substantial lessons. Three explicitly requested agents handle final lessons, numerical probes and independent coverage review, with main integration.

114 source checks and all 26 standalone lesson Python blocks pass, including independent gradients and information-flow checks. Notes cover all 25 topics; all 9 cleared-study code cells execute from the verified source ZIP. Six publisher examples and nine reader content tests pass. [The rewrite package](../implementation/2026-10/tutorial-0-prerequisite-foundations-rebuild/00-resume.md) records final source/reader/preservation evidence and the corrections found by review. The reader overview uses the topic route; the former fixed eleven-lesson assertion is metadata-derived.

The learner child still resumes linear(), now fully specified in Lesson1 with the old Lesson3 URL retained. Later courses/children, prior notes exports and TUI reference-row removal are preserved. This supersedes prior course structure, not the actual P00-M1 setup-only checkpoint or model/learner/hosted completion gates. No commits/publication.


## 2026-10-12 — remove the reference exercise from the TUI

Hidden the automatic P00 reference shortcut from TUI navigation, filtering, counts and default selection. Docs contains the two documentation sites; Projects retains both implementation repositories. Tutorial 0 remains in the documentation reader and the existing explicit p00 CLI commands remain available. CLI tests and TUI smoke verification are recorded in the owning [TUI package](../../../cli/docs/implementation/2026-10/project-centric-tui-and-grouped-documentation/00-resume.md). Course files, child implementations and learner checkpoint are unchanged.

## 2026-10-12 — written maths and study notes in Tutorial 0

Strengthened the existing eleven-lesson prerequisite route using Nielsen's chapters 1–3 and D2L's reverse computation-graph explanation. Original examples now connect affine offsets, shared gradients, local descent, likelihood/NLL, output cross-entropy cancellation, a complete 2→2→2 Tanh backward pass and the opposite-sign task's need for nonlinearity. Four added native practice reveals ask for changed-input reasoning.

Every lesson has a topic-specific notes prompt and stopping point. The editable Markdown workbook and cleared study notebook follow the same lesson order; notebook probes import the checked standard-library calculations. The source exporter delivers both study assets beside the unchanged miniature trainer and separately generated Colab notebook, protecting existing personal notes from overwrite.

Eighty source checks and six publisher examples pass; complete controlled and unequal-width gradients agree with CPU float64 autograd, and all five study code cells execute from an isolated extracted source ZIP. [The written maths review](../implementation/2026-10/tutorial-0-prerequisite-foundations-rebuild/07-written-maths-review.md) records reader/mobile/visual checks and exact preservation evidence. All 133 later-course files and both implementation children remain unchanged. Learner P00-M1 remains Lesson 2 setup complete with Lesson 3 part A linear() next; hosted recovery/mastery remain pending. No commits or publication.


## 2026-10-12 — Tutorial 0 rebuilt as prerequisites

Rebuilt the existing eleven-lesson foundations route as Tutorial 0 — Maths and PyTorch foundations, ahead of translation/RoBERTa and Small GPT. Advanced Python is assumed; syntax and general programming primers are removed. The route teaches matrix maps/statistics/probabilities, tensor contracts, gradients and optimizer history, text coordinates, batching, neural modules, a 42-parameter classifier and complete-state checkpoint/recovery. Fifteen original diagrams, fourteen native practices and three explorations support the calculations. Separate source modules and a cleared notebook use the same bounded CPU trainer.

Local verification and preservation evidence are recorded in [the rebuild package](../implementation/2026-10/tutorial-0-prerequisite-foundations-rebuild/00-resume.md). The previous complete tutorial is archived; later course bytes and both child repositories are preserved. Tutorial 0 → Course 1 → Course 2 is the reader order, retaining P00/P01 and stable lesson URLs. Learner P00-M1 remains Lesson 2 setup complete with linear() next; hosted/mastery gates remain open. No new commits or publication.

## 2026-10-12 — topic-based Transformer and RoBERTa course

Authored one 24-lesson Course 1: Transformer Foundations: Translation and RoBERTa. Five source videos inform shared prerequisites and two complete builds, rather than five separate tutorials. The course includes original matrix/shape/flow diagrams, worked derivatives/losses, complete Python reference modules, native practice/controls and separate project/source/results/readiness records. Small GPT becomes Course 2 with its P01 identity and paths preserved; older foundations remain reference material.

Local verification: 71 component/session checks, exact translation/MLM/QA fresh-process continuation, two reproduced bounded training runs, source/runtime identity rejection, verified source ZIP and output-cleared Colab notebook. Nine reader tests pass; browser navigation/maths/diagrams/practice/download/print/mobile checks cover all 84 records. Final root/prefix outcomes and visual corrections are recorded in [the course package](../implementation/2026-10/transformer-foundations-translation-and-roberta-course/00-resume.md). Existing attention child and its signed commit are unchanged; no new commits, learner repositories or hosted runs are part of this tutorial work. P00-M1 remains in progress with Lesson 2 setup complete and linear() next.

## 2026-10-12 — video 1 verified and committed

Reviewed the retained first-video reference notebook and closing caption summary against the implemented stages: raw attention, learned projections, separate and vectorized heads, padding, causal/combined masks, cross-attention and the final efficient Attention class. GitHub main still resolves to 2f18a5470c4ef0a3d413dd7b18d07941f45779e9. Added the explicitly missing runnable/notebook example with Q = K = V = X, its hand-calculated check and two numbered matrix displays. Existing final API/projection names, later-file names and display-module separation are preserved. Removed the now-unnecessary tests/.gitkeep placeholder; raw sources/captions and ignored generated files are retained.

Verification: 25 attention checks plus ten matrix-example checks pass on CPU. All 37 notebook cells execute with 28 saved figures and widget state, and both added diagrams were visually inspected. Direct attention.py and the workspace Attention launcher finish with PASS; maximum explicit/SDPA demo difference remains 1.19e-7 at seed 7. The refreshed 30-file source ZIP has verified archive/member hashes; an isolated extracted source copy creates a fresh Python 3.12.13 environment and passes all 35 tests plus the demo. Lock validation, staged whitespace, source/license/caption hashes, practical links, resource paths and ignore rules pass. Python code contains no video narration or notebook plotting definitions.

Created signed local commit 475c6347104c1de9a8c158171b9f6f3228d1aba3 (Implement attention mechanisms and visual walkthroughs) in the independent attention project. SSH signature verification passes with the configured public signing key; the child working tree is clean. Only its source/notebook/tests/license/practical docs were committed. Existing parent and unrelated workspace edits are preserved; no remote publication or parent commit was performed. This user request supersedes the earlier no-commit boundary. Next is video 2 data preparation when requested; translation training, hosted checks and the foundations learner checkpoint are unchanged.

## 2026-10-11 — explicit matrix walkthroughs and separate display module

Moved plotting, widget construction and auxiliary visualization calculations into attention_visuals.py. Notebook display cells now make short function calls alongside the original attention examples. Ten new traditional matrix equations cover A×B, XW+b projections into Q/K/V, key transpose, QKᵀ, scaling/row-wise softmax, weights×values, head merge and output projection. Each multiplication includes numbers, dimensions, highlighted rows/columns and a scalar expansion. Controlled projections use the actual attention classes; the same Q/K/V values connect through to the context matrix. Display-example construction preserves the caller's random state.

Verification: all 35 notebook code cells execute with 26 saved figures and serialized widget state. The 24 attention checks and ten matrix-example checks pass, including independent hand-calculated projection, softmax/context and output answers. All ten new figures were visually inspected. The four extracted widget callback families pass 18 control changes, and all sixteen prior embedded figures are byte-identical after extraction. Hashes of all fifteen pre-existing Python files and sources of all twelve original notebook example cells are unchanged. The source ZIP includes the display module and its checks; its 31 members and hashes match the current files. Practical docs, launcher resources and the six compact snapshots describe the separate display module. No commits, training, hosted execution or new child work logs. The foundations learner checkpoint and next data-preparation work are unchanged.

## 2026-10-11 — attention notebook visuals

Added sixteen original Matplotlib figures to the existing attention concept notebook, including annotated dot-product cells, scores/softmax/context, weighted-value bars, query/value geometry, a controlled scaling experiment, Q/K/V flow, head reindexing and weight matrices, padding/causal traces, a same-parameter future-token comparison, rectangular cross-attention, explicit/SDPA output differences and gradient flow. Four widget control families inspect matrix cells, query angle/temperature, heads/query rows and padding/causality. Plotting helpers are collapsible; existing calculations and Python implementations stay separate. Matplotlib/ipywidgets belong only to the optional notebook extra.

Verification: all 25 code cells execute, with 16 saved plot outputs and serialized widget state. Numeric assertions cover weighted sums, score cells, row normalization, head reshape recovery, allowed masks, causal invariance and path agreement. An additional temporary QA cell exercised 28 plot calls and all four widget callback families, including all nine dot-product pairs, multiple query/temperature cases, all mask combinations and head selection. Every figure was visually inspected; crowded titles/labels and a backward-arrow placement were repaired. All Python source/test file hashes match the start of this visual edit; the original commit and unstaged state are preserved. The 29-file source export has matching member/archive hashes and includes the executed notebook. Documentation links and original caption hashes pass. Plot inspection copies stay in ignored runs/attention-visuals/, and the authored notebook embeds the saved figures. No hosted widget interaction, training, commits or new child work-log files. The foundations learner checkpoint and next data-preparation work are unchanged.

## 2026-10-10 — video 1 attention implemented

Implemented the first video in the existing Attention Is All You Need repo. The final Attention class retains upstream constructor/forward arguments and q_proj/k_proj/v_proj/out_proj names. Earlier stages include single-head, loop/vectorized heads, padding/causal masks and cross-attention. Explicit calculations and efficient SDPA share semantics. The Python demo prints intermediate weights and final tensors; the basic-concepts notebook imports the code. The notebook and later translation attention contract were inspected at 2f18a5470c4ef0a3d413dd7b18d07941f45779e9, with the MIT license and source/departure notes retained.

Verification: 24 CPU tests pass, covering independent PyTorch MultiheadAttention outputs/weights, manual/SDPA gradients, causal and padding isolation, masked rows, dropout and public interfaces. All 12 notebook code cells execute; both the direct Python demo and ./ai run attention-is-all-you-need succeed. PyTorch 2.14.1+cpu/NumPy 2.5.4, FP32, CPU, seed 7 gives maximum demo path difference 1.19e-7; the final width-9/3-head attention layer has 360 parameters. The locked environment check passes. Demo output is saved in the ignored project runs/attention-demo.txt. The refreshed source export contains 29 source files including the notebook, tests and license, with archive/member hashes verified. Captions, raw upstream material, environments, data and run output are excluded. All six caption hashes match; authored notebook is Git-eligible and all local raw sources/run output remain ignored. The initial commit is unchanged and all implementation changes are uncommitted.

Videos 2–3 source files and names are preserved. No commits, translation training or hosted/GPU/Drive recovery were performed; the foundations learner checkpoint is unchanged. The removed child agent instructions/generated logs remain absent. Next is video 2 tokenizer.py/prepare_data.py when requested.

## 2026-10-10 — video-project writing and cleanup

Rewrote the Attention Is All You Need README, sources, Colab guide, Python comments and launcher descriptions as personal project notes. Removed its AGENTS.md and three generated implementation-log files. Kept source metadata, captions, configs, dependencies and the launcher settings used by the TUI. Replaced links to the removed records in the shared docs. Implementation and learning state are unchanged.

Checks: the four remaining local doc links resolve; every external source URL and setup command is preserved. All fourteen Python files remain comment-only; launcher behavior and package dependencies are unchanged. The lock check passed with 87 packages. All six original caption hashes match; nine caption/transcript files remain present, ignored and untracked. The refreshed source ZIP has 26 files with matching member/archive hashes and excludes captions, agent instructions and removed logs. No commit, publication or model/training work was performed.

## 2026-10-10 — project-centric CLI/TUI

The TUI now groups both sites and the course reference under Docs, with Projects and Research below. Projects contains only the two existing implementation repositories: transformer-foundations-lab and attention-is-all-you-need. Selected entries supply their own purpose, focus, resources and file launchers through child-owned .ai-project.toml metadata. Site controls/ports appear for documentation services; Python entries show their own file choices and native environment/package actions. Foundations Run launches first_prediction.py; the video companion offers six comment-only entry points. Existing course references, histories and learner gates are preserved.

Local evidence and the final suite/layout results are recorded in [the CLI package](../../../cli/docs/implementation/2026-10/project-centric-tui-and-grouped-documentation/00-resume.md). Real foundations output remains 6/1/5; attention and inference placeholder launches exit 0 without model computation. No model code, training, hosted recovery, commit or publication is part of this tooling change.

## 2026-10-10 — separate P00 three-video boilerplate

Prepared the independent [attention-is-all-you-need](../../projects/transformer-labs/attention-is-all-you-need/README.md) companion from the three supplied Priyam Mazumdar videos. Downloaded English `en-orig` and `en` auto-captions for all three and recorded hashes/chapter metadata. The intended task is English-to-French encoder–decoder translation. Fourteen comment-only Python files reserve attention, tokenizer/data/collation, model, trainer, checkpoints, inference and BLEU destinations; there is no tutorial, model implementation, real test code or tracked notebook. Locked scaffold/dependency metadata and two proposed small configs are supplied with a source-ZIP/Colab launch route.

Local scaffold installation, build/import, empty-file scope and caption hashes were checked. Initial registration and source-export checks passed; their scope covered packaging and local empty-file launches only. The video dependency extra was resolved but not installed or tested in Colab. Companion model implementation, training, learner gates and persistent hosted recovery are not-started. The standard initial child snapshot and submodule registration are local only; no parent commit or remote publication is part of this setup.

The existing foundations child remains **P00-M1 in-progress, Lesson 2 setup complete**, with Lesson 3 part A next when resumed. P01 and Research are unchanged. For the companion, next is watching video 1 and writing attention when the user requests implementation.

## 2026-10-10 — P00 Lesson 2 setup

Created the independent `projects/transformer-labs/transformer-foundations-lab` child through the launcher and its child-owned implementation package. Installed the frozen Python 3.12.13/pytest 9.1.1 environment, verified the package import and saved `first_prediction.py`. Its prediction/target/error outputs are 6/1/5 for input 3 and 8/1/7 for input 4; the saved input remains 3. [Setup evidence](../../projects/transformer-labs/transformer-foundations-lab/docs/implementation/00_p00-foundations-implementation-and-learning/06-review.md).

Current state: **P00-M1 in-progress, Lesson 2 setup complete**. This is assistant-performed setup. Numerical/tensor/gradient/optimizer/MLP checks, learner explanation, model training and hosted recovery remain pending. Next: Lesson 3 part A, `linear()` and known-answer checks in the existing child. The launcher made its standard initial local scaffold snapshot and staged registration; no parent commit or remote publication occurred. Earlier entries below retain their historical states.

## 2026-10-10 — retired the superseded GPT-1 tutorial

Removed “GPT-1: build the mental model” and its active package/catalog entry at the user's request. P00 and the integrated P01 Small GPT course are now the two learning routes. Live guide references and CLI/site test fixtures use the current course; dated historical evidence and optional architecture study remain. A verified local recovery archive preserves the retired files. [Removal review](../implementation/2026-10/p01-small-gpt-practical-tutorial/07-retire-gpt1-tutorial.md). P00-M1 remains not-started.

## 2026-10-10 — P01 Small GPT integrated tutorial

Created [the required P01 tutorial](../tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md): twelve continuous lessons combining conceptual depth, maths, exact files/code/checks, text data, training/generation, full-state recovery, a thin Colab route and P02 readiness. Seven diagrams, four interactive explorations and 24 practice reveals use the existing reader. The 120832-parameter reference has checked CPU training and recovery evidence; [author verification](../implementation/2026-10/p01-small-gpt-practical-tutorial/06-review.md) separates it from learner and hosted outcomes. P00 and historical GPT-1 tutorial source remain unchanged. Current learning is still **P00-M1 not-started**.

## 2026-10-09 — reusable P00 tutorial standard

Updated the canonical workspace tutorial skill, related teaching/defense guidance, reusable templates and topic initializer from the accepted P00 course. Future courses combine conceptual depth, maths background and exact implementation in one route with project records and explicit exit evidence. The reader now uses each course's own first lesson and supports variable lesson counts. Existing tutorial source is preserved; learner state remains P00-M1 not-started. [Scope and verification](../implementation/2026-10/learn-inspired-adaptive-tutorial-improvements/07-p00-course-standard.md).


## 2026-10-09 — P00 maths and pre-P01 second pass

Expanded the same eleven-lesson route with prerequisite maths, hand calculations, executable checks and six additional practice reveals. Added a project readiness record mapping all six P00 milestones to actual evidence and distinguishing required Small GPT work from optional historical study. Clarified remaining character, model-boundary and persistence exercises. Verification is recorded in [the second-pass review](../implementation/2026-10/p00-practical-foundations-tutorial/08-second-pass.md).

Learning state remains **P00-M1 not-started**. No independent learner repository, successful task run or hosted recovery was created. Next learner action remains Lesson 1, then Lesson 2 implementation kickoff.


Updated: 2026-10-10
Authoritative learning-project status and evidence ledger. Plan: [roadmap.md](roadmap.md). Current checkpoint: [resume.md](resume.md).

Statuses: not-started, in-progress, blocked, complete, deferred. Complete requires the roadmap gates and evidence. Track model verification, learner understanding and publication independently. “Not started” does not imply a blocker.

## Current project

P00 Tokenization and Transformer Foundations / P00-M1 is in-progress with Lesson 2 setup complete. The independent foundations child/package and locked local Python environment exist; the saved first prediction and changed-input case are checked. Next: Lesson 3 part A, `linear()` and known-answer checks. Tensor/gradient/source/CPU/Free contracts, complete-model training, hosted notebook/recovery and learner explanation remain pending. P01 follows actual P00 gates. Tutorial author runs and retired setup-lab evidence remain separate preparation.

Compute requirement: P00–P04 develop/check locally and train complete miniature models on Colab Free under [COLAB.md](COLAB.md). Shared requirements are recorded; child notebooks, full persistent checkpoints and hosted recovery proof are not implemented/tested yet. Decide later project tiers from measured evidence, with Pro when requested.

| ID | Project | Status | Evidence / coverage |
| --- | --- | --- | --- |
| P00 | Tokenization and Transformer Foundations | in-progress | Lesson 2 independent child/environment/saved prediction checked; next Lesson 3 linear(); remaining M1 and model/hosted/explanation gates pending. |
| P01 | Small GPT Language Model | not-started | P00 prerequisite; pre-norm text LM required; exact GPT-1/classifier optional; Free run/recovery pending. |
| P02 | GPT-2 | not-started | Colab Free profile/notebook and recovery required; unverified. |
| P03 | Llama 2 → 3 | not-started | Colab Free profile/notebook and recovery required per variant; unverified. |
| P04 | Qwen3 Dense | not-started | Colab Free profile/notebook and recovery required; unverified. |
| P05 | Switch-style routing → Qwen3 MoE | not-started | — |
| P06 | Qwen3-Next | not-started | — |
| P07 | PaliGemma | not-started | — |
| P08 | Qwen3.5 | not-started | — |
| P09 | Qwen3.8 dense VLM | not-started | — |
| P10 | Gemma 4 E2B text | not-started | — |
| P11 | DeepSeek V3 → V3.2 | not-started | — |
| P12 | DeepSeek V4-Flash-style | not-started | — |
| P13 | Qwen3.8-Flash-Next-style | not-started | — |
| P14 | Kimi Linear → K3-style | not-started | — |
| P15 | DeepSeek V4.1-Flash-style | not-started | — |
| P16 | Mamba 2 → 3 | not-started | — |
| P17 | RWKV-7 | not-started | — |
| P18 | LLaDA → iLLaDA | not-started | — |
| P19 | Gemma 4 unified multimodal | not-started | — |

## P00 milestones

| ID | Status | Implementation / verification | Training and learner explanation |
| --- | --- | --- | --- |
| P00-M1 | in-progress | Lesson 2 child/package, locked Python environment, import and first prediction checked; remaining numerical/tensor/gradient/source/CPU/Free contracts pending | Assistant-performed setup; hand checks, tiny MLP fit and learner explanation pending |
| P00-M2 | not-started | Character/byte/BPE, artifact determinism/round-trip and leakage checks pending | Merge trace and token/byte interpretation pending |
| P00-M3 | not-started | Embeddings/positions/attention/masks/norm/FFN numerical checks pending | Attention/gradient/mask explanation pending |
| P00-M4 | not-started | Complete post-norm encoder–decoder, loss and conditional generation pending | Source/cross-attention/teacher-forcing explanation pending |
| P00-M5 | not-started | Common trainer/notebook/full persistent resume and fresh Free recovery pending | Copy fitting/held-out generation and controlled reversal pending |
| P00-M6 | not-started | Parameter/memory/profile/departure and P01 handoff records pending | Diagnosis, mechanism explanation and transfer assessment pending |

## P01 milestones

P01 begins after the P00 gates. M1–M3 and M5 are required; M4/M6 are optional extensions, and leaving them unstarted does not imply an incomplete required text-model scope.

| ID | Status | Implementation / verification | Training and learner explanation |
| --- | --- | --- | --- |
| P01-M1 | not-started | P00 handoff, source/config/frozen tokenizer/text-data and Free contracts pending | Decoder-only architecture and shifted-target explanation pending |
| P01-M2 | not-started | Complete pre-norm decoder/reference/causal/gradient checks pending | Differences from P00's post-norm/sinusoidal/ReLU model pending |
| P01-M3 | not-started | Text LM train/generate/common notebook/full resume/hosted recovery pending | Loss/gradient inspection, held-out LM evaluation and sampling pending |
| P01-M4 | not-started (optional) | Historical GPT-1 post-norm/pooling/classifier pending | Optional downstream objective/explanation pending |
| P01-M5 | not-started | Parameter/memory/profile evidence pending | Diagnosis/lifetime/access map and limitations pending |
| P01-M6 | not-started (optional) | Early cache parity/comparison pending; P02 requires cache coverage | Optional cache tradeoff explanation pending |

## Tutorial status

[P00](../tutorials/projects/p00/transformer-foundations/v1/lessons/01-prediction.md) remains the first learning route: eleven integrated lessons and its project/readiness records. Its prior author verification and failed copy-task demonstration remain preserved in the P00 package.

[P01 Small GPT](../tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md) now prepares the required pre-norm language-model route in twelve lessons. The superseded GPT-1 tutorial has been removed from the catalog and active sources. Availability, author checks and learner/hosted completion are distinct; P00-M1 setup is in-progress, with no milestone completion established. The versioned catalog is [tutorials/index.json](../tutorials/index.json).

## Session log

| Date | Project | Work and evidence | Next action |
| --- | --- | --- | --- |
| 2026-10-10 | P00 Lesson 2 | Independent child/submodule/package; frozen Python environment/import and arithmetic 6/1/5 → 8/1/7 checked. Assistant setup complete; M1 in-progress. | Lesson 3 part A: linear() and known-answer checks |
| 2026-10-09 | P00 simpler teaching | Build pages now explain small examples before code; all 12 concept chapters have gentler introductions. One-function numerical checks, Python explanations and dedicated build navigation preserve the complete technical route. Author checks/review remain in the P00 package. | Build 1, then Build 2 Learn A and linear() only; learner milestone unchanged |
| 2026-10-09 | P00 implementation guide | Ten ordered build pages with exact folders/files, integrated Python package code, independent tests and notebook generation. Actual scaffold/CPU install, 23 checks, 20→40 resume and local ZIP/notebook verification pass; all 46 reader routes checked at four widths. [Author review](../implementation/2026-10/p00-practical-foundations-tutorial/06-review.md). | Build 1 setup, then Build 2 numerics/tests/demo; P00-M1 remains not-started |
| 2026-10-09 | P00 reading/rendering review | All 12 P00 chapters read and visually checked; clearer introductions and complete attention/position/mask/PAD-loss examples. Shared reader gets larger text, wrapping tables, section navigation and less repeated material. All 40 records checked at four widths; [author review](../implementation/2026-10/p00-practical-foundations-tutorial/06-review.md) retains final verification. | Begin chapter 1; P00-M1 remains not-started |
| 2026-10-09 | P00 learner-readiness | Clear first session, C++/Python bridge, deeper worked math/data/model traces, 25 practice checks and a runnable standard-library walkthrough; Colab recipe ordering repaired. New arithmetic/snippets and 20→40-update smoke/resume checked. See the existing P00 author review for reader verification. | Start chapter 1 and its first-session exercise; P00-M1 remains not-started |
| 2026-10-09 | Docs organization | Root reduced from 25 to 8 entries; three setup/kickoff/map docs combined; website code/output grouped; selected evidence relocated. 62 tutorial files unchanged; 168 original tracked destinations preserved. [Review](../implementation/2026-10/docs-organization/00-resume.md) records final checks. | Begin P00-M1; learning state unchanged |
| 2026-10-09 | Docs cleanup | Removed 69 temporary generated files and two empty obsolete folders; reclaimed about 30 MiB. Six selected reader screenshots retained; tutorial/site sources, local edits and history preserved. | Begin P00-M1; learning state unchanged |
| 2026-10-09 | Tutorial reader | Dashboard manifest/chapter approach adapted to the existing static site: 15 chapter pages, seven supporting Markdown documents, two overviews and complete lesson views. Contents/previous-next, root/prefix routing, browser history/reload, mobile menu, existing teaching controls and print pass. [Review](../implementation/2026-10/multi-page-tutorial-reader/06-review.md). | Begin P00-M1; model/hosted/mastery gates remain not-started |
| 2026-10-09 | P00 tutorial | Detailed practical reference, workbook, numerical/model/CPU recovery verification and reader integration; preserved historical P01 lesson. Author task failure retained honestly. | Begin independent P00-M1; learner implementation/explanation and Colab recovery pending |
| 2026-10-09 | Planning | Architecture roadmap reviewed; hands-on learning and Projects/Research separation recorded in decisions.md | Start P01-M1 |
| 2026-10-09 | Workspace | Canonical plan/tracker/checkpoint created; CLI moved to workspace cli/ with history and local edits preserved | Integration verified; begin P01-M1 |
| 2026-10-09 | Readiness | CLI 48 tests; all 4 existing Projects test/build entries and 3 demos passed; documentation content/browser and isolated GPT kickoff passed. START_HERE.md added; scaffolds now follow their area owner and ignore checkpoints. | Begin P01-M1; no model milestone completed |
| 2026-10-09 | Documentation | All shared files consolidated under docs/; existing documentation history and 139 source hashes preserved. Tutorial/whiteboard skills adapted; P01 orientation and two language examples added. CLI 54 and Docs 7 tests/build/browser passed. | Start P01-M1; tutorial availability is separate from mastery |
| 2026-10-09 | Root cleanup | Retired both identical old roadmap copies after preserving 12 sections, 19 model plans and 60 resource URLs in roadmap.md, SETUP.md and this checklist. Removed the Projects-root ai wrapper; root visible files are README.md and AGENTS.md. | CLI 54, Docs 7, build/browser 13 records and shared launcher pass; start P01-M1 |
| 2026-10-09 | Skill readiness | All 7 entries/references/links validate; tutorial and planning/record workflows exercised in isolation. Fixed coverage/render/snapshot/proof handling and Project/Research routing. CLI 64 and Docs 7 tests/build/browser pass. | Start P01-M1 with workspace-local skills |
| 2026-10-09 | Agent context | Existing workspace/Projects/Docs README and AGENTS now expose P01-M1, completed preparation and the next action; session instructions require reading/reconciling canonical progress and updating snapshots on milestone changes. | Continue P01-M1; no model gate advanced |
| 2026-10-09 | Tutorial improvement | Adapted teaching ideas from pinned amosblomqvist/learn into bounded prerequisite checks, concept dependencies, motivated steps and attempt/feedback/transfer guidance. P01 pilot and composing skill validators/examples pass; site/instruction checks are recorded in the [owning package](../implementation/2026-10/learn-inspired-adaptive-tutorial-improvements/06-review.md). | Begin P01-M1; no human mastery or model gate advanced |
| 2026-10-09 | Colab requirements | [Free training/recovery contract](COLAB.md), roadmap/kickoff/setup/P01 plan and six snapshots updated. [Verification](../implementation/2026-10/colab-free-training-and-recovery-requirements/06-review.md): 7 content tests, build/browser across 14 records, desktop/mobile guide inspection passed. | gpt-lab M1 plus Free setup/storage/budgets; notebook, training and hosted recovery pending M3 |
| 2026-10-09 | Commit baseline | Reviewed the nine independent repositories; local commit preparation covers six dirty owners, preserves history and updates umbrella pointers after child commits. CLI 64, Projects Docs 7, Research Docs 5 and Generalization 4 tests pass; both browser suites, skills/examples and TUI/doctor pass. [Review](../implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md). | Begin P01-M1 after the final clean-tree/pointer audit; no model gate advanced |
| 2026-10-09 | P00 roadmap revision | Source-backed P00 tokenizer/attention/full encoder–decoder plan added; P01 clarified as small GPT, historical scope optional. Six snapshots, kickoff/state/compute/tutorial membership/source catalog/site/architecture updated. [Review](../implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/06-review.md): Docs 7 tests/build/browser across 15 records/13 sources, CLI 64 tests/TUI, preservation/link checks and desktop/mobile inspection pass. | Begin P00-M1; no foundation/model/training/hosted/mastery gate advanced |
| 2026-10-09 | Commit baseline completion | Included the completed P00 roadmap revision under the user’s explicit instruction. Preparation is saved in focused local commits, with child revisions recorded before umbrella pointers; [commit review](../implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md) and Git history retain scope/checks. | Begin P00-M1 after the final clean-tree/pointer audit; no model gate advanced |

Workspace integration evidence: [CLI relocation tracker](../../../cli/docs/implementation/2026-10/workspace-cli-relocation/05-tracker.md). CLI 46 tests, Projects content/build/browser checks, canonical links/IDs and original catalogue/history preservation passed. This completes infrastructure work, not P01.

Append a session row after meaningful work. Update status/coverage only with actual evidence; link the child-owned package, commands, source revisions, configuration, run results and learner explanation. Keep bulky/generated runs ignored. No roadmap models are externally published.


## Detailed completion checklist

Apply this inherited checklist alongside the [current roadmap gates](roadmap.md). These unchecked items are requirements, not evidence of completed work.


Advance when you can:

- [ ] Draw the full model and annotate important tensor/state shapes.
- [ ] Explain each model-specific equation and implement it in a small reference case.
- [ ] Run a complete forward pass, loss, backward pass, and generation path.
- [ ] Demonstrate causal or conditional-generation correctness.
- [ ] Demonstrate equivalent state/cache execution where expected.
- [ ] Save/reload the model and count its exact parameters.
- [ ] For P00–P04, run the thin common-trainer notebook on Colab Free; verify full-state persistent saves, interrupted-write fallback and recovery after runtime deletion. Record actual compute/budget evidence.
- [ ] Compare at least one optimized path with the reference numerically.
- [ ] Produce a reproducible local profile and interpret its limits.
- [ ] Record the implementation's coverage and simplifications.
- [ ] Propose the next experiment from observed evidence.

Reading a tutorial or obtaining plausible-looking output is not the completion criterion. A working, verified implementation and an evidence-backed explanation are the required deliverables.

## 2026-10-09 — P00 launcher and setup-lab retirement

P00 is the default tutorial entry: Run for first-session arithmetic, Test for numerical checks, Open for chapter 1. Four clean setup repositories and their active catalog cards were retired after checkout-hash and Git-bundle verification. Research synthetic measurements remain archived evidence. The CLI-owned package is `cli/docs/implementation/2026-10/p00-launcher-and-retire-setup-labs/` at workspace home. P00-M1, independent implementation and hosted gates remain pending.

Verification: 72 CLI tests, P00 run/test from /tmp, TUI smoke/80x24/120x36, Projects 8 tests plus root/prefix browser checks (36 records), Research 5 tests plus browser checks (4 records). Four archive recovery clones matched source HEADs/bytes and passed Git fsck. No learning or hosted-training gate advances.

## Integrated P00 lessons and project area — 2026-10-09

P00 now has eleven continuous lessons combining the former chapter explanations, derivations, diagrams, practice and implementation. Lesson 1 starts with one prediction before setup. Project navigation groups implementation, sources, reference runs and practice rather than a supporting-material shelf. All 95 former chapter sections remain represented; code and historical evidence are preserved. Old reader URLs redirect to the combined lessons.

Verification: 23 worked-code tests, eight site tests, 73 CLI tests, real P00 Test and TUI smoke. All 36 routes pass four-width browser checks at root and /project_docs/; selected live desktop/mobile views and canonical lesson links pass. See [integrated course review](../implementation/2026-10/p00-practical-foundations-tutorial/06-review.md). P00-M1 remains not-started; next learning action is [Lesson 1](../tutorials/projects/p00/transformer-foundations/v1/lessons/01-prediction.md).
