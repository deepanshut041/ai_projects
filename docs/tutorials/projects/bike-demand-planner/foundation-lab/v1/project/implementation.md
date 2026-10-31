# Project notes and completion record

The complete local implementation is signed as `0b49c0aafe154888929d8b22dbe6637ba2e56661`, with commit subject `foundations(bike-demand-planner): finish evaluation and local application`. All ten planned local increments are implemented. This course follows their teaching order while keeping the original experiments and results.

## Return to the right calculation

- [One hourly prediction and one numerical update](lessons/01-prediction-contract.md): 40 -> 40.30 for target 45; 40 -> 39.70 for target 35.
- [Understand the hourly data, calendar features and time split](lessons/02-hourly-data-and-splits.md): 17379 rows, 45 columns, 8645/4357/4375 memberships and two embargoes.
- [Build and compare simple calendar predictions](lessons/03-calendar-baselines.md): Validation MAE 154.2452 -> 91.5729 on identical hours.
- [Fit linear regression and follow its gradients](lessons/04-linear-regression.md): 200 updates; validation MAE 116.7312; 724 negative predictions.
- [Compare batches, momentum and Adam](lessons/05-batches-and-optimizers.md): Ten passes per trial; Adam batch-32 mean MAE 114.3608.
- [Change the loss, regularization and calendar structure](lessons/06-losses-and-calendar-structure.md): Interactions raw/clipped MAE 108.6937/100.5164; groups retained.
- [Follow tree leaves, forests and boosted corrections](lessons/07-trees-and-ensembles.md): 27 main fits; group validation MAE 91.5729 still wins.
- [Read hourly errors and freeze a supported candidate](lessons/08-residuals-and-candidate-choice.md): June group MAE 142.9952; rush MAE 178.1954 across 750 hours.
- [Reload predictions and recover the next training update](lessons/09-saved-predictions-and-recovery.md): 4357 predictions exactly replay; recovered numerical training state matches.
- [Evaluate later hours and inspect the completed application](lessons/10-final-evaluation-and-application.md): Final groups MAE 125.4167 vs forest 103.6474 over 4375 hours.

## What was checked in the completed project

The source has 128 passing checks. A separate clean source reconstruction ran the documented preparation commands and reproduced all prepared bytes, all seven representative model files and all final metric sets. Original and rebuilt assets ran through an installed wheel with source-project access denied. Actual desktop/mobile checks covered historical replay, separate outcome reveal, missing observations, invalid requests and the final assessment. These are the earlier completion results; converting the tutorial does not rerun or extend the experiment.

## Keep the limits with the result

Groups/raw remains the validation-selected model. Its later MAE is 125.4167 over 4375 observed hours, while the retained forest has lower later MAE 103.6474. That reversal is disclosed rather than used to rewrite the earlier choice. The large underprediction at rush hours remains a practical limitation. Local software completion supplies neither an operational quality threshold nor learner mastery or hosted recovery evidence. No next project is started here.

## Personal notes

Use the learning-points page beside a lesson for your own wording and questions. No explanation answer or completed practice is recorded on your behalf. The original private work records remain a historical archive so the media’s source hashes and old links stay valid; this tutorial is the consolidated reading route.
