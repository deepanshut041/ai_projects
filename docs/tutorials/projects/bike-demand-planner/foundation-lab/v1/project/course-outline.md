# Foundation Lab: Bike Demand Planner — reading route

This course consolidates the completed project into one route. The primary pages are the notes to understanding: watch each short video, follow the adjacent calculation and source, then inspect the actual result. The existing tutorial/PDF background is assumed; routine implementation is already available. There is no requirement to type the project from scratch.

1. [One hourly prediction and one numerical update](lessons/01-prediction-contract.md) — Which information can a forecast use, and how does an error move the weights?
2. [Understand the hourly data, calendar features and time split](lessons/02-hourly-data-and-splits.md) — How do real records become a leakage-safe matrix?
3. [Build and compare simple calendar predictions](lessons/03-calendar-baselines.md) — What can a small prediction rule achieve?
4. [Fit linear regression and follow its gradients](lessons/04-linear-regression.md) — Can a weighted calendar sum explain the same hours?
5. [Compare batches, momentum and Adam](lessons/05-batches-and-optimizers.md) — What changes when each update sees fewer rows?
6. [Change the loss, regularization and calendar structure](lessons/06-losses-and-calendar-structure.md) — Which errors and calendar patterns does the model represent?
7. [Follow tree leaves, forests and boosted corrections](lessons/07-trees-and-ensembles.md) — Can tree partitions and ensembles improve later estimates?
8. [Read hourly errors and freeze a supported candidate](lessons/08-residuals-and-candidate-choice.md) — Where do the estimates fail and what requests are supported?
9. [Reload predictions and recover the next training update](lessons/09-saved-predictions-and-recovery.md) — What must survive a process restart?
10. [Evaluate later hours and inspect the completed application](lessons/10-final-evaluation-and-application.md) — What do untouched later outcomes say about the selected function?

Start with one prediction and its two changed targets. Stop when you can account for the direction and size of that update. Continue at your own pace; the entire course need not be read in one sitting. The separate learning points and comparisons are review aids beside each lesson, not a second required route.
