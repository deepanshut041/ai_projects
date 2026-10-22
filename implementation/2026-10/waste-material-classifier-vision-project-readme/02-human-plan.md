# Plan — Waste Material Classifier vision README

Date: 2026-10-15 (America/Toronto)
Lead: ai-project-workflow through workspace planning; supporting workspace tutorial/whiteboard guidance for reference-design preparation.

## Authorized outcome

Create projects/foundations/waste-material-classifier/README.md and update the main foundations README. Keep the two existing classical ML project READMEs intact. Reconcile canonical preparation/order records and six compact entry-point snapshots as required. No model scaffolds, data/weight downloads, dependency installations, training, applications, CLI/TUI changes, commits or publication.

## Source/design findings

- UCI RealWaste ID 908: real landfill imagery, nine material classes, public CC BY 4.0 dataset. Inspect official archive/hash/class mapping at kickoff; this task reads metadata only. Image/group/scene audit is still pending.
- PyTorch Conv2d and CrossEntropyLoss references inspected through their stable redirects (2.14 docs visible on 2026-10-15); transfer-learning tutorial and original residual-learning paper read as design references. Installed versions are not selected by documentation lookup.
- One supervised image-classification application, moving from an MLP to a small CNN, controlled architectural/training improvements, a residual comparison, then frozen/fine-tuned pretrained CNN comparison. RNN/LSTM/text/embedding and detection/segmentation remain outside this deliverable.
- Record scratch versus pretrained conditions, pretrained transform/weight identity, BatchNorm state, grouped split/augmentation boundaries, fair budgets and final holdout. No guaranteed task accuracy or hardware fit.

## Tasks

| ID | Deliverable | Dependencies | Acceptance |
| --- | --- | --- | --- |
| V01 | Reconcile instructions, existing state and primary sources | None | Scope and real-world data contract recorded |
| V02 | Write deep project README and update overview | V01 | Phased NN/CNN route; numerical/shape examples; experiments, application, compute/recovery and readiness |
| V03 | Update preparation/order records | V02 | Classical ML stage -> selected vision DL stage -> later sequence/embedding work; earlier evidence preserved |
| V04 | Verify and deliver | V02, V03 | Local links, hand arithmetic/counts/shapes, fences, whitespace and scoped diff checks |

## Teaching route and verification boundary

| Prior result | Next problem | Mechanism | Planned check |
| --- | --- | --- | --- |
| Linear/logistic model from the ML stage | Learn nonlinear visual features | MLP + activations and CE | Known gradients and overfit one batch |
| Flattened-image classifier | Share local pattern detectors | Cross-correlation, channels, pooling | Independent patch calculation and shape/parameter counts |
| Small CNN | Diagnose capacity and optimization | Initialization, normalization, regularization, residual paths | Controlled studies plus activation/gradient inspection |
| Understood scratch model | Compare learned prior features | Frozen versus fine-tuned pretrained CNN | Distinct training/weight/transform identity and fair budgets |
| Frozen candidate | Use and resume reliably | Inference, review policy, complete persistent state | Fresh-process parity and fresh-runtime hosted recovery when implemented |

Current validation covers Markdown and authored calculations only. This README is a reference-design project plan, not an executable course, learner mastery record or experimental result.
