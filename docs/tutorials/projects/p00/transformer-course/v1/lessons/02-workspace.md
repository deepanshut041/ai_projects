## Lesson 2 — Reproducible source and implementation workspace

**Goal:** run an inspectable reference and keep source, environment, artifacts and learner work distinct. Prerequisite: task contracts. Python syntax and virtual-environment basics are assumed; these instructions name the actual runtime rather than recreate a general programming course.

### The reference is one self-contained source package

From workspace home, the course directory is `ai_projects/docs/tutorials/projects/p00/transformer-course/v1/`, with model/helper files under `examples/`. Its code has local imports between those files; copying only a model class loses data, masks, training and recovery contracts.

| Source group | Files and responsibility |
|---|---|
| Translation data/model | `translation_data.py`, `translation_attention.py`, `translation_model.py` |
| Translation learning/quality | `translation_train.py`, `translation_metrics.py`, `translation_checks.py` |
| RoBERTa and QA | `roberta_data.py`, `roberta_model.py`, `roberta_train.py`, `roberta_checks.py` |
| Complete sessions/recovery | `course_checkpoint.py`, `course_session.py`, their checks and `course_recovery_checks.py` |
| Arithmetic/source delivery | `foundation_math.py/.cpp`, independent tests and `course_source_export.py` |

Model Python remains the implementation asset. A notebook calls it; notebook cell ordering must not supply hidden definitions needed by a standalone model. Plotting, study questions and inspection should not become imports that a trainer unexpectedly requires.

### Select the checked reference interpreter

The existing attention child's frozen environment supplies Python 3.12.13, Torch 2.14.1+cpu and NumPy 2.5.4 for these local references. From workspace home:

```bash
source ai_projects/projects/transformer-labs/attention-is-all-you-need/.venv/bin/activate
cd ai_projects/docs/tutorials/projects/p00/transformer-course/v1/examples
python -c 'import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)'
python translation_model.py
```

The model command reports logits `(6,5,12)`, 30,796 parameters, a finite loss and a head gradient. Here and in later shell blocks, `python` means this selected interpreter, and the working directory is `examples/` unless stated otherwise. Do not change the existing learner repository or install a second environment there to make an author reference command work.

When recreating an environment elsewhere, pin and verify its actual versions. [CPU source/Colab export](examples/course_source_export.py) records the exporting runtime. An installed package and an already-imported package can differ until a kernel restart; inspect the loaded version. Keep environments, downloaded datasets, model weights and generated runs outside Git.

### Preserve existing implementation work

The current learner child `ai_projects/projects/transformer-labs/transformer-foundations-lab` already contains setup and a saved first prediction. Inspect its source/checkpoint and the owning Projects Git state before any learner edit; its next exercise remains `linear()`. Course authoring does not create a new model child, reset it or replace that work.

For a reference reconstruction, use a fresh disposable directory and copy the complete source export into it. For actual learner implementation, use the child and its checkpoint when those tasks begin. The topic lessons identify the reference file/class/method and a specific exercise or check; copying all completed sources is not the same evidence as implementing and explaining them.

### Configurations and artifacts define compatibility

A run's identity includes source bytes, architectural configuration, tokenizer vocabulary/rules/special IDs, preprocessing/split identity, optimizer/schedule settings and precision/runtime assumptions. A count such as V=12 does not identify which token each row means. Matching state-dict shapes does not prove semantic compatibility.

A complete checkpoint also contains evolving state: optimizer moments, update/schedule position, shuffle/data cursor and RNG/corruption generators. Static configuration and evolving state are different. The source exporter pins the source ZIP; recovery later verifies both identity and state before accepting a generation.

A backend/library change can alter numerical behavior. Use FP64 for small derivative comparisons and recorded tolerances; use the same FP32 CPU source for controlled training/recovery. Neither a CPU success nor a configured `device` string demonstrates GPU throughput or deleted-hosted-runtime survival.

### Run a targeted check before a large experiment

```bash
python -m unittest discover -s . -p test_foundation_math.py
python translation_checks.py
```

The second command includes a bounded six-pair learning test. Read the [reference record](project/reference-runs.md) for its exact domain before interpreting its scores. These are local author checks; later chapters explain each path and offer independent probes. The [implementation record](project/implementation.md) is where I keep my actual files/checks/next action.

[[practice:workspace-boundaries]]
[[practice:workspace-transfer]]

### Notes and failure cases

In my notes: draw the model/helper/trainer/artifact boundaries and record the interpreter/source identity. Explain why changing tokenizer row meanings with identical tensor shapes breaks compatibility. Diagnose a local import failure, wrong loaded Torch version and a weights-only restart as separate problems. Continue with [tensor interfaces and budgets](lessons/03-tensors.md).
