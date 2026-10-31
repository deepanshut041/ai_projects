## Lesson 27 — backbone transfer and stage identity

**Goal:** move compatible pretrained contextual weights into a new QA model, verify the copy independently, and distinguish task transfer from exact training resume. Prerequisites: the complete MLM model, vocabulary identity, training state and the recovery lesson. Pretraining supplies an initialization; it does not supply a ready-trained span classifier or its optimizer history.

### Separate reusable representation from task-specific heads

The MLM model contains `roberta` (embeddings and contextual encoder layers) and `mlm_head` (dense–GELU–LayerNorm–vocabulary scoring). The QA model contains the same backbone architecture plus `qa_head: Linear(D,2)`. Transfer copies only the backbone. The MLM head's transform and vocabulary bias are not QA endpoint parameters; discard them from the downstream model rather than inventing a conversion.

For hidden state $h_t\in\mathbb R^D$, MLM produces $V$ token scores, while QA produces two scores $[s_t,e_t]$. With $D=32$, the new QA head has $2D+2=66$ scalars. Its output-column meaning is start/end endpoint, not two vocabulary entries. The default backbone contains 38,496 parameters; the QA model therefore contains 38,562. Those counts describe this own-format reference, not an official RoBERTa checkpoint.

A pretrained encoder's useful contextual features are an empirical transfer premise. Backbone equality after copying proves the operation, not improvement on a held-out QA task. Test a new random-backbone baseline under the same QA data/budget if you want to measure the benefit of pretraining.

### Specify what the transfer artifact contains

`export_backbone` in [roberta_train.py](examples/roberta_train.py) requires an MLM bundle and writes:

| Artifact field | Meaning | Compatibility role |
| --- | --- | --- |
| `format` | `course-roberta-backbone-v1` | Names this schema rather than an official library format |
| `config` | Resolved full `RobertaConfig` | Width/depth/heads/positions/dropout/PAD and related architecture |
| `tokenizer` | Frozen ordered own-format byte merges | Defines every vocabulary row's meaning |
| `backbone` | `model.roberta.state_dict()` | Registered embedding/normalization/encoder tensors |

It deliberately has no optimizer, scheduler, mask/batch RNG, sampler/progress counter or MLM head. It is a weight-transfer artifact, not a full recovery generation. Do not label a successful `torch.load` as a demonstration of the next optimizer update.

`load_backbone_for_qa` checks the format, exact configuration and tokenizer fingerprint, then uses strict backbone loading. Matching vocabulary **size** is insufficient. If IDs 6 and 7 exchange meanings, equal-shaped tables still send the wrong vectors into every downstream operation. Position rules and PAD identity also address different learned rows. Official RoBERTa preprocessing, token-type handling, parameter names and weights have another mapping; this function provides no automatic conversion path. [Official contract](source:hf-roberta), [source map](project/source-map.md).

The in-memory `model.load_mlm_backbone(pretrained)` convenience checks configuration and copies a model's backbone, but receives no tokenizer artifact. It cannot independently validate tokenizer semantics. The default demonstration constructs both models from the same deterministic fixture tokenizer; general artifact transfer should use the checked loader or a separate explicit tokenizer check.

[[diagram:backbone-transfer]]

### Start a new objective with new optimization state

Let pretrained weights be $\theta_{\mathrm{MLM}}$ and new head parameters $\psi_0$. Fine-tuning initializes $\theta_0^{\mathrm{QA}}=\theta_{\mathrm{MLM}}$, samples/initializes $\psi_0$, then optimizes the QA objective. The reference constructs a new AdamW instance, a QA schedule starting at index zero, new named generators and QA progress `step=0`.

By contrast, exact QA resume restores $(\theta_t,\psi_t)$ together with Adam moments/step, scheduler position, generators, source/data/config identity and counters. Copying only $\theta_t$ and resetting the optimizer defines a changed experiment. Keeping MLM optimizer moments under the new task also differs from the declared fresh QA protocol; shared parameter shapes cannot supply a justification for that choice.

A simple adaptive-update example shows the distinction: Adam with a nonzero first moment can move a weight even when its current gradient is zero, while a new optimizer with zero history makes no data-gradient move. Resetting history therefore changes the next update without changing initial parameter values. Tutorial 0's optimizer/recovery derivations develop that calculation.

`course_session.py` records the SHA-256 of an initialization backbone in the new run contract. It rejects `--backbone` during `resume`, because a resumed QA run restores its own state rather than taking new initialization. Keep a separate run ID for each stage and retain the source/data/tokenizer identities that made the artifact compatible.

### Prove equality, fresh state and head ownership

From `examples/` using the frozen interpreter:

```python
from pathlib import Path
import tempfile
import torch
from roberta_data import ByteBPETokenizer
from roberta_train import build_training_bundle, training_step, export_backbone, load_backbone_for_qa

torch.set_num_threads(1)
mlm = build_training_bundle('mlm', total_steps=10)
training_step(mlm)
qa = build_training_bundle('qa', total_steps=10)
head_before = {n:t.detach().clone() for n,t in qa.model.qa_head.state_dict().items()}
with tempfile.TemporaryDirectory(prefix='course-transfer-') as folder:
    artifact = Path(folder) / 'backbone.pt'
    export_backbone(mlm, artifact)
    load_backbone_for_qa(qa, artifact)
    for name, tensor in mlm.model.roberta.state_dict().items():
        torch.testing.assert_close(qa.model.roberta.state_dict()[name], tensor, atol=0, rtol=0)
    for name, tensor in head_before.items():
        torch.testing.assert_close(qa.model.qa_head.state_dict()[name], tensor, atol=0, rtol=0)
    assert mlm.step == 1 and len(mlm.optimizer.state) > 0
    assert qa.step == 0 and len(qa.optimizer.state) == 0 and qa.scheduler.last_epoch == 0
    qa.tokenizer = ByteBPETokenizer()
    try:
        load_backbone_for_qa(qa, artifact)
    except ValueError as error:
        assert 'Tokenizer identity differs' in str(error)
    else:
        raise AssertionError('Mismatched tokenizer must be rejected')
print('exact backbone copy; fresh QA optimization state; incompatible tokenizer rejected')
```

The checks compare every registered backbone tensor, retain the newly initialized head and inspect the actual optimizer history. All files are temporary diagnostics; no learner run is overwritten. Copying values into existing registered parameters preserves the parameter objects to which the new optimizer refers.

### Use separate stage commands

After a completed/persisted MLM session exports `backbone.pt`, start QA in a fresh run directory:

```bash
python course_session.py --task qa --mode new --run-directory /tmp/course-qa-study \
  --total-steps 120 --additional-updates 25 --backbone /tmp/course-mlm-study/backbone.pt
```

That example requires the MLM artifact from Lesson 26. On a later QA session, choose `--mode resume` with the same QA run directory and total goal, and omit `--backbone`. A path existing on disk does not establish that its contents match the required artifact; loading must perform the compatibility checks.

### Changed-condition practice and notes

[[practice:backbone-transfer-transfer]]

[[practice:backbone-transfer-resume-case]]

In [my study notes](project/study-notes.md), draw the tensors copied and the state reset, derive the 66-parameter QA head, and explain why equal vocabulary size is insufficient. Write the separate acceptance evidence for transfer equality, downstream improvement and exact QA continuation. Next, [offsets and windows](lessons/21-qa-offsets-and-windows.md) construct valid supervised spans for that new task.
