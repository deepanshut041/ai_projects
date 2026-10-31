## Lesson 21 — Full-state continuation, verified generations and Colab

**Goal:** preserve the exact state needed for the next translation update, prove local new-process continuation and perform an honest persistent-storage drill. Prerequisites: translation optimizer/schedule, evaluation and update boundaries. Work in a disposable `examples/` copy with Lesson 2's compatible runtime. [course_checkpoint.py](examples/course_checkpoint.py) owns snapshots/recovery; [course_session.py](examples/course_session.py) wraps the existing trainers with bounded sessions. Neither introduces a different model or optimizer.

### Reconstruct a state transition, not just its weights

Let the completed training state include the parameters, optimizer moments, schedule, data/random state, identity and update index:

$$
S_t=(\theta,m,v,\mathrm{schedule},\mathrm{data},\mathrm{RNG},\mathrm{identity},t).
$$

An update applies $S_{t+1}=U(S_t)$. Equal θ alone is insufficient because U also consumes optimizer history, the next rate and the next data/random choices.

For Adam with both betas .5 and gradients 2 then−1, first moments are 1 then 0; second moments are 2 then 1.5. The second corrected numerator is 0. Resetting Adam at the same θ makes the new gradient−1 a first step with a nonzero numerator. Weight-only reload is therefore a new optimization trajectory. A LambdaLR state also cannot recreate an arbitrary changed Python closure: restore its position under the same defining warmup/total schedule.

| Saved owner | Required state/identity |
|---|---|
| Model | Parameters, buffers, model class, mode and shapes |
| Optimizer | Moments, step counters, parameter mapping, group rates/decay |
| Scheduler/optional scaler | Position and state; defining function/precision contract |
| Data | Corpus/tokenizer meanings and consumed position or explicit sampling generator |
| Randomness | Python, NumPy, Torch CPU, applicable CUDA and named generator states |
| Progress | Completed updates, attempts, examples/input-token/supervised-target counts |
| Compatibility | Source/config/protocol, package/runtime/device/dtype/thread semantics |

This translation reference uses a fixed full batch, so it has no shuffled permutation/cursor to restore. Its corpus/tokenizer identity and global update count describe that order. A real shuffled loader must add its actual permutation/cursor or a verified reconstruction rule. The later MLM/QA trainers use named `mask`/`batch` generators; the shared session supplies them automatically. [Checkpoint guidance](source:torch-recovery), [runtime limits](source:torch-randomness).

### Save only at a declared completed-update boundary

The reference saves after optimizer and scheduler steps and the counter advance. Completed gradients can still be present; the next update clears them. They are not pending accumulation work. Saving halfway through a group would additionally require partial gradients, its total denominator, microbatch cursor and consumed RNG state. The low-level saver does not infer or enforce that boundary for you; the session runner supplies it by construction.

Construct fresh model/optimizer/scheduler objects before restoration. Model initialization consumes random draws. Load component state, then restore saved RNG immediately before the next random/data operation. Restoring RNG before building a random model disturbs the recovered stream.

### A generation becomes acceptable only after verification

`save_generation` creates a new numbered directory with a random suffix; it never overwrites a prior generation. It clones model/optimizer tensors to CPU, serializes `state.pt`, reads it back and checks deserialization, then writes byte count/hash/runtime identity into `manifest.json`. Finally it writes `complete.json`, containing the manifest hash, and runs `_verified` before rotating older copies.

[[diagram:recovery-generations]]

Recovery checks completion→manifest hash/schema→state bytes/size/hash, then payload identity and model key/shape compatibility. It scans newest first, reports incomplete/corrupt/incompatible entries and restores the newest compatible verified generation. Keep at least two verified generations of the same identity; foreign/incomplete entries are not promoted into good backups.

This is logical acceptance of complete generations. The checkpoint writer does not implement an fsync/atomic-rename durability transaction. The session's separate read/write preflight calls fsync on its small probe, which does not prove every checkpoint survives a storage/server failure. Mounted Drive durability needs actual runtime deletion and successful reread; hashes establish byte integrity, not independent origin or service availability.

Compatibility checks precede applying the declared model state. A later optimizer/RNG restoration failure is still an error: discard those newly built objects, rather than continuing a partly restored session. Resume never implicitly falls back to starting a new experiment. A deliberate warm start or MLM→QA transfer loads appropriate weights under a new stage identity and creates task-specific optimizer state.

### Compare the same next update and a corrupt-newest fallback

Run this complete probe from `examples/`; all generated files live in an automatically removed temporary directory:

```python
from pathlib import Path
from copy import deepcopy
import tempfile
import torch
from translation_train import make_state,update
from course_checkpoint import save_generation,load_latest
from course_recovery_checks import compare,random_draws

torch.set_num_threads(1)
def fields(state):
    return dict(model=state.model,optimizer=state.optimizer,scheduler=state.scheduler,identity=state.identity)
with tempfile.TemporaryDirectory(prefix='tutorial1-recovery-') as temporary:
    directory = Path(temporary)
    a = make_state(seed=7,steps=32)
    for _ in range(3): update(a)
    save_generation(directory,step=a.step,data_state={'completed_updates':a.step},**fields(a))
    draws = random_draws()
    expected = update(a)
    weights = deepcopy(a.model.state_dict())
    optimizer = deepcopy(a.optimizer.state_dict())
    b = make_state(seed=7,steps=32)
    restored = load_latest(directory,**fields(b)); b.step = restored['step']
    assert b.step==3 and random_draws()==draws
    compare(expected,update(b)); compare(weights,b.model.state_dict())
    compare(optimizer,b.optimizer.state_dict()); compare(a.scheduler.state_dict(),b.scheduler.state_dict())
    newest = save_generation(directory,step=b.step,**fields(b))
    with (newest/'state.pt').open('ab') as stream: stream.write(b'broken')
    c = make_state(seed=7,steps=32)
    fallback = load_latest(directory,**fields(c))
    assert fallback['step']==3 and fallback['skipped'][0]['reason']=='incomplete or corrupt'
    print('same next update:',b.step,'fallback:',fallback['step'])
```

Expected updates are 4 and 3. This proves a new-object replay, not yet a separate process. Run `python course_recovery_checks.py --project translation` for the stronger three-updates/save/new-interpreter/five-updates comparison. It checks exact parameters, optimizer/scheduler, update 8 and future random draws, plus incomplete/corrupt fallback, source/schedule/tokenizer rejection and two-good-copy retention. An unexplained CPU→GPU, precision or package change is rejected by strict runtime identity; it is not evidence of universal bitwise reproducibility.

### Slice one fixed experiment into bounded sessions

From `examples/`, choose a fresh run path:

```bash
python course_session.py --task translation --mode new --run-directory /tmp/tutorial1-translation-run --total-steps 32 --additional-updates 3 --save-updates 1
python course_session.py --task translation --mode resume --run-directory /tmp/tutorial1-translation-run --total-steps 32 --additional-updates 2 --save-updates 1
```

The first finishes at 3; the second restores 3 and finishes at 5. `new` refuses even an empty existing directory; resume requires `run.json` and compatible verified state. Total steps 32 is the unchanged experiment/schedule, while 3/2 are process allowances. A failed newer generation may restore an earlier durable update and replay lost work; printed progress is not durable progress.

Session defaults are 1000 additional updates or 3600 seconds, saves after 100 updates or 300 seconds and at normal end, `keep=2`, a 256 MiB run budget and 64 MiB free-space reserve. Estimate enough storage for retained copies **plus** the new write before rotation; actual saves are read back. An update or storage operation can cross a wall-time threshold. Persistence failure records the last good generation and stops before another update.

For translation, one batch consumes six examples, sixty nonpadding source/prefix input tokens and thirty labels. A three-update slice counts 18/180/90 presentations. MLM counts selected clean labels; QA counts supervised span examples, so those denominators are not interchangeable. `input_counts` copies sampling-generator state to inspect a future batch without consuming its RNG. `python course_session_checks.py` checks shared task/refusal/accounting/budget/failure boundaries when those later task implementations are understood.

### Run the same source after deleting the hosted runtime

`python course_source_export.py /tmp/tutorial1-course-source.zip` exports safe source/runtime identities and an output-cleared thin Colab notebook. Upload the matching notebook and ZIP; its upload checks pinned archive/member hashes, safe flat names and identical existing source. It installs/inspects the exporting CPU Torch/NumPy versions and calls the same checks/trainers/session code. Keep personal notes separate and choose fresh export destinations. [Colab resource contract](source:colab-faq).

The notebook mounts Drive, chooses task/run ID and starts a short persistent slice. Use translation first, CPU/FP32/one thread, and retain the same source/configuration/total schedule. Then record a verified generation, choose **Disconnect and delete runtime**, recreate source/dependencies, remount the same Drive run and explicitly select resume. Record restored/advanced updates, skipped generations, byte counts, actual environment and persistence latency. A local temporary-directory replay, cell restart or syntax-valid notebook does not perform this drill. Hosted recovery remains pending until those observations exist.

Later, MLM can export `backbone.pt` and a **new** QA run can accept that compatible initializer. QA resume restores its own full state and rejects another initializer; a random-initialized QA run is a debugging baseline. The next lessons define those stage contracts rather than treating every reload as continuation.

### Practice and notes

[[practice:checkpoints-and-colab-practice-1]]
[[practice:full-state-resume]]

In my notes, draw complete-update→write→verify→complete→rotate, calculate the rollback/presentation counters, and compare weight-only reload, exact resume and a new task's warm start. Record which local process/storage and hosted deletion checks I actually performed. Continue with [BERT-to-RoBERTa design](lessons/16-bert-to-roberta.md).
