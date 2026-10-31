## Lesson 24 — checkpointing and reproducible recovery

**Goal:** identify the state required for the next optimizer update, explain verified checkpoint generations, and prove continuation under a declared runtime. Prerequisites: optimizer history, schedule position, random sampling, loss accounting and the complete training experiment. Weight export, same-environment continuation and survival of a deleted hosted runtime are different claims.

### Training is a state transition, not a bag of weights

Represent training state at a completed update $t$ as $S_t$, and the next update as $S_{t+1}=U(S_t)$. The state includes more than $\theta_t$: optimizer moments determine the update direction/scale; the schedule determines the rate; the sampler determines the records; RNG states determine future random choices. Model buffers and module mode can affect the forward computation too.

Consider one scalar with Adam rate 0.1, $\beta_1=0.9$, $\beta_2=0.999$, negligible epsilon and first gradient 2. Its corrected moments are 2 and 4, so the first parameter move is approximately -0.1. If the next gradient is zero, the stored moments become $m_2=0.18$ and $v_2=0.003996$. Correcting their bias gives

$$
\begin{aligned}
\widehat m_2&=0.18/(1-0.9^2)≈0.947368,\\
\widehat v_2&=0.003996/(1-0.999^2)≈1.998999,\\
\Delta\theta_2&≈-0.1\frac{0.947368}{\sqrt{1.998999}}≈-0.067006.
\end{aligned}
$$

A fresh optimizer loaded with only the same weights sees zero first gradient and makes zero data-driven movement instead. Identical weights at reload time therefore do not imply identical next weights. [Checkpoint guidance](source:checkpoint-docs).

### Define the save boundary and state owners

The [reference snapshot](examples/prerequisite_checkpoint.py) saves only completed FP32 CPU updates with no parameter gradients pending. Saving partway through accumulation would also require partial gradients, loss denominator, microbatch position and consumed data/RNG state. This profile rejects that condition rather than pretending it is resumable.

| State | Why the next update needs it | Reference owner |
| --- | --- | --- |
| Model parameters and buffers | Same forward function and registered state | `model.state_dict()` |
| Adam moments and optimizer step | Same adaptive update, including bias correction | `optimizer.state_dict()` |
| Scheduler position and current rates | Same rate at the next global update | `scheduler.state_dict()` |
| Shuffled order, cursor, epoch and explicit generator | Same next records and future shuffle | `Sampler.state_dict()` |
| Python, NumPy and PyTorch CPU RNG | Same subsequent draws from each used source | `random_state()` |
| Update, examples seen and metric position | Distinguish durable progress and replayed work | `TrainingState` counters |
| Best validation loss and model mode | Preserve declared monitoring and behavior | Snapshot fields |
| Source/config/data/runtime identity and precision | Detect a changed experiment before continuation | `state.identity()` and schema |

This numerical task has no tokenizer. A text model must add the exact vocabulary, merge ranks, special IDs and tokenizer hash, along with split/data identity. Same vocabulary size alone is insufficient: swapping the meanings of two IDs changes every associated embedding/output row without changing a shape. GPU/AMP training must account for device RNG and scaler state as applicable; the current reference explicitly supports FP32 CPU and no scaler.

### A seed describes a start; RNG state describes a position

With 64 training records, batch size 16 and one microbatch per update, update 3 has consumed 48 records in the initial shuffled order. The next update consumes its remaining 16. Restoring just seed 7 and generating a new initial order restarts its cursor at zero. The sampler's explicit generator controls later shuffles; it is distinct from PyTorch's global CPU RNG.

Model initialization itself consumes random draws. Set up fresh objects first, load parameters/optimizer/schedule/sampler, then restore all saved RNG states immediately before the next draw. Restoring RNG before constructing a new random model would consume the saved stream during setup. Worker prefetch and streaming datasets add consumed-versus-queued positions; the first parity drill keeps sampling in one process. [Reproducibility scope](source:randomness).

### Verify a generation before treating it as a recovery copy

The publisher uses a new numbered directory for each generation and writes `state.pt`, `manifest.json` and `complete.json`. It performs these steps:

1. Clone a stable completed-update snapshot, serialize it in a temporary local directory, and check loadability.
2. Reserve storage for retained good copies plus the new generation; write into a new persistent directory.
3. Re-read state bytes and check them against the serialized bytes; load the persisted file.
4. Write a manifest containing format, byte size, state SHA-256, identity SHA-256 and update position; re-read it.
5. Write a completion marker containing the manifest hash **last**. Verify the entire generation through the same recovery predicate.
6. Only then rotate older compatible verified generations, retaining at least two.

A directory name or presence of `complete.json` is insufficient. Recovery checks the marker's manifest hash, the manifest's size/hash and the payload schema/identity. A partially written marker or state is rejected. This gives logical all-or-nothing **acceptance** of generations. The reference does not implement an fsync/rename durability protocol or prove remote-storage survival by checksums. Mounted storage behavior must be tested through actual runtime deletion; do not infer it from successful local reads.

The 8 MiB reference run budget reserves compatible retained copies plus an in-progress generation and counts foreign/incomplete files. The free-space guard leaves 1 MiB beyond the next write. A failure stops training rather than deleting the only good checkpoint to fit an unverified one. Source hashes and readback checks establish integrity of a known package, not independent proof of its origin.

[[diagram:prereq-recovery]]

### Restore compatible state and report fallback

Recovery scans newest numbered generations first, rejecting incomplete/corrupt entries and incompatible identity. It loads a compatible verified generation and reports any skipped candidates. A changed total goal, source file, data grid, hidden width or runtime can change identity. Deliberately loading weights into a new experiment is a **warm start** with a separate run identity; it is not exact continuation.

Identity, boundary and model mappings are checked before the reference mutates parameters. If a subsequent restoration operation raises, discard those newly constructed objects rather than continuing from a partly restored session. There is no implicit fallback from `resume` to `new`. Likewise, `new` rejects a nonempty run directory.

If durable update 100 is followed by a crash after update 137, recovery restores 100 and replays 37 updates. Printed progress is not persistent progress. Include the selected generation, skipped generations, restored update, new session ID and advanced update in the report. The sampler and RNG also roll back, so these replayed records must not be counted as previously unseen training evidence.

### Run a complete local restore-and-corruption probe

From the course `examples/` directory with the frozen interpreter:

```python
from pathlib import Path
import tempfile
import torch
from prerequisite_mlp import make_training_state, update
from prerequisite_checkpoint import save_generation, load_latest, snapshot
from prerequisite_mlp_checks import compare

torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='tutorial0-recovery-') as folder:
    directory = Path(folder)
    uninterrupted = make_training_state(seed=7, total_steps=1000)
    for _ in range(3):
        update(uninterrupted)
    saved = save_generation(directory, uninterrupted)
    expected_update = update(uninterrupted)
    expected_state = snapshot(uninterrupted)

    recovered = make_training_state(seed=7, total_steps=1000)
    restored = load_latest(directory, recovered)
    assert restored['step'] == 3
    actual_update = update(recovered)
    compare(expected_update, actual_update)
    compare(expected_state, snapshot(recovered))
    newest = save_generation(directory, recovered)
    (directory / newest['generation'] / 'state.pt').write_bytes(b'broken')
    fallback = make_training_state(seed=7, total_steps=1000)
    report = load_latest(directory, fallback)
    assert report['step'] == 3
    assert report['skipped'][0]['reason'] == 'incomplete or corrupt'
    print('same update:', recovered.step, 'fallback update:', fallback.step)
```

Expect `same update: 4 fallback update: 3`. The compared state includes Adam, scheduler, sampler, counters and RNG, not only parameters. Corruption is confined to an automatically removed temporary fixture. This snippet constructs new objects in one process; the following suite performs the stronger separate-process continuation test:

```bash
python prerequisite_mlp_checks.py
```

Its replay drill saves at update 3, checks subsequent Python/NumPy/PyTorch draws and five updates, then launches a separate interpreter to recover and reproduce update 8 with exact equality under the declared CPU profile. Other checks cover incomplete/corrupt newest generations, pending-gradient and identity rejection, retained fallback copies, storage guards and persistence failure. Inspect their assertions in [the editable checks](examples/prerequisite_mlp_checks.py).

### Keep the experiment goal fixed across session slices

Choose a fresh run name for this two-command drill:

```bash
python prerequisite_mlp.py --mode new --run-dir /tmp/tutorial0-sliced-run \
  --total-steps 1000 --max-updates 25 --checkpoint-updates 10
python prerequisite_mlp.py --mode resume --run-dir /tmp/tutorial0-sliced-run \
  --total-steps 1000 --max-updates 25 --checkpoint-updates 10
```

The second restores update 25 and finishes at 50. `total-steps` is the unchanged experiment goal; `max-updates` caps additional work in this session. Changing the goal to 50 changes the recorded configuration. Local equality is limited to the measured runtime/source/settings; it is not a guarantee across PyTorch releases, platforms, numerical precision or CPU/GPU changes.

### Exercise and notes

[[practice:recovery-transfer]]

In [my study notes](project/study-notes.md), list the state needed for the **next** batch and update, then draw the generation's write/verify/complete/rotate order. Predict the restored update after a failed newer save, distinguish warm start from resume, and record which fresh-process comparison you actually performed. Hosted persistence remains pending until Lesson 25's deleted-runtime drill succeeds.

Continue with [paper assembly and the implementation handoff](lessons/11-colab-and-handoff.md).
