## Lesson 9 — save an update you can actually recover

Suppose training has completed update 150. Saving only the model weights lets you generate text later, but it does not let you continue the same optimization path. Adam's moments, the next learning rate, the next batch and the dropout randomness all matter.

### Define the recovery boundary

We save immediately after a completed optimizer update, with gradients cleared and no partially accumulated microbatch. The saved step means “this many updates are complete.” Its next schedule input is step+1. If interrupted in the middle of backward or an optimizer step, discard that partial work and restore the previous verified checkpoint. Do not serialize half-updated weights in a generic shutdown handler.

A state at update 100 followed by a crash at update 137 rolls back up to 37 completed updates, plus any in-progress update. A five-minute save interval does not guarantee five-minute maximum loss: an unusually slow update or storage write can extend it. Measure actual rollback and persistence latency.

[[diagram:recovery]]

### Know what belongs in the state

| Saved field | Why weights alone cannot replace it |
| --- | --- |
| Model tensors | Learned parameters and any registered buffers |
| Optimizer state | Adam first/second moments, optimizer step and parameter grouping |
| Global step/tokens | Schedule position, experiment budget and consumed work |
| Sampler generator/draw count | The next randomly selected windows |
| Python, CPU and applicable CUDA RNG | Dropout and other random operations after recovery |
| Config/data/tokenizer/source/runtime identity | Reject incompatible continuation rather than mixing experiments |
| Best metric/step and event position | Preserve evaluation selection and identify replayed metrics |
| Precision/schedule/scaler | This reference has FP32, a pure schedule function and no AMP scaler; these choices are explicit |

The [checkpoint documentation](source:checkpoint) motivates preserving optimizer state. The larger identity/RNG/data protocol here follows this workspace's Colab contract. If you add NumPy, worker processes, AMP, early stopping or queued data prefetch, their relevant state must join the protocol and its tests. An absent feature must not be described as saved state.

### Publish a new generation, never overwrite the only good one

Create `gpt_checkpoint.py`. Read its three jobs in this order: identify valid completed generations, select the newest compatible state, and publish/retain a new one.

<!-- file:gpt_checkpoint.py mode:write -->
```python
"""Verified immutable generations. Local tests do not prove Drive durability."""
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import tempfile
import uuid
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(root):
    good, rejected = [], []
    for folder in sorted(Path(root).glob("step-*")):
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
            if (folder / "COMPLETE").read_text() != sha(folder / "manifest.json"):
                raise ValueError("Completion record mismatch")
            file = folder / "state.pt"
            if file.stat().st_size != manifest["bytes"] or sha(file) != manifest["sha256"]:
                raise ValueError("State checksum mismatch")
            state = torch.load(file, map_location="cpu", weights_only=True)
            if manifest["schema"] != 1 or state["schema"] != 1 or state["step"] != manifest["step"] or state["identity"] != manifest["identity"]:
                raise ValueError("State/manifest mismatch")
            good.append((state["step"], folder, state))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            rejected.append({"generation": folder.name, "reason": str(error)})
    return sorted(good, key=lambda entry: (entry[0], entry[1].name)), rejected


def recover(root, identity):
    good, rejected = verified(root)
    if not good:
        raise ValueError("No verified checkpoint; resume cannot become a new run")
    step, folder, state = good[-1]
    if state["identity"] != identity:
        raise ValueError("Incompatible config/source/data/tokenizer/runtime; use an explicit new run")
    return state, {"generation": folder.name, "restored_step": step, "rejected": rejected}


def publish(root, state, budget_bytes=64*1024*1024):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        local = Path(scratch) / "state.pt"
        torch.save(state, local)
        check = torch.load(local, map_location="cpu", weights_only=True)
        if check["step"] != state["step"]:
            raise ValueError("Local serialization failed")
        size = local.stat().st_size
        used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if used + size + 16384 > budget_bytes or shutil.disk_usage(root).free < 2*size+16384:
            raise OSError("Checkpoint storage budget exhausted; preserve old states and stop")
        folder = root / f'step-{state["step"]:08d}-{uuid.uuid4().hex[:8]}'
        folder.mkdir()
        shutil.copyfile(local, folder / "state.pt")
        if sha(local) != sha(folder / "state.pt"):
            raise OSError("Persistent copy did not verify")
        manifest = {"schema": 1, "step": state["step"], "bytes": size,
                    "sha256": sha(local), "identity": state["identity"]}
        (folder / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        # Written last; never depend on mounted-storage rename semantics.
        (folder / "COMPLETE").write_text(sha(folder / "manifest.json"))
    good, rejected = verified(root)
    if not any(path == folder for _, path, _ in good):
        raise OSError("New persistent generation is not loadable")
    keep = {entry[1] for entry in good[-2:]}
    best = [entry for entry in good if entry[2]["best_step"] == entry[0]]
    if best:
        keep.add(min(best, key=lambda entry: entry[2]["best_nll"])[1])
    for _, old, _ in good:
        if old not in keep:
            shutil.rmtree(old)
    return {"generation": folder.name, "step": state["step"], "state_bytes": size,
            "retained": sorted(path.name for path in keep), "rejected": rejected}
```

A generation is a new directory containing `state.pt`, `manifest.json` and `COMPLETE`. First serialize and reload locally. Next copy to the persistent destination, check the copied bytes and write the manifest. Write completion last. Recovery still rechecks hashes and loadability; the completion marker alone does not make a file valid.

The state format contains tensors and primitive containers, so the loader uses `weights_only=True`. Even so, use checkpoints you produced and control. The checksum detects accidental changes relative to the manifest; it is not a signature proving who wrote the manifest.

Mounted storage may not have ordinary local rename semantics. This protocol avoids replacing the last good generation and does not depend on a single atomic rename. An interrupted copy leaves an incomplete folder, which recovery rejects. If the newest completed generation is corrupt, the older verified state remains a candidate and the report names the rejected generation.

### Retention has a space cost

We keep at least the two newest verified full states and the best verified state. Before pruning, the new generation must coexist with the retained ones and successfully verify. Budget enough space for this temporary extra copy, manifests and metrics. The reference checks its configured run-directory budget and available filesystem space and stops when it cannot persist; it does not silently continue training with only local protection.

The budget is a guardrail, not a filesystem quota or a guarantee about a remote account. The generated run stores metrics too; long logs and incomplete generations use space. Record the actual checkpoint size and retained footprint, and inspect failed/incomplete generations before cleanup. Never delete the only valid generation to make room for a speculative save.

[[scenario:recovery-choice]]
[[practice:recovery]]

### Four failure fixtures to understand

The Lesson 10 training test file will exercise these cases after a real Trainer state exists:

1. Save several completed updates, corrupt the newest state bytes, and recover an older verified update. Inspect the rollback report.
2. Inject a copy failure after partial bytes are written. Confirm that no completion record is accepted and the previous state still loads.
3. Set the storage budget to one byte. Saving must fail while old verified states remain intact.
4. Change tokenizer/config/source identity. Recovery must reject continuation even if tensor shapes happen to match.

A successful local injected failure tests your code path. It does not establish mounted Drive behavior after a deleted Colab runtime. The hosted drill remains a separate step.

### Keep identity changes explicit

Strict resume in this reference requires the same source bytes, model/training config, data/tokenizer and recorded runtime/device. If Colab provides a different device or version, fail clearly. A deliberately non-identical warm start or cross-device continuation can be a separate protocol with numerical checks, but it must not claim exact deterministic parity. [PyTorch's reproducibility guidance](source:reproducibility) explains why matching seeds alone does not promise identical results across platforms or releases.

You can now store and recover a complete state. Next connect this protocol to the real training loop and execute a bounded new-run/resume pair.

[Continue to Lesson 10](lessons/10-train-and-resume.md)

### Complete files from this lesson

[gpt_checkpoint.py](examples/gpt_checkpoint.py)
