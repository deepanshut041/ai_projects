## Lesson 10 — run one bounded experiment across sessions

The model, loss, sampler, evaluation and persistent store now exist. Join them with one importable trainer. The terminal and Colab notebook will call the same code; the notebook will not grow its own hidden training loop.

### Start with the resolved experiment contract

Create `cpu.json` in the learner root for the first checked run. You may organize later profiles under `configs/`, but the commands below use this exact path. The total budget is 300 optimizer updates, not 300 per session. This config keeps the roadmap's model dimensions and uses dropout 0.1 for corpus training.

<!-- file:cpu.json mode:write -->
```json
{
  "model": {
    "vocab": 260,
    "context": 64,
    "width": 64,
    "heads": 4,
    "layers": 2,
    "hidden": 256,
    "dropout": 0.1
  },
  "train": {
    "total": 300,
    "warmup": 20,
    "peak_lr": 0.001,
    "floor_lr": 0.0001,
    "batch": 8,
    "accumulation": 1,
    "decay": 0.1,
    "clip": 1.0,
    "seed": 17
  }
}
```

Copy it to `colab-free.json` for the first hosted smoke. Using the same small architecture makes recovery easier to diagnose; a Free label is not permission to allocate a larger model. Measure actual throughput/memory before proposing a larger, separately identified run.

Create `docs/experiment-01.md` **before** starting. Record this config, split/tokenizer identities, initial validation and unigram baseline, planned total updates, checkpoint frequency, storage budget and sample prompts. The author's declared demonstration criterion is at least 0.05 nats improvement over initial held-out NLL, with comparison to the unigram baseline reported separately. That modest teaching criterion does not establish a useful general-purpose language model.

### Write the trainer and trace one update

Create `gpt_trainer.py` with this complete implementation. Start reading `TrainConfig`, then construction, then `update`, then validation/state/restore. Stop between methods and connect each to a file you already wrote.

<!-- file:gpt_trainer.py mode:write -->
```python
"""FP32 reference trainer, explicit completed-update boundaries, no hidden notebook loop."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import platform
import random
import torch
from gpt_config import Config
from gpt_data import Batches, windows
from gpt_model import GPT
from gpt_objective import nll_sum, lr_at, optimizer_for, evaluate
from gpt_tokenizer import PAD


@dataclass(frozen=True)
class TrainConfig:
    total: int = 300
    warmup: int = 20
    peak_lr: float = 0.001
    floor_lr: float = 0.0001
    batch: int = 8
    accumulation: int = 1
    decay: float = 0.1
    clip: float = 1.0
    seed: int = 17

    def __post_init__(self):
        lr_at(1, self.total, self.warmup, self.peak_lr, self.floor_lr)
        if min(self.batch, self.accumulation) < 1 or self.clip <= 0 or self.decay < 0:
            raise ValueError("Invalid optimizer/batch contract")


def source_identity():
    # Bytes include dirty edits; do not identify a run by Git HEAD alone.
    names = ["gpt_config.py", "gpt_tokenizer.py", "gpt_data.py", "gpt_attention.py",
             "gpt_model.py", "gpt_objective.py", "gpt_checkpoint.py", "gpt_trainer.py",
             "gpt_train.py"]
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
```

The immutable training settings describe the whole experiment. `source_identity` reads actual source bytes, so a dirty working tree has a specific identity. Finish both trainer files on this page before instantiating the class: the identity list includes the command entry point introduced below.

### Construct the model and bind the data

Append the class declaration and initializer. It owns the model, optimizer, sampler, validation arrays and counters. `self` is the particular trainer instance, like the object reached through `this` in C++. `Config(**dictionary)` unpacks named fields from the recorded configuration.

<!-- file:gpt_trainer.py mode:append -->
```python


class Trainer:
    def __init__(self, model_config, train_config, splits, tok, manifest, device="cpu"):
        if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Choose an available cpu/cuda device explicitly")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(train_config.seed)
        torch.manual_seed(train_config.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(train_config.seed)
        if model_config.vocab != tok.size:
            raise ValueError("Model vocabulary must equal tokenizer vocabulary")
        self.device, self.config = device, train_config
        self.model = GPT(model_config).to(device)
        self.optimizer = optimizer_for(self.model, train_config.peak_lr, train_config.decay)
        x, y = windows(splits["train"], tok, model_config.context)
        self.batches = Batches(x, y, train_config.seed+1)
        self.valid = windows(splits["validation"], tok, model_config.context)
        self.identity = {"model": asdict(model_config), "train": asdict(train_config),
                         "data": manifest, "tokenizer": tok.identity, "source": source_identity(),
                         "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                                     "device": device, "cuda": torch.version.cuda,
                                     "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.machine(),
                                     "threads": 1, "deterministic": True, "precision": "float32"}}
        self.step, self.tokens = 0, 0
        self.best_nll, self.best_step = float("inf"), 0
        self.events = 0
```

Check the state on paper: step=0, tokens=0, best loss=infinity, separate sampler seed=18. The first validation establishes a finite baseline; it does not increment the optimizer step. Both model vocabulary and tokenizer size must agree before a run can begin.

### Complete exactly one optimizer update

Append `update`. On a full default batch the denominator is 8×64=512; if a sampled document tail contains padding it is smaller. With accumulation two, the denominator includes valid labels from both microbatches before either backward call.

<!-- file:gpt_trainer.py mode:append -->
```python

    def update(self):
        c = self.config
        if self.step >= c.total:
            raise ValueError("Declared total-update budget is complete")
        self.model.train()
        batches = [self.batches.next(c.batch, self.device) for _ in range(c.accumulation)]
        denominator = sum(int((y != PAD).sum()) for _, y in batches)
        lr = lr_at(self.step+1, c.total, c.warmup, c.peak_lr, c.floor_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        summed = 0.0
        for x, y in batches:
            loss, _ = nll_sum(self.model(x), y)
            (loss / denominator).backward()
            summed += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.clip,
                                              error_if_nonfinite=True)
        if not math.isfinite(summed):
            raise FloatingPointError("Nonfinite loss; no checkpoint/update accepted")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens += denominator
        self.events += 1
        return {"step": self.step, "tokens": self.tokens, "nll": summed/denominator,
                "lr": lr, "gradient_norm_before_clip": float(norm), "event": self.events}
```

Trace the first update: global step zero selects schedule input one, gradients start empty, each loss sum is scaled by the shared denominator, clipping runs once and AdamW changes the parameters. Counters advance only after the step succeeds. A failed persistence operation later will still require rollback to the previous saved completed update.

### Evaluate, capture and restore

Append the remaining methods. Validation must preserve training mode and leave both sampler/dropout streams untouched. State capture requires cleared gradients; recovery validates the entire identity and restores RNG last.

<!-- file:gpt_trainer.py mode:append -->
```python

    def validation(self):
        result = evaluate(self.model, *self.valid, device=self.device)
        if not math.isfinite(result["nll"]):
            raise FloatingPointError("Nonfinite validation")
        if result["nll"] < self.best_nll:
            self.best_nll, self.best_step = result["nll"], self.step
        return result

    def state(self):
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError("Save only after a complete update and cleared gradients")
        return {"schema": 1, "identity": self.identity, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "step": self.step, "tokens": self.tokens,
                "schedule": {"next_update": self.step+1}, "scaler": None,
                "batch": self.batches.state(), "python_rng": random.getstate(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device == "cuda" else [],
                "best_nll": self.best_nll, "best_step": self.best_step, "events": self.events}

    def restore(self, state):
        if state["identity"] != self.identity or state["schedule"]["next_update"] != state["step"]+1:
            raise ValueError("Run identity/schedule mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.tokens = state["step"], state["tokens"]
        self.best_nll, self.best_step = state["best_nll"], state["best_step"]
        self.events = state["events"]
        self.batches.restore(state["batch"])
        # Restore RNG LAST, after construction/loading, before another batch/update.
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["cpu_rng"])
        if self.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
```

Construction fixes random seeds, builds the model/optimizer and binds text identities. Source identity hashes the relevant source files' actual bytes, including uncommitted edits. This avoids labelling two different working trees as the same run merely because Git HEAD matches.

`update` first draws all microbatches and counts their valid labels. It computes LR for global step+1, clears gradients, accumulates normalized loss gradients, clips once, applies one optimizer step and clears again. Only then does it increase completed-step and token counters. The batch generator belongs to the sampler; the model RNG belongs to dropout. Both survive recovery.

`validation` uses the fixed validation arrays and restores model mode. It updates best loss and the step that produced it. The state includes those fields. `restore` validates identity before loading, then restores RNG **after** construction/loading so setup cannot consume the next saved random draws. No live import from P00 or another workspace is needed.

[[practice:state-order]]

### Add an explicit new/resume command

Create `gpt_train.py`:

<!-- file:gpt_train.py mode:write -->
```python
"""Run from the learner project: python gpt_train.py --help."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import uuid
# Required before CUDA context creation for deterministic CUDA matrix products.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from gpt_config import Config
from gpt_data import load_data
from gpt_trainer import Trainer, TrainConfig
from gpt_checkpoint import publish, recover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=["new", "resume"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--save-seconds", type=float, default=300)
    parser.add_argument("--storage-mib", type=int, default=64)
    args = parser.parse_args()
    if min(args.updates, args.seconds, args.save_every, args.save_seconds, args.storage_mib) <= 0:
        parser.error("Session/save/storage budgets must be positive")
    resolved = json.loads(args.config.read_text())
    splits, tok, manifest = load_data(args.data)
    trainer = Trainer(Config(**resolved["model"]), TrainConfig(**resolved["train"]),
                      splits, tok, manifest, args.device)
    if args.mode == "new":
        args.run.mkdir(parents=True, exist_ok=False)
        (args.run / "run.json").write_text(json.dumps(trainer.identity, indent=2))
    else:
        if json.loads((args.run / "run.json").read_text()) != trainer.identity:
            raise ValueError("Run identity differs; source/config/device changes require a new run")
        state, recovery = recover(args.run, trainer.identity)
        trainer.restore(state)
        print(json.dumps({"recovery": recovery}), flush=True)
    session = uuid.uuid4().hex[:8]
    begin, start_step = time.monotonic(), trainer.step
    last_save = begin
    def save():
        result = trainer.validation()
        report = publish(args.run, trainer.state(), args.storage_mib*1024*1024)
        print(json.dumps({"session": session, "validation": result, "persistent": report}), flush=True)
    # Establish a verified starting checkpoint; failures abort before more training.
    if args.mode == "new":
        save()
    try:
        while (trainer.step < trainer.config.total and trainer.step-start_step < args.updates
               and time.monotonic()-begin < args.seconds):
            event = trainer.update()
            event.update(session=session, elapsed_seconds=time.monotonic()-begin)
            print(json.dumps(event), flush=True)
            with (args.run / "metrics.jsonl").open("a") as output:
                output.write(json.dumps(event)+"\n")
            if trainer.step % args.save_every == 0 or time.monotonic()-last_save >= args.save_seconds:
                save()
                last_save = time.monotonic()
        save()
    except KeyboardInterrupt:
        # May be inside backward/step: do not serialize partially applied work.
        print("Interrupted: resume the last verified generation; unsaved updates roll back.", flush=True)
        raise SystemExit(130)
    print(json.dumps({"stopped_at": trainer.step, "run": str(args.run),
                      "session_updates": trainer.step-start_step}), flush=True)


if __name__ == "__main__":
    main()
```

`--mode new` refuses an existing run directory. `--mode resume` requires compatible run metadata and a verified generation; it never becomes a new experiment silently. The session ID distinguishes repeated global event positions after rollback. For analysis, use run/global-step/session together and note which history belongs to the restored branch; counting every appended log row would overcount replayed work.

The process checks update/time limits between updates. Evaluation/save can extend the wall-clock budget; the loop is bounded cooperatively, not a hard real-time timer. A persistence error aborts rather than training ahead unprotected. A keyboard interrupt returns code 130 and leaves the last verified checkpoint; it intentionally avoids saving potentially partial optimizer state.

### Run, stop normally, then resume

Run these from the learner root after the preceding files exist:

```bash
python gpt_train.py --data data/alice-byte --run runs/alice-v1 --config cpu.json --mode new --updates 150 --save-every 50
python gpt_train.py --data data/alice-byte --run runs/alice-v1 --config cpu.json --mode resume --updates 150 --save-every 50
python gpt_sample.py --data data/alice-byte --run runs/alice-v1 --prompt "Alice " --tokens 80
```

The first command should stop at global update 150. The second should report restored update 150, then stop at 300. It continues the original LR curve. The initial save establishes a recoverable starting point before updates. Save output includes generation identity, state bytes, retained copies and rejected folders; metric output includes update, valid-token count, loss, LR and pre-clip norm.

Check one update record manually. With eight full T=64 windows and accumulation one, it adds 512 valid labels; a tail window can make the count smaller. An update with loss decreasing but tokens always zero would reveal a bookkeeping error. Rising validation while training loss falls is a signal to investigate overfitting or split shift, not a reason to print a more flattering metric.

### Test the state contract separately from training quality

Save [test_gpt_training.py](examples/test_gpt_training.py) and run:

```bash
python -m unittest test_gpt_data test_gpt_model test_gpt_training -v
```

The reference has twenty-one tests. Training tests include nonzero-dropout resume, unchanged validation RNG, corrupt-newest fallback, partial-copy/storage failures, incompatible identity rejection, seeded/EOS/bounded generation and token-weighted evaluation. Read the fixtures and change one value to predict a failure.

For your child project add a **fresh-process** drill in `tests/test_fresh_process_resume.py`: run 8 updates uninterrupted, then 4 plus process exit and 4 restored. Use a small fixed corpus, nonzero dropout and matching CPU runtime. Compare model tensors, Adam state, sampler/CPU/Python RNG, global tokens/steps, schedule and monitoring fields. The author has run that drill; your own evidence belongs in the child. Do not compare only the final loss and call all state equal.

[[practice:resume-vs-warm-start]]

### Read the observed reference run honestly

The [reference-run record](project/reference-runs.md) contains the actual initial/final validation, unigram comparison, fitting result, checkpoint sizes, generated samples and remaining limits. Compare your run only under matching source/config/data identities. If it differs, preserve the result and inspect those identities before changing hyperparameters.

Required remaining failure work in your child: abrupt process termination inside accumulation and during persistent writes, a full run-new overwrite test, and any worker/AMP/device transitions you introduce. Use a separate disposable run for injected failures. Do not corrupt the only copy of an active training run.

The same experiment can now span local sessions. Next move its source and persistent run contract into a thin Free notebook.

[Continue to Lesson 11](lessons/11-colab.md) · [Reference results](project/reference-runs.md)

### Complete files from this lesson

[cpu.json](examples/cpu.json) · [gpt_trainer.py](examples/gpt_trainer.py) · [gpt_train.py](examples/gpt_train.py)

[Matching initial Colab Free profile](examples/colab-free.json)
