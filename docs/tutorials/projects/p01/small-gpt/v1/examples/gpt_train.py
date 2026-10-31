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
