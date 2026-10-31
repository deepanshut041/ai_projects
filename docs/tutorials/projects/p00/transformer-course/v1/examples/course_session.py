"""Run a bounded training slice with explicit new/resume and persistent saves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch

from course_checkpoint import load_latest, save_generation


MIB = 1024 * 1024


def build(task, seed, total_steps, device, backbone=None):
    if task == "translation":
        if backbone is not None:
            raise ValueError("A RoBERTa backbone belongs only to a new QA run")
        from translation_train import make_state
        return make_state(seed=seed, steps=total_steps, device=device)
    from roberta_train import build_training_bundle, load_backbone_for_qa
    state = build_training_bundle(task, seed=seed, total_steps=total_steps, device=device)
    if backbone is not None:
        if task != "qa":
            raise ValueError("A backbone can initialize only QA")
        load_backbone_for_qa(state, backbone)
    return state


def checkpoint_fields(state, contract):
    generators = ({"mask": state.mask_generator, "batch": state.batch_generator}
                  if hasattr(state, "mask_generator") else {})
    return {"model": state.model, "optimizer": state.optimizer,
            "scheduler": state.scheduler, "identity": contract, "generators": generators}


def trainer_identity(state):
    return state.identity() if callable(state.identity) else state.identity


def source_identity():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("course_session.py", "course_checkpoint.py")}


def estimate_checkpoint_bytes(model):
    tensors = list(model.parameters()) + list(model.buffers())
    # Parameters, two Adam moments, buffers and conservative serialization overhead.
    return 4 * sum(value.numel() * value.element_size() for value in tensors) + 256 * 1024


def used_bytes(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())


def persistence_preflight(directory, estimate, *, keep=2, storage_budget_bytes=256 * MIB,
                          minimum_free_bytes=64 * MIB):
    directory = Path(directory)
    retention_bytes = (keep + 1) * estimate
    if retention_bytes > storage_budget_bytes or used_bytes(directory) + estimate > storage_budget_bytes:
        raise OSError("Run storage budget cannot fit retained checkpoints plus a new generation; increase its budget")
    free = shutil.disk_usage(directory).free
    if free < retention_bytes + minimum_free_bytes:
        raise OSError("Persistent storage has too little free space for checkpoint retention")
    probe = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".persistence-probe-", dir=directory, delete=False) as stream:
            probe = Path(stream.name)
            stream.write(b"persistent-read-write-check")
            stream.flush()
            os.fsync(stream.fileno())
        if probe.read_bytes() != b"persistent-read-write-check":
            raise OSError("Persistent readback differs from written bytes")
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)
    return {"free_bytes": free, "estimated_checkpoint_bytes": estimate,
            "retention_with_new_generation_bytes": retention_bytes,
            "storage_budget_bytes": storage_budget_bytes}


def input_counts(task, state):
    """Inspect the next sampled indices using a copy, without advancing its RNG."""
    if task == "translation":
        batch = state.train_batch
        tokens = int(batch["src_pad_mask"].sum() + batch["tgt_pad_mask"].sum())
        return int(batch["src_input_ids"].shape[0]), tokens
    probe = torch.Generator().set_state(state.batch_generator.get_state())
    examples = tokens = 0
    for _ in range(state.accumulation_steps):
        indices = torch.randint(len(state.data), (state.batch_size,), generator=probe).tolist()
        examples += len(indices)
        for index in indices:
            item = state.data[index]
            tokens += len(item) if task == "mlm" else sum(item.attention_mask)
    return examples, tokens


def run_session(task, run_directory, *, mode, total_steps=None, seed=7, device="cpu",
                additional_updates=1000, max_seconds=3600, save_updates=100,
                save_seconds=300, keep=2, storage_budget_bytes=256 * MIB,
                minimum_free_bytes=64 * MIB, backbone=None, export_backbone=False,
                emit=None):
    """Complete one bounded slice; failed persistence raises before another update."""
    if task not in {"translation", "mlm", "qa"} or mode not in {"new", "resume"}:
        raise ValueError("Choose translation/mlm/qa and an explicit new/resume mode")
    total_steps = (400 if task == "translation" else 120) if total_steps is None else total_steps
    if total_steps < 2 or min(additional_updates, max_seconds, save_updates, save_seconds) <= 0 or keep < 2:
        raise ValueError("Use a positive session/save budget, at least two total updates and keep >= 2")
    if storage_budget_bytes <= 0 or minimum_free_bytes < 0:
        raise ValueError("Storage budget must be positive and free-space reserve nonnegative")
    if export_backbone and task != "mlm":
        raise ValueError("Only an MLM run can export its backbone")
    directory = Path(run_directory).resolve()
    if mode == "new" and directory.exists():
        raise FileExistsError("New refuses an existing run directory; choose another run ID or resume")
    if mode == "resume" and not (directory / "run.json").is_file():
        raise FileNotFoundError("Resume requires the existing run manifest; it never starts a new run")
    if mode == "resume" and backbone is not None:
        raise ValueError("Resume restores its own backbone; do not supply new initialization weights")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    state = build(task, seed, total_steps, device, backbone)
    base_contract = {"trainer": trainer_identity(state), "session_source_sha256": source_identity()}
    configuration = {"schema": 1, "task": task, "seed": seed,
                     "total_steps": total_steps, "device": str(torch.device(device))}
    if mode == "new":
        directory.mkdir(parents=True, exist_ok=False)
        initializer = None if backbone is None else hashlib.sha256(Path(backbone).read_bytes()).hexdigest()
        contract = {**base_contract, "initial_backbone_sha256": initializer}
        manifest = {**configuration, "contract": contract}
    else:
        manifest = json.loads((directory / "run.json").read_text())
        if any(manifest.get(name) != value for name, value in configuration.items()):
            raise ValueError("Run task/seed/total schedule/device differs from its manifest")
        contract = {**base_contract, "initial_backbone_sha256": manifest["contract"]["initial_backbone_sha256"]}
        if contract != manifest["contract"]:
            raise ValueError("Run source/tokenizer/corpus/training protocol identity differs")
    estimate = estimate_checkpoint_bytes(state.model)
    budget = persistence_preflight(directory, estimate, keep=keep,
                                   storage_budget_bytes=storage_budget_bytes,
                                   minimum_free_bytes=minimum_free_bytes)
    if mode == "new":
        with (directory / "run.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
    session_id = uuid.uuid4().hex[:12]
    emit = emit or (lambda record: print(json.dumps(record), flush=True))
    started = last_save_time = time.monotonic()
    counters = {"completed_updates": 0, "attempts": 0, "examples_seen": 0,
                "input_tokens_seen": 0, "supervised_targets_seen": 0}
    generations = directory / "checkpoints"
    last_generation = None
    if mode == "resume":
        restored = load_latest(generations, **checkpoint_fields(state, contract))
        state.step = restored["step"]
        counters = restored["data_state"]
        if counters.get("completed_updates") != state.step:
            raise ValueError("Checkpoint progress counter differs from completed updates")
        last_generation = restored["generation"]
        emit({"event": "restored", "session": session_id, "step": state.step,
              "generation": last_generation, "skipped": restored["skipped"], **counters})
    first_step = last_save_step = state.step

    def persist():
        nonlocal last_generation, last_save_step, last_save_time
        save_started = time.monotonic()
        try:
            persistence_preflight(directory, estimate, keep=keep,
                                  storage_budget_bytes=storage_budget_bytes,
                                  minimum_free_bytes=minimum_free_bytes)
            generation = save_generation(generations, step=state.step, data_state=counters,
                                         keep=keep, **checkpoint_fields(state, contract))
        except Exception as error:
            emit({"event": "persistence_failed", "session": session_id,
                  "step": state.step, "last_verified_generation": last_generation,
                  "reason": str(error)})
            raise RuntimeError("Persistent save failed; training stopped before another update") from error
        last_generation, last_save_step = generation.name, state.step
        last_save_time = time.monotonic()
        emit({"event": "checkpoint", "session": session_id, "step": state.step,
              "generation": last_generation, "bytes": used_bytes(generation),
              "persistence_seconds": last_save_time - save_started})

    emit({"event": "preflight", "session": session_id, "mode": mode, "task": task,
          "step": state.step, "run_directory": str(directory), **budget})
    if mode == "new":
        persist()
    while state.step < total_steps and state.step - first_step < additional_updates and time.monotonic() - started < max_seconds:
        examples, tokens = input_counts(task, state)
        if task == "translation":
            from translation_train import update
            result = update(state)
        else:
            from roberta_train import training_step
            result = training_step(state)
        counters["attempts"] += 1
        counters["examples_seen"] += examples
        counters["input_tokens_seen"] += tokens
        counters["supervised_targets_seen"] += int(result.get("tokens", result.get("target_count", 0)))
        counters["completed_updates"] = state.step
        emit({"event": "update" if result.get("updated", True) else "no_targets",
              "session": session_id, "elapsed_seconds": time.monotonic() - started,
              "learning_rates": state.scheduler.get_last_lr(), **counters, **result})
        if state.step - last_save_step >= save_updates or time.monotonic() - last_save_time >= save_seconds:
            persist()
    # Include consumed RNG/data attempts even if the final batch had no MLM targets.
    if mode == "new" or counters["attempts"] > 0:
        persist()
    if export_backbone:
        from roberta_train import export_backbone as write_backbone
        write_backbone(state, directory / "backbone.pt")
    summary = {"event": "session_complete", "session": session_id, "task": task,
               "step": state.step, "additional_updates": state.step - first_step,
               "total_steps": total_steps, "elapsed_seconds": time.monotonic() - started,
               "generation": last_generation, **counters}
    emit(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("translation", "mlm", "qa"), required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("new", "resume"), required=True)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--additional-updates", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--save-updates", type=int, default=100)
    parser.add_argument("--save-seconds", type=float, default=300)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--storage-budget-mb", type=int, default=256)
    parser.add_argument("--minimum-free-mb", type=int, default=64)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--export-backbone", action="store_true")
    args = vars(parser.parse_args())
    args["storage_budget_bytes"] = args.pop("storage_budget_mb") * MIB
    args["minimum_free_bytes"] = args.pop("minimum_free_mb") * MIB
    run_session(args.pop("task"), args.pop("run_directory"), **args)


if __name__ == "__main__":
    main()
