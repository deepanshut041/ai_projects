"""Train a small two-layer classifier and inspect its counted updates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    input_features: int = 2
    hidden_features: int = 8
    num_classes: int = 2
    batch_size: int = 16
    accumulation_steps: int = 1
    learning_rate: float = 0.03
    warmup_steps: int = 20
    total_steps: int = 1000
    gradient_norm_limit: float = 1.0
    seed: int = 7

    def __post_init__(self):
        if self.input_features != 2 or self.num_classes != 2:
            raise ValueError("The opposite-sign fixture has two features and two classes")
        if min(self.hidden_features, self.batch_size, self.accumulation_steps, self.total_steps) < 1:
            raise ValueError("Widths, batch sizes, accumulation and total updates must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("Warmup must finish before the total-update goal")
        if self.learning_rate <= 0 or self.gradient_norm_limit <= 0:
            raise ValueError("Learning rate and gradient norm limit must be positive")


class TinyClassifier(nn.Module):
    def __init__(self, input_features=2, hidden_features=8, num_classes=2, *, config: Config | None = None):
        super().__init__()
        self.config = config or Config(input_features=input_features, hidden_features=hidden_features, num_classes=num_classes)
        self.hidden = nn.Linear(self.config.input_features, self.config.hidden_features)
        self.activation = nn.Tanh()
        self.output = nn.Linear(self.config.hidden_features, self.config.num_classes)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.config.input_features:
            raise ValueError("Features must have shape [batch,2]")
        return self.output(self.activation(self.hidden(x)))


def fixture():
    """Return disjoint structured points; class one means opposite feature signs."""
    train_coordinates = [-1.5, -1.1, -0.7, -0.3, 0.3, 0.7, 1.1, 1.5]
    validation_coordinates = [-1.3, -0.9, -0.5, 0.5, 0.9, 1.3]

    def grid(coordinates):
        features = torch.tensor([[left, right] for left in coordinates for right in coordinates], dtype=torch.float32)
        labels = (features[:, 0] * features[:, 1] < 0).long()
        return features, labels

    return {"train": grid(train_coordinates), "validation": grid(validation_coordinates)}


class Sampler:
    def __init__(self, size, seed):
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor = 0
        self.epoch = 0

    def next(self, count):
        if count < 1:
            raise ValueError("A batch must request at least one record")
        indices = []
        for _ in range(count):
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.order), generator=self.generator)
                self.cursor = 0
                self.epoch += 1
            indices.append(int(self.order[self.cursor]))
            self.cursor += 1
        return torch.tensor(indices, dtype=torch.long)

    def state_dict(self):
        return {"order": self.order.clone(), "cursor": self.cursor, "epoch": self.epoch, "generator": self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if sorted(state["order"].tolist()) != list(range(len(self.order))) or not 0 <= state["cursor"] <= len(self.order) or state["epoch"] < 0:
            raise ValueError("Sampler order, cursor or epoch is invalid")
        self.order = state["order"].clone()
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]
        self.generator.set_state(state["generator"])


def warmup_factor(step, warmup_steps):
    return min((step + 1) / max(1, warmup_steps), 1.0)


@dataclass
class TrainingState:
    config: Config
    model: TinyClassifier
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    sampler: Sampler
    data: dict
    step: int = 0
    examples_seen: int = 0
    best_validation_loss: float | None = None

    def identity(self):
        records = {name: {"features": features.tolist(), "labels": labels.tolist()} for name, (features, labels) in self.data.items()}
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("prerequisite_mlp.py", "prerequisite_checkpoint.py") if Path(__file__).with_name(name).exists()}
        return {"schema": 1, "task": "opposite-sign-classification-v1", "config": asdict(self.config), "data_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(), "source_sha256": sources, "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__), "parameter_devices": sorted({str(parameter.device) for parameter in self.model.parameters()}), "parameter_dtypes": sorted({str(parameter.dtype) for parameter in self.model.parameters()}), "platform": platform.system(), "machine": platform.machine(), "threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8}, "schedule": "linear-warmup-then-constant-v1"}


def make_training_state(seed=7, total_steps=1000, hidden_features=8, batch_size=16, accumulation_steps=1):
    config = Config(seed=seed, total_steps=total_steps, hidden_features=hidden_features, batch_size=batch_size, accumulation_steps=accumulation_steps, warmup_steps=min(20, total_steps - 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyClassifier(config=config).cpu().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_factor(step, config.warmup_steps))
    return TrainingState(config, model, optimizer, scheduler, Sampler(64, seed + 84), fixture())


def nll_statistics(logits, labels):
    if logits.ndim != 2 or labels.shape != (logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError("Scores require [batch,classes] and integer labels [batch]")
    count = len(labels)
    if count == 0:
        raise ValueError("Classification loss needs at least one target")
    loss_sum = F.cross_entropy(logits, labels, reduction="sum")
    correct = int((logits.argmax(-1) == labels).sum())
    return loss_sum, count, correct


def update(state: TrainingState, batches=None):
    if state.step >= state.config.total_steps:
        raise ValueError("The recorded experiment has reached its total-update goal")
    if batches is None:
        features, labels = state.data["train"]
        indices = [state.sampler.next(state.config.batch_size) for _ in range(state.config.accumulation_steps)]
        batches = [(features[index], labels[index]) for index in indices]
    denominator = sum(len(labels) for _, labels in batches)
    if denominator < 1:
        raise ValueError("An update needs supervised records")
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    loss_value, correct = 0.0, 0
    learning_rate_used = state.optimizer.param_groups[0]["lr"]
    for features, labels in batches:
        loss_sum, _, batch_correct = nll_statistics(state.model(features), labels)
        if not torch.isfinite(loss_sum):
            state.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite loss; no optimizer update was made")
        (loss_sum / denominator).backward()
        loss_value += float(loss_sum.detach())
        correct += batch_correct
    norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.config.gradient_norm_limit, error_if_nonfinite=True)
    state.optimizer.step()
    state.scheduler.step()
    state.step += 1
    state.examples_seen += denominator
    state.optimizer.zero_grad(set_to_none=True)
    return {"step": state.step, "examples": denominator, "examples_seen": state.examples_seen, "loss": loss_value / denominator, "correct": correct, "accuracy": correct / denominator, "learning_rate_used": learning_rate_used, "gradient_norm_before_clip": float(norm)}


@torch.no_grad()
def evaluate(state: TrainingState, split="validation"):
    if split not in state.data:
        raise ValueError("Choose the declared train or validation split")
    was_training = state.model.training
    state.model.eval()
    try:
        features, labels = state.data[split]
        loss_sum, count, correct = nll_statistics(state.model(features), labels)
        return {"split": split, "loss": float(loss_sum) / count, "correct": correct, "count": count, "accuracy": correct / count}
    finally:
        state.model.train(was_training)


def resources(state, elapsed_seconds):
    parameter_count = sum(parameter.numel() for parameter in state.model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in state.model.parameters())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(peak if platform.system() == "Darwin" else peak * 1024)
    return {"device": "cpu", "precision": "fp32", "parameters": parameter_count, "parameter_bytes": parameter_bytes, "process_peak_rss_bytes": rss_bytes, "elapsed_seconds": elapsed_seconds, "completed_updates": state.step}


def run_session(directory, *, mode, total_steps=1000, max_updates=1000, max_seconds=3600, checkpoint_updates=100, checkpoint_seconds=300, seed=7, progress=None):
    from prerequisite_checkpoint import load_latest, save_generation

    if mode not in ("new", "resume") or min(max_updates, max_seconds, checkpoint_updates, checkpoint_seconds) <= 0:
        raise ValueError("Choose new/resume explicitly and positive session/checkpoint limits")
    if max_updates > 1000 or max_seconds > 3600 or checkpoint_updates > 100 or checkpoint_seconds > 300 or not math.isfinite(max_seconds) or not math.isfinite(checkpoint_seconds):
        raise ValueError("The prerequisite profile caps sessions at 1000 updates/60 minutes and saves at 100 updates/5 minutes")
    directory = Path(directory)
    if mode == "new" and directory.exists() and any(directory.iterdir()):
        raise ValueError("A new run requires an empty directory; choose a new name or explicit resume")
    state = make_training_state(seed=seed, total_steps=total_steps)
    start = time.perf_counter()
    session_id = uuid.uuid4().hex[:12]
    rollback = None
    if mode == "resume":
        rollback = load_latest(directory, state)
    else:
        save_generation(directory, state)
    start_step = state.step
    last_saved_step = state.step
    last_saved_time = time.perf_counter()
    before = evaluate(state)
    last_update = None
    saves = []
    while state.step < state.config.total_steps and state.step - start_step < max_updates and time.perf_counter() - start < max_seconds:
        last_update = update(state)
        now = time.perf_counter()
        if progress is not None and (state.step == start_step + 1 or state.step % 100 == 0):
            progress({"session": session_id, **last_update, "elapsed_seconds": now - start})
        if state.step - last_saved_step >= checkpoint_updates or now - last_saved_time >= checkpoint_seconds:
            saved = save_generation(directory, state)
            saves.append(saved)
            if progress is not None:
                progress({"session": session_id, "verified_checkpoint": saved})
            last_saved_step, last_saved_time = state.step, time.perf_counter()
    after = evaluate(state)
    state.best_validation_loss = after["loss"] if state.best_validation_loss is None else min(state.best_validation_loss, after["loss"])
    saved = save_generation(directory, state)
    saves.append(saved)
    elapsed = time.perf_counter() - start
    return {"mode": mode, "session": session_id, "identity": state.identity(), "before": before, "after": after, "last_update": last_update, "restored": rollback, "persistent_saves": saves, "resources": resources(state, elapsed), "remaining_updates": state.config.total_steps - state.step, "hosted_execution": "unverified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("new", "resume"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--checkpoint-updates", type=int, default=100)
    parser.add_argument("--checkpoint-seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run_session(args.run_dir, mode=args.mode, total_steps=args.total_steps, max_updates=args.max_updates, max_seconds=args.max_seconds, checkpoint_updates=args.checkpoint_updates, checkpoint_seconds=args.checkpoint_seconds, seed=args.seed, progress=lambda event: print(json.dumps(event), file=sys.stderr, flush=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
