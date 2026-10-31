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
