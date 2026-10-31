"""Complete educational P00 model and bounded common trainer.

Run --check for correctness/recovery demonstrations; --steps N for a bounded
copy-task run. Checkpoints are created only in an explicitly supplied --run-dir.
This reference is not the independent learner project or hosted Colab proof.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import tempfile
import time
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from byte_bpe import BOS, EOS, PAD, SEP, ByteBPE, canonical, digest


@dataclass(frozen=True)
class Config:
    vocab: int = 260
    width: int = 32
    heads: int = 4
    hidden: int = 128
    layers: int = 1
    limit: int = 32
    dropout: float = 0.0

    def __post_init__(self):
        if (self.width % self.heads or self.width % 2 or self.layers < 1
                or self.vocab < 260 or self.limit < 2):
            raise ValueError("Invalid architecture dimensions")


class LayerNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.scale + self.bias


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_width = cfg.heads, cfg.width // cfg.heads
        self.q, self.k, self.v, self.out = [nn.Linear(cfg.width, cfg.width) for _ in range(4)]
        self.drop = nn.Dropout(cfg.dropout)

    def split(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.head_width).transpose(1, 2)

    def forward(self, query, memory, allowed, query_valid):
        q, k, v = self.split(self.q(query)), self.split(self.k(memory)), self.split(self.v(memory))
        # allowed is [B,1,Tq,Tk]; True always means visible in this reference.
        if not allowed.any(-1).all():
            raise ValueError("All-masked attention row; validate source/prefix")
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        mixed = self.drop(weights) @ v
        joined = mixed.transpose(1, 2).contiguous().reshape(query.shape)
        return self.out(joined) * query_valid.unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, cfg, decoder=False):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.cross_attn = Attention(cfg) if decoder else None
        self.norms = nn.ModuleList([LayerNorm(cfg.width) for _ in range(3 if decoder else 2)])
        self.ffn = nn.Sequential(nn.Linear(cfg.width, cfg.hidden), nn.ReLU(),
                                 nn.Linear(cfg.hidden, cfg.width))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, allowed, valid, memory=None, cross_allowed=None):
        x = self.norms[0](x + self.drop(self.self_attn(x, x, allowed, valid)))
        if self.cross_attn is not None:
            x = self.norms[1](x + self.drop(self.cross_attn(x, memory, cross_allowed, valid)))
        x = self.norms[-1](x + self.drop(self.ffn(x)))
        return x * valid.unsqueeze(-1)  # Defined padded-query policy.


class TinyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Deliberately untied: separate source/target embeddings and output head.
        self.source_embed = nn.Embedding(cfg.vocab, cfg.width, padding_idx=PAD)
        self.target_embed = nn.Embedding(cfg.vocab, cfg.width, padding_idx=PAD)
        positions = torch.arange(cfg.limit, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, cfg.width, 2) * (-math.log(10000.0) / cfg.width))
        pe = torch.zeros(cfg.limit, cfg.width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * frequencies), torch.cos(positions * frequencies)
        self.register_buffer("positions", pe)
        self.encoders = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.decoders = nn.ModuleList([Block(cfg, decoder=True) for _ in range(cfg.layers)])
        self.head = nn.Linear(cfg.width, cfg.vocab)
        self.drop = nn.Dropout(cfg.dropout)

    def embed(self, ids, table):
        if ids.ndim != 2 or ids.shape[1] < 1 or ids.shape[1] > self.cfg.limit:
            raise ValueError("Expected nonempty bounded [B,T] IDs")
        if (ids < 0).any() or (ids >= self.cfg.vocab).any():
            raise ValueError("ID outside frozen vocabulary")
        return self.drop(table(ids) * math.sqrt(self.cfg.width) + self.positions[:ids.shape[1]])

    def encode(self, source):
        valid = source != PAD
        if not valid.any(-1).all():
            raise ValueError("Empty/all-PAD source")
        x = self.embed(source, self.source_embed)
        allowed = valid[:, None, None, :].expand(-1, 1, source.shape[1], -1)
        for block in self.encoders:
            x = block(x, allowed, valid)
        return x, valid

    def decode(self, prefix, memory, source_valid):
        if (prefix[:, 0] != BOS).any():
            raise ValueError("Every decoder prefix must begin with BOS")
        valid = prefix != PAD
        t = prefix.shape[1]
        causal = torch.ones(t, t, device=prefix.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        cross = source_valid[:, None, None, :].expand(-1, 1, t, -1)
        x = self.embed(prefix, self.target_embed)
        for block in self.decoders:
            x = block(x, allowed, valid, memory, cross)
        return self.head(x)

    def forward(self, source, prefix):
        memory, valid = self.encode(source)
        return self.decode(prefix, memory, valid)

    @torch.no_grad()
    def generate(self, source, max_new=16):
        if not 1 <= max_new <= self.cfg.limit:
            raise ValueError("Generation bound outside configured context")
        was_training = self.training
        self.eval()
        try:
            memory, valid = self.encode(source)
            prefix = torch.full((source.shape[0], 1), BOS, dtype=torch.long, device=source.device)
            done = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
            for _ in range(max_new):
                logits = self.decode(prefix, memory, valid)[:, -1].clone()
                logits[:, [PAD, BOS, SEP]] = -torch.inf
                next_id = logits.argmax(-1)
                next_id = torch.where(done, EOS, next_id)
                prefix = torch.cat([prefix, next_id[:, None]], dim=1)
                done |= next_id == EOS
                if done.all():
                    break
            # All request-local tensors are recreated on every call; no cache.
            return prefix[:, 1:]
        finally:
            self.train(was_training)


def records(seed=73, n=384):
    rng, seen, data = random.Random(seed), set(), []
    while len(data) < n:
        text = "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 6)))
        if text not in seen:
            seen.add(text)
            data.append(text)
    return data[:256], data[256:]  # Deduplicate BEFORE tokenizer and batching.


def batch(texts, tokenizer, device="cpu", reverse=False):
    if not texts or any(not text for text in texts):
        raise ValueError("Nonempty records required")
    src = [tokenizer.encode(text) + [EOS] for text in texts]
    targets = [tokenizer.encode(text[::-1] if reverse else text) for text in texts]
    inputs = [[BOS] + ids for ids in targets]
    labels = [ids + [EOS] for ids in targets]

    def pad(rows):
        return torch.tensor([row + [PAD] * (max(map(len, rows)) - len(row)) for row in rows],
                            device=device, dtype=torch.long)
    return pad(src), pad(inputs), pad(labels)


def objective(logits, labels):
    if not (labels != PAD).any():
        raise ValueError("No supervised target")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=PAD)


class Sampler:
    def __init__(self, size, seed=91):
        self.rng, self.order, self.cursor = random.Random(seed), list(range(size)), 0
        self.rng.shuffle(self.order)

    def next(self, n):
        result = []
        for _ in range(n):
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            result.append(self.order[self.cursor])
            self.cursor += 1
        return result

    def state(self):
        return {"rng": self.rng.getstate(), "order": self.order[:], "cursor": self.cursor}

    def restore(self, state):
        self.rng.setstate(state["rng"])
        self.order, self.cursor = state["order"], state["cursor"]


class Trainer:
    def __init__(self, cfg=Config(), seed=17, device="cpu", reverse=False):
        self.device, self.reverse = device, reverse
        random.seed(seed)
        torch.manual_seed(seed)
        self.model = TinyTransformer(cfg).to(device)
        self.tokenizer = ByteBPE()
        if cfg.vocab != self.tokenizer.size:
            raise ValueError("This copy reference uses byte baseline V260")
        self.train, self.valid = records()
        self.sampler = Sampler(len(self.train))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min((step + 1) / 20, 1.0))
        self.step, self.best, self.last_grad_norm = 0, None, None
        source_hash = hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
                                           for name in ["tiny_transformer.py", "byte_bpe.py"])).hexdigest()
        self.identity = {"schema": 1, "config": asdict(cfg), "seed": seed,
                         "tokenizer": self.tokenizer.identity, "split": digest([self.train, self.valid]),
                         "source_sha256": source_hash, "torch": torch.__version__,
                         "python": platform.python_version(), "device": device,
                         "precision": "fp32", "task": "reverse" if reverse else "copy",
                         "optimizer": "Adam-lr0.002-warmup20", "batch": 16,
                         "deterministic": torch.are_deterministic_algorithms_enabled(),
                         "threads": torch.get_num_threads()}

    def update(self):
        self.model.train()
        indices = self.sampler.next(16)
        source, prefix, labels = batch([self.train[i] for i in indices], self.tokenizer,
                                       self.device, self.reverse)
        self.optimizer.zero_grad(set_to_none=True)
        loss = objective(self.model(source, prefix), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.last_grad_norm = float(norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        # No pending gradients at the save boundary.
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    def state(self):
        return {"identity": self.identity, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "step": self.step, "best": self.best, "metric_position": self.step,
                "sampler": self.sampler.state(), "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.startswith("cuda") else None,
                "precision": {"dtype": "fp32", "scaler": None},
                "accumulation": {"pending": 0}, "last_grad_norm": self.last_grad_norm}

    def restore(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Incompatible run identity; use an explicit new run")
        if state["accumulation"]["pending"] != 0:
            raise ValueError("Mid-window checkpoints unsupported")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for values in self.optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(self.device)
        self.scheduler.load_state_dict(state["scheduler"])
        self.step, self.best = state["step"], state["best"]
        self.last_grad_norm = state["last_grad_norm"]
        self.sampler.restore(state["sampler"])
        # Restore RNG last, before the next batch/dropout draw.
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])

    @torch.no_grad()
    def evaluate(self, shuffled=False):
        self.model.eval()
        loss_sum, count, teacher_correct = 0.0, 0, 0
        exact, gen_correct, gen_total, eos_count = 0, 0, 0, 0
        for start in range(0, len(self.valid), 16):
            texts = self.valid[start:start + 16]
            src, inp, labels = batch(texts, self.tokenizer, self.device, self.reverse)
            if shuffled:
                src = src.roll(1, 0)
            logits = self.model(src, inp)
            mask = labels != PAD
            count += int(mask.sum())
            loss_sum += float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                             ignore_index=PAD, reduction="sum"))
            teacher_correct += int(((logits.argmax(-1) == labels) & mask).sum())
            outputs = self.model.generate(src, max_new=8).tolist()
            for row, text in zip(outputs, texts):
                expected = self.tokenizer.encode(text[::-1] if self.reverse else text)
                has_eos = EOS in row
                eos_count += int(has_eos)
                actual = row[:row.index(EOS)] if has_eos else row
                exact += int(has_eos and actual == expected)
                gen_correct += sum(i < len(actual) and token == actual[i]
                                   for i, token in enumerate(expected))
                gen_total += len(expected)  # Extra tokens penalize exact match, not this denominator.
        return {"records": len(self.valid), "target_tokens_including_eos": count,
                "loss": loss_sum / count, "teacher_token_accuracy": teacher_correct / count,
                "generated_reference_token_accuracy": gen_correct / gen_total,
                "generated_exact_match": exact / len(self.valid), "eos_rate": eos_count / len(self.valid)}


def publish(trainer, root):
    """Completion last; immutable generations; verify bytes before acceptance.

    Does not promise mounted-Drive atomicity or survival after runtime deletion.
    Retains all generations here; child implementation must add bounded retention.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation = root / f"step-{trainer.step:09d}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    generation.mkdir()
    with tempfile.TemporaryDirectory() as local:
        path = Path(local) / "state.pt"
        torch.save(trainer.state(), path)
        raw = path.read_bytes()
        (generation / "state.pt").write_bytes(raw)
    manifest = {"schema": 1, "step": trainer.step, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    (generation / "manifest.json").write_bytes(canonical(manifest))
    verify_generation(generation, complete=False)
    (generation / "COMPLETE").write_text("verified\n")
    return generation


def verify_generation(path, complete=True):
    if complete and not (path / "COMPLETE").is_file():
        raise ValueError("Incomplete generation")
    manifest = json.loads((path / "manifest.json").read_text())
    raw = (path / "state.pt").read_bytes()
    if len(raw) != manifest["bytes"] or hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint bytes fail checksum/size")
    # Only use this for locally produced/trusted educational checkpoint files.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if manifest["schema"] != 1 or state["step"] != manifest["step"]:
        raise ValueError("Manifest/state identity mismatch")
    return state


def recover(trainer, root):
    rejected = []
    for generation in sorted(Path(root).glob("step-*"), reverse=True):
        try:
            state = verify_generation(generation)
            trainer.restore(state)
            return {"selected_step": trainer.step, "generation": generation.name,
                    "rejected_newer": rejected, "rollback_report_required": bool(rejected)}
        except (ValueError, OSError, KeyError, EOFError, RuntimeError, json.JSONDecodeError) as error:
            rejected.append({"generation": generation.name, "reason": str(error)})
    raise ValueError(f"No compatible verified checkpoint; rejected={rejected}")


def check():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(5)
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    norm = LayerNorm(4).double()
    torch.testing.assert_close(norm(x), F.layer_norm(x, (4,), norm.scale, norm.bias, norm.eps))
    assert torch.autograd.gradcheck(norm, (x,))
    q, k, v = [torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    mask = torch.ones(3, 3, dtype=torch.bool).tril()[None, None]
    manual = ((q @ k.transpose(-2, -1)) / 2).masked_fill(~mask, -torch.inf).softmax(-1) @ v
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0)
    torch.testing.assert_close(manual, reference, atol=1e-10, rtol=1e-10)
    g1, g2 = torch.autograd.grad(manual.sum(), (q, k, v), retain_graph=True), torch.autograd.grad(reference.sum(), (q, k, v))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    trainer = Trainer()
    src, inp, labels = batch(["123", "45"], trainer.tokenizer)
    model = trainer.model.eval()
    logits = model(src, inp)
    changed = inp.clone()
    changed[:, 2:] = 57
    torch.testing.assert_close(logits[:, :2], model(src, changed)[:, :2], atol=1e-6, rtol=1e-6)
    padded = F.pad(src, (0, 2), value=PAD)
    torch.testing.assert_close(logits, model(padded, inp), atol=1e-6, rtol=1e-6)
    padded_inp = F.pad(inp, (0, 2), value=PAD)
    padded_labels = F.pad(labels, (0, 2), value=PAD)
    torch.testing.assert_close(objective(logits, labels), objective(model(src, padded_inp), padded_labels))
    objective(logits, labels).backward()
    assert model.source_embed.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    changed_src = src.clone()
    changed_src[0, 0] = 57
    assert not torch.allclose(logits[0], model(changed_src, inp)[0])
    model.head.bias.data[EOS] = 1000
    assert (model.generate(src) == EOS).all()
    first = model.generate(src[:1]).clone()
    model.generate(src[1:])
    assert torch.equal(first, model.generate(src[:1]))
    try:
        model.encode(torch.full((1, 2), PAD))
        raise AssertionError("Accepted all-PAD source")
    except ValueError:
        pass
    # Nonzero dropout makes restored RNG consequential.
    cfg = Config(dropout=0.1)
    full = Trainer(cfg)
    full_losses = [full.update() for _ in range(8)]
    split = Trainer(cfg)
    split_losses = [split.update() for _ in range(3)]
    with tempfile.TemporaryDirectory() as temp:
        good = publish(split, temp)
        fresh = Trainer(cfg)
        recovery = recover(fresh, temp)
        split_losses += [fresh.update() for _ in range(5)]
        assert full_losses == split_losses
        for a, b in zip(full.model.parameters(), fresh.model.parameters()):
            assert torch.equal(a, b)
        assert full.sampler.state() == fresh.sampler.state()
        assert full.scheduler.state_dict() == fresh.scheduler.state_dict()
        for a, b in zip(full.optimizer.state.values(), fresh.optimizer.state.values()):
            for key in a:
                assert torch.equal(a[key], b[key]) if isinstance(a[key], torch.Tensor) else a[key] == b[key]
        bad = publish(fresh, temp)
        (bad / "state.pt").write_bytes(b"corrupt")
        fallback = recover(Trainer(cfg), temp)
        assert fallback["selected_step"] == 3 and fallback["rejected_newer"]
        incomplete = Path(temp) / "step-999999999-incomplete"
        incomplete.mkdir()
        fallback = recover(Trainer(cfg), temp)
        assert len(fallback["rejected_newer"]) == 2
        mismatch = Trainer(Config(width=16, heads=4, hidden=64, dropout=0.1))
        try:
            mismatch.restore(verify_generation(good))
            raise AssertionError("Accepted wrong config")
        except ValueError:
            pass
    print(json.dumps({"layernorm_gradcheck": True, "attention_forward_backward_parity": True,
                      "causality_padding_gradients_generation": True, "resume_bitwise_cpu": True,
                      "corrupt_incomplete_fallback": True, "identity_rejection": True,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "runtime": torch.__version__}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=20, help="Total target optimizer updates, including restored ones")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.steps < 1 or args.save_every < 1 or not args.run_dir:
        parser.error("Positive bounds and explicit --run-dir are required")
    if args.resume and not args.run_dir.exists():
        parser.error("Resume run directory does not exist")
    if not args.resume and args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("Nonempty run directory: select --resume or a new directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    trainer = Trainer(device=args.device, reverse=args.reverse)
    recovery = recover(trainer, args.run_dir) if args.resume else None
    started, losses = time.perf_counter(), []
    while trainer.step < args.steps:
        losses.append(trainer.update())
        if trainer.step % args.save_every == 0:
            publish(trainer, args.run_dir)
    publish(trainer, args.run_dir)
    report = {"step": trainer.step, "parameters": sum(p.numel() for p in trainer.model.parameters()),
              "training_seconds": time.perf_counter() - started,
              "initial_loss_this_session": losses[0] if losses else None,
              "final_loss_this_session": losses[-1] if losses else None,
              "last_grad_norm": trainer.last_grad_norm, "identity": trainer.identity,
              "recovery": recovery, "held_out": trainer.evaluate(),
              "shuffled_source_control": trainer.evaluate(shuffled=True)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
