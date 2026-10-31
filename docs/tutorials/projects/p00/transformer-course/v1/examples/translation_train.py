"""Bounded token-weighted training for an offline translation fixture."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from translation_data import TRAIN_PAIRS, VALIDATION_PAIRS, collate_translation, encode_pairs, make_fixture
from translation_metrics import corpus_bleu_tokens
from translation_model import Transformer, TransformerConfig


def token_statistics(logits, labels):
    count = (labels != -100).sum()
    if not count:
        raise ValueError("loss requires at least one supervised target")
    loss_sum = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
    correct = ((logits.argmax(dim=-1) == labels) & (labels != -100)).sum()
    return loss_sum, count, correct


def logits_for(model, batch):
    return model(batch["src_input_ids"], batch["tgt_input_ids"],
                 batch["src_pad_mask"], batch["tgt_pad_mask"])


def train_update(model, optimizer, batches, *, scheduler=None, max_grad_norm=1.0):
    """One optimizer update; each nonpadding label has equal total weight."""
    model.train()
    denominator = sum(int((b["tgt_outputs"] != -100).sum()) for b in batches)
    if denominator == 0:
        raise ValueError("update needs at least one supervised token")
    optimizer.zero_grad(set_to_none=True)
    detached_loss, correct = 0.0, 0
    for batch in batches:
        loss_sum, _, batch_correct = token_statistics(logits_for(model, batch), batch["tgt_outputs"])
        (loss_sum / denominator).backward()
        detached_loss += float(loss_sum.detach())
        correct += int(batch_correct)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite gradients; optimizer was not updated")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {"loss": detached_loss / denominator, "token_accuracy": correct / denominator,
            "tokens": denominator, "gradient_norm": float(grad_norm)}


@torch.no_grad()
def evaluate(model, batches):
    was_training = model.training
    model.eval()
    try:
        total_loss, total_tokens, total_correct = 0.0, 0, 0
        for batch in batches:
            loss, tokens, correct = token_statistics(logits_for(model, batch), batch["tgt_outputs"])
            total_loss += float(loss)
            total_tokens += int(tokens)
            total_correct += int(correct)
        if not total_tokens:
            raise ValueError("evaluation needs supervised tokens")
        return {"loss": total_loss / total_tokens, "token_accuracy": total_correct / total_tokens,
                "tokens": total_tokens}
    finally:
        model.train(was_training)


def adamw_groups(model, decay=0.001):
    decay_parameters, plain_parameters = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay_parameters if parameter.ndim >= 2 else plain_parameters).append(parameter)
    return [{"params": decay_parameters, "weight_decay": decay},
            {"params": plain_parameters, "weight_decay": 0.0}]


def warmup_cosine(step, *, total_steps, warmup_steps):
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("choose 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class TrainState:
    model: Transformer
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    src_tokenizer: object
    tgt_tokenizer: object
    train_batch: dict
    validation_batch: dict
    identity: dict
    step: int = 0


def make_state(*, seed=7, steps=400, device="cpu"):
    if steps < 2:
        raise ValueError("the bounded training schedule needs at least two steps")
    torch.manual_seed(seed)
    src, tgt, batch = make_fixture()
    validation = collate_translation(encode_pairs(VALIDATION_PAIRS, src, tgt), src.pad_token_id, tgt.pad_token_id)
    config = TransformerConfig(src_vocab_size=src.vocab_size, tgt_vocab_size=tgt.vocab_size,
                               src_pad_id=src.pad_token_id, tgt_pad_id=tgt.pad_token_id)
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(adamw_groups(model), lr=0.01, betas=(0.9, 0.98), eps=1e-6)
    warmup = min(20, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, total_steps=steps, warmup_steps=warmup))
    batch = {name: value.to(device) for name, value in batch.items()}
    validation = {name: value.to(device) for name, value in validation.items()}
    serialized = json.dumps({"train": TRAIN_PAIRS, "validation": VALIDATION_PAIRS}, ensure_ascii=False).encode()
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ("translation_data.py", "translation_attention.py",
                            "translation_model.py", "translation_train.py")}
    protocol = {"optimizer": {"name": "AdamW", "learning_rate": 0.01,
                              "betas": [0.9, 0.98], "epsilon": 1e-6,
                              "matrix_weight_decay": 0.001, "vector_weight_decay": 0.0},
                "scheduler": {"name": "warmup-cosine-v1", "total_steps": steps,
                              "warmup_steps": warmup, "step_after_optimizer": True},
                "gradient_norm_limit": 1.0, "loss": "supervised-token-mean",
                "ignore_index": -100, "data_order": "one fixed full training batch per update"}
    identity = {"project": "translation", "config": asdict(config),
                "source_tokens": list(src.tokens), "target_tokens": list(tgt.tokens),
                "corpus_sha256": hashlib.sha256(serialized).hexdigest(),
                "seed": seed, "total_steps": steps, "warmup_steps": warmup,
                "optimizer": "AdamW", "learning_rate": 0.01,
                "data_order": "one fixed full training batch per update",
                "source_sha256": sources, "protocol": protocol}
    return TrainState(model, optimizer, scheduler, src, tgt, batch, validation, identity)


def update(state, batches=None):
    result = train_update(state.model, state.optimizer, [state.train_batch] if batches is None else batches,
                          scheduler=state.scheduler)
    state.step += 1
    return result


@torch.no_grad()
def translation_report(state, pairs):
    records = encode_pairs(pairs, state.src_tokenizer, state.tgt_tokenizer)
    batch = collate_translation(records, state.src_tokenizer.pad_token_id, state.tgt_tokenizer.pad_token_id)
    device = next(state.model.parameters()).device
    generated = state.model.inference(batch["src_input_ids"].to(device),
                                      state.tgt_tokenizer.bos_token_id, state.tgt_tokenizer.eos_token_id,
                                      max_len=12)
    predictions = [state.tgt_tokenizer.decode(row.tolist()) for row in generated]
    references = [target for _, target in pairs]
    exact = sum(p == r for p, r in zip(predictions, references)) / len(pairs)
    bleu = corpus_bleu_tokens([p.split() for p in predictions], [r.split() for r in references])
    return {"exact_match": exact, "fixture_bleu": bleu,
            "examples": [{"source": source, "reference": reference, "prediction": prediction}
                         for (source, reference), prediction in zip(pairs, predictions)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = make_state(seed=args.seed, steps=args.steps, device=args.device)
    initial = evaluate(state.model, [state.train_batch])
    for _ in range(args.steps):
        update(state)
    final = evaluate(state.model, [state.train_batch])
    validation = evaluate(state.model, [state.validation_batch])
    train_translation = translation_report(state, TRAIN_PAIRS)
    heldout_translation = translation_report(state, VALIDATION_PAIRS)
    report = {"step": state.step, "parameters": sum(p.numel() for p in state.model.parameters()),
              "initial_train": initial, "final_train": final, "validation": validation,
              "train_generation": train_translation, "heldout_generation": heldout_translation,
              "identity": state.identity, "scope": "six training pairs and two held-out combinations; not WMT quality"}
    print(f"updates={state.step} initial_loss={initial['loss']:.4f} final_loss={final['loss']:.4f}")
    print(f"train_token_accuracy={final['token_accuracy']:.3f} heldout_token_accuracy={validation['token_accuracy']:.3f}")
    print(f"train_exact={train_translation['exact_match']:.3f} heldout_exact={heldout_translation['exact_match']:.3f}")
    for sample in heldout_translation["examples"]:
        print(sample["source"], "=>", sample["prediction"], "| reference:", sample["reference"])
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
