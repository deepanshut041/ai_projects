"""Run bounded masked-language and extractive-question-answering experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from roberta_data import ByteBPETokenizer, mlm_batch
from roberta_model import RobertaConfig, RobertaForMaskedLM, optimizer_groups, warmup_linear_factor


TRAIN_TEXTS = [
    "Paris is the capital of France. France is a country in Europe.",
    "Rome is the capital of Italy. Italy is a country in Europe.",
    "Berlin is the capital of Germany. Germany is a country in Europe.",
    "Tokyo is the capital of Japan. Japan is a country in Asia.",
    "The red apple is on the table. The green apple is in the bag.",
    "A bird can fly. A fish can swim. A cat can climb.",
    "The library opens on Monday. The museum opens on Tuesday.",
    "A café serves coffee. The baker makes bread every morning.",
]
VALIDATION_TEXTS = [
    "Europe contains France and Italy.",
    "The table holds a red apple.",
]
QA_EXAMPLES = [
    ("What is the capital of France?", TRAIN_TEXTS[0], "Paris"),
    ("What is the capital of Italy?", TRAIN_TEXTS[1], "Rome"),
    ("What is the capital of Germany?", TRAIN_TEXTS[2], "Berlin"),
    ("What is the capital of Japan?", TRAIN_TEXTS[3], "Tokyo"),
    ("Where is the red apple?", TRAIN_TEXTS[4], "the table"),
    ("Which animal can swim?", TRAIN_TEXTS[5], "fish"),
    ("When does the library open?", TRAIN_TEXTS[6], "Monday"),
    ("What does a café serve?", TRAIN_TEXTS[7], "coffee"),
]


@dataclass
class TrainingBundle:
    task: str
    model: torch.nn.Module
    tokenizer: ByteBPETokenizer
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    mask_generator: torch.Generator
    batch_generator: torch.Generator
    data: list
    protocol: dict
    step: int = 0
    batch_size: int = 4
    accumulation_steps: int = 1

    def identity(self):
        corpus = json.dumps({"train": TRAIN_TEXTS, "validation": VALIDATION_TEXTS, "qa": QA_EXAMPLES}, ensure_ascii=False, sort_keys=True)
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("roberta_data.py", "roberta_model.py", "roberta_train.py")}
        return {"task": self.task, "config": self.model.config.to_dict(), "tokenizer_sha256": self.tokenizer.fingerprint(), "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(), "batch_size": self.batch_size, "accumulation_steps": self.accumulation_steps, "protocol": self.protocol, "source_sha256": sources}


def build_training_bundle(task="mlm", seed=7, total_steps=120, pretrained=None, device="cpu"):
    torch.manual_seed(seed)
    tokenizer = ByteBPETokenizer.train(TRAIN_TEXTS + [question for question, _, _ in QA_EXAMPLES], merge_count=80)
    config = RobertaConfig(vocab_size=tokenizer.vocab_size)
    if task == "mlm":
        model = RobertaForMaskedLM(config)
        data = [tokenizer.encode(text) for text in TRAIN_TEXTS]
        if any(len(tokens) > config.context_length for tokens in data):
            raise ValueError("A fixture is longer than the configured context")
    elif task == "qa":
        from roberta_data import qa_features
        from roberta_model import RobertaForQuestionAnswering

        model = RobertaForQuestionAnswering(config)
        if pretrained is not None:
            model.load_mlm_backbone(pretrained)
        data = []
        for question, context, answer in QA_EXAMPLES:
            features = qa_features(question, context, tokenizer, context_length=config.context_length, overlap=8, answer_start=context.index(answer), answer_text=answer)
            data.extend(feature for feature in features if feature.start_position != -100)
        if len(data) < len(QA_EXAMPLES):
            raise ValueError("Every training answer must fit completely in at least one window")
    else:
        raise ValueError("Choose mlm or qa")
    model = model.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model), lr=3e-3, betas=(0.9, 0.98), eps=1e-6)
    warmup = min(10, total_steps - 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_linear_factor(step, total_steps, warmup))
    protocol = {"seed": seed, "total_steps": total_steps, "warmup_steps": warmup, "masking_probability": 0.35, "sampling": "independent-with-replacement", "gradient_norm_limit": 1.0, "optimizer": {"name": "AdamW", "learning_rate": 3e-3, "betas": [0.9, 0.98], "epsilon": 1e-6, "matrix_weight_decay": 0.01}, "scheduler": "warmup-linear-v1"}
    return TrainingBundle(task, model, tokenizer, optimizer, scheduler, torch.Generator().manual_seed(seed + 1), torch.Generator().manual_seed(seed + 2), data, protocol)


def next_batch(bundle):
    indices = torch.randint(len(bundle.data), (bundle.batch_size,), generator=bundle.batch_generator).tolist()
    samples = [bundle.data[index] for index in indices]
    if bundle.task == "mlm":
        return mlm_batch(samples, bundle.tokenizer, bundle.mask_generator, probability=bundle.protocol["masking_probability"])
    from roberta_data import qa_batch

    return qa_batch(samples)


def to_model_device(batch, model):
    device = next(model.parameters()).device
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def training_step(bundle, accumulation_steps=None):
    if accumulation_steps is None:
        accumulation_steps = bundle.accumulation_steps
    if accumulation_steps < 1:
        raise ValueError("An update requires at least one microbatch")
    bundle.accumulation_steps = accumulation_steps
    microbatches = [next_batch(bundle) for _ in range(accumulation_steps)]
    counts = [int(batch["labels"].ne(-100).sum()) if bundle.task == "mlm" else int(batch["start_positions"].ne(-100).sum()) for batch in microbatches]
    total_count = sum(counts)
    if total_count == 0:
        return {"updated": False, "step": bundle.step, "target_count": 0, "loss": 0.0}
    bundle.model.train()
    bundle.optimizer.zero_grad(set_to_none=True)
    total_sum = 0.0
    for batch in microbatches:
        result = bundle.model(**to_model_device(batch, bundle.model))
        (result["loss_sum"] / total_count).backward()
        total_sum += float(result["loss_sum"].detach())
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), bundle.protocol["gradient_norm_limit"]))
    bundle.optimizer.step()
    bundle.scheduler.step()
    bundle.step += 1
    return {"updated": True, "step": bundle.step, "target_count": total_count, "loss": total_sum / total_count, "gradient_norm_before_clip": gradient_norm}


def fixed_mlm_evaluation(bundle):
    sequences = [bundle.tokenizer.encode(text) for text in VALIDATION_TEXTS]
    batch = mlm_batch(sequences, bundle.tokenizer, torch.Generator().manual_seed(900), probability=0.35)
    batch = to_model_device(batch, bundle.model)
    bundle.model.eval()
    with torch.no_grad():
        result = bundle.model(**batch)
    selected = batch["labels"].ne(-100)
    correct = int((result["logits"].argmax(-1)[selected] == batch["labels"][selected]).sum())
    return {"loss": float(result["loss"]), "correct": correct, "target_count": int(selected.sum())}


def fill_mask(model, tokenizer, text, top_k=5):
    """Rank ordinary token replacements for exactly one explicit corruption marker."""
    if text.count("<mask>") != 1 or top_k < 1:
        raise ValueError("Provide exactly one <mask> marker and a positive top_k")
    left, right = text.split("<mask>")
    left_ids = tokenizer.encode(left, add_special_tokens=False)
    right_ids = tokenizer.encode(right, add_special_tokens=False)
    ids = [tokenizer.bos_id, *left_ids, tokenizer.mask_id, *right_ids, tokenizer.eos_id]
    if len(ids) > model.config.context_length:
        raise ValueError("The fill-mask input is longer than the model context")
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        logits = model(torch.tensor([ids], device=device))["logits"][0, len(left_ids) + 1]
        probabilities = logits.softmax(-1)
        ordinary = probabilities[5:]
        values, indices = ordinary.topk(min(top_k, len(ordinary)))
    return [{"token_id": int(index + 5), "token_text": tokenizer.decode([int(index + 5)]), "probability": float(value)} for value, index in zip(values, indices)]


def predict_qa(model, tokenizer, question, context, overlap=8):
    from roberta_data import best_span, qa_batch, qa_features

    features = qa_features(question, context, tokenizer, context_length=model.config.context_length, overlap=overlap)
    model.eval()
    best = None
    with torch.no_grad():
        for feature in features:
            batch = to_model_device(qa_batch([feature]), model)
            result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            candidate = best_span(result["start_logits"][0], result["end_logits"][0], feature.offsets, context)
            if candidate is not None and (best is None or candidate["score"] > best["score"]):
                best = candidate
    return best


def export_backbone(bundle, destination):
    if bundle.task != "mlm":
        raise ValueError("Export the masked-language backbone before QA fine tuning")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "course-roberta-backbone-v1", "config": bundle.model.config.to_dict(), "tokenizer": bundle.tokenizer.to_dict(), "backbone": bundle.model.roberta.state_dict()}, destination)


def load_backbone_for_qa(bundle, source):
    state = torch.load(source, map_location="cpu", weights_only=True)
    if state.get("format") != "course-roberta-backbone-v1" or state["config"] != bundle.model.config.to_dict():
        raise ValueError("Backbone format or model configuration does not match")
    tokenizer = ByteBPETokenizer.from_dict(state["tokenizer"])
    if tokenizer.fingerprint() != bundle.tokenizer.fingerprint():
        raise ValueError("Tokenizer identity differs; token IDs would address the wrong embeddings")
    bundle.model.roberta.load_state_dict(state["backbone"], strict=True)


def run_demo(task="both", steps=120, export_path=None, backbone_path=None, device="cpu"):
    torch.set_num_threads(1)
    mlm = None
    reports = {}
    if task in ("mlm", "both"):
        mlm = build_training_bundle("mlm", total_steps=steps, device=device)
        before = fixed_mlm_evaluation(mlm)
        train_reports = [training_step(mlm) for _ in range(steps)]
        after = fixed_mlm_evaluation(mlm)
        reports["mlm"] = {"before": before, "after": after, "last_update": train_reports[-1], "parameters": sum(parameter.numel() for parameter in mlm.model.parameters()), "vocabulary_size": mlm.tokenizer.vocab_size, "one_token_fill_candidates": fill_mask(mlm.model, mlm.tokenizer, "Paris is the capital of <mask>.")}
        if export_path:
            export_backbone(mlm, export_path)
    if task in ("qa", "both"):
        from roberta_data import answer_metrics, qa_batch

        qa = build_training_bundle("qa", total_steps=steps, pretrained=mlm.model if mlm is not None else None, device=device)
        if backbone_path:
            load_backbone_for_qa(qa, backbone_path)
        features = to_model_device(qa_batch(qa.data), qa.model)
        qa.model.eval()
        with torch.no_grad():
            before = float(qa.model(**features)["loss"])
        for _ in range(steps):
            training_step(qa)
        qa.model.eval()
        with torch.no_grad():
            after = float(qa.model(**features)["loss"])
        predictions = []
        for question, context, answer in QA_EXAMPLES:
            prediction = predict_qa(qa.model, qa.tokenizer, question, context)
            predictions.append({"question": question, "prediction": prediction["text"], "gold": answer, **answer_metrics(prediction["text"], [answer])})
        reports["qa"] = {"training_fixture_loss_before": before, "training_fixture_loss_after": after, "seen_fixture_predictions": predictions}
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("mlm", "qa", "both"), default="both")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--export-backbone")
    parser.add_argument("--backbone")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("Use at least two optimizer steps")
    print(json.dumps(run_demo(args.task, args.steps, args.export_backbone, args.backbone, args.device), indent=2))
