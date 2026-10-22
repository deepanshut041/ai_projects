"""Inspect batches, stable probabilities and small neural modules."""

import argparse
import math

import torch
from torch import nn


def collate_token_records(records, *, pad_id=0, ignore_index=-100):
    """Keep input visibility separate from per-position supervision."""
    if not records or pad_id < 0 or ignore_index >= 0:
        raise ValueError("use a nonempty batch, nonnegative pad ID and negative ignored label")
    for record in records:
        ids, labels = record["input_ids"], record["labels"]
        if not ids or len(ids) != len(labels):
            raise ValueError("every record needs aligned nonempty inputs and labels")
        if any(type(value) is not int or value < 0 or value == pad_id for value in ids):
            raise ValueError("real input IDs must be nonnegative integers other than padding")
        if any(type(value) is not int or (value < 0 and value != ignore_index) for value in labels):
            raise ValueError("labels must be class IDs or the ignored-label sentinel")
    width = max(len(record["input_ids"]) for record in records)
    input_ids = torch.full((len(records), width), pad_id, dtype=torch.long)
    labels = torch.full((len(records), width), ignore_index, dtype=torch.long)
    valid = torch.zeros((len(records), width), dtype=torch.bool)
    for row, record in enumerate(records):
        length = len(record["input_ids"])
        input_ids[row, :length] = torch.tensor(record["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(record["labels"], dtype=torch.long)
        valid[row, :length] = True
    return {"input_ids": input_ids, "valid": valid, "labels": labels}


def valid_loss_totals(losses, labels, *, ignore_index=-100):
    if losses.shape != labels.shape or not losses.is_floating_point():
        raise ValueError("floating losses and labels must have the same shape")
    selected = labels != ignore_index
    count = int(selected.sum())
    if not count or not torch.isfinite(losses[selected]).all():
        raise ValueError("need at least one finite supervised loss")
    loss_sum = torch.where(selected, losses, torch.zeros_like(losses)).sum()
    return loss_sum, count


def token_mean(losses, labels, *, ignore_index=-100):
    loss_sum, count = valid_loss_totals(losses, labels, ignore_index=ignore_index)
    return loss_sum / count


def stable_softmax(logits, allowed=None, *, dim=-1):
    if not logits.is_floating_point() or logits.shape[dim] == 0:
        raise ValueError("softmax needs floating scores and a nonempty normalization axis")
    if allowed is None:
        allowed = torch.ones_like(logits, dtype=torch.bool)
    if allowed.dtype != torch.bool:
        raise ValueError("allowed must be Boolean")
    allowed = torch.broadcast_to(allowed, logits.shape)
    if not torch.isfinite(logits[allowed]).all():
        raise ValueError("allowed scores must be finite")
    masked = logits.masked_fill(~allowed, -torch.inf)
    nonempty = allowed.any(dim=dim, keepdim=True)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(nonempty, maximum, torch.zeros_like(maximum))
    numerator = (masked - maximum).exp()
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator.clamp_min(1.0)


def stable_logsumexp(logits, *, dim=-1):
    if not logits.is_floating_point() or logits.shape[dim] == 0 or not torch.isfinite(logits).all():
        raise ValueError("logsumexp needs nonempty finite floating scores")
    maximum = logits.amax(dim=dim, keepdim=True)
    shifted_sum = (logits - maximum).exp().sum(dim=dim)
    return maximum.squeeze(dim) + shifted_sum.log()


def categorical_nll(logits, targets, *, ignore_index=-100, reduction="mean"):
    if not logits.is_floating_point() or logits.shape[-1] == 0 or not torch.isfinite(logits).all():
        raise ValueError("NLL needs nonempty finite floating scores")
    if targets.dtype != torch.long or targets.shape != logits.shape[:-1]:
        raise ValueError("targets must be long class IDs matching every logit row")
    selected = targets != ignore_index
    if ((targets[selected] < 0) | (targets[selected] >= logits.shape[-1])).any():
        raise ValueError("supervised label is outside the class vocabulary")
    safe_targets = torch.where(selected, targets, torch.zeros_like(targets))
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    chosen = shifted.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    losses = shifted.exp().sum(dim=-1).log() - chosen
    losses = torch.where(selected, losses, torch.zeros_like(losses))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return token_mean(losses, targets, ignore_index=ignore_index)
    raise ValueError("reduction must be none, sum or mean")


def weighted_information(weights, values):
    if weights.ndim != 2 or values.ndim != 2 or weights.shape[1] != values.shape[0]:
        raise ValueError("weights [queries,sources] must match values [sources,features]")
    if not weights.is_floating_point() or not values.is_floating_point():
        raise ValueError("probabilities and payload features must be floating-point tensors")
    if not torch.isfinite(weights).all() or not torch.isfinite(values).all() or (weights < 0).any():
        raise ValueError("use finite payloads and nonnegative finite weights")
    sums = weights.sum(dim=-1)
    if not ((sums == 0) | torch.isclose(sums, torch.ones_like(sums))).all():
        raise ValueError("each weight row must sum to one or be wholly blocked")
    return weights @ values


def feature_statistics(x):
    if not x.is_floating_point() or x.shape[-1] == 0:
        raise ValueError("feature statistics need a nonempty floating feature axis")
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return mean, variance


def manual_layer_norm(x, scale=None, bias=None, *, eps=1e-5):
    if eps <= 0:
        raise ValueError("eps must be positive")
    mean, variance = feature_statistics(x)
    normalized = (x - mean) / (variance + eps).sqrt()
    if scale is not None:
        normalized = normalized * scale
    if bias is not None:
        normalized = normalized + bias
    return normalized


def standard_normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class FeatureMLP(nn.Module):
    def __init__(self, input_features=2, hidden_features=8, num_classes=2):
        super().__init__()
        self.register_buffer("input_scale", torch.ones(input_features))
        self.hidden = nn.Linear(input_features, hidden_features)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_features, num_classes)

    def forward(self, features):
        if not features.is_floating_point() or features.shape[-1] != self.hidden.in_features:
            raise ValueError("features must be floating vectors with the declared input width")
        scaled = features / self.input_scale
        return self.output(self.activation(self.hidden(scaled)))


def data_demo():
    records = [{"input_ids": [4, 9, 2], "labels": [1, 0, 1]},
               {"input_ids": [8], "labels": [0]}]
    batch = collate_token_records(records)
    for name, tensor in batch.items():
        print(name, tensor.tolist())
    losses = torch.tensor([[1., 1., 1.], [3., 100., 100.]])
    print("supervised_token_mean", float(token_mean(losses, batch["labels"])))


def probability_demo():
    logits = torch.tensor([[0., math.log(3)], [1000., 1000. + math.log(3)]], dtype=torch.float64)
    print("probabilities", stable_softmax(logits).tolist())
    print("loss_for_class_1", categorical_nll(logits, torch.tensor([1, 1])).item())
    weights = torch.tensor([[.25, .75], [.6, .4]], dtype=torch.float64)
    values = torch.tensor([[2., 0.], [0., 4.]], dtype=torch.float64)
    print("weighted_information", weighted_information(weights, values).tolist())


def module_demo():
    torch.manual_seed(7)
    model = FeatureMLP()
    output = model(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
    print("MLP_shape", tuple(output.shape), "parameters", sum(p.numel() for p in model.parameters()))
    print("buffers", list(dict(model.named_buffers())))
    print("layer_norm", manual_layer_norm(torch.tensor([[1., 3.]], dtype=torch.float64)).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("data", "probability", "modules", "all"), default="all")
    args = parser.parse_args()
    for name, function in (("data", data_demo), ("probability", probability_demo), ("modules", module_demo)):
        if args.section in (name, "all"):
            function()


if __name__ == "__main__":
    main()
