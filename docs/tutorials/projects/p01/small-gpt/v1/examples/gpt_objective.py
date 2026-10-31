"""Loss sums and denominators stay visible for evaluation and accumulation."""
import math
import torch
from torch.nn import functional as F
from gpt_tokenizer import PAD


def nll_sum(logits, labels):
    count = int((labels != PAD).sum())
    if count == 0:
        raise ValueError("No supervised tokens")
    total = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1), ignore_index=PAD, reduction="sum")
    return total, count


def lr_at(update, total, warmup, peak, floor):
    if not 1 <= update <= total or not 1 <= warmup < total or not 0 <= floor <= peak:
        raise ValueError("Invalid fixed learning-rate schedule")
    if update <= warmup:
        return peak * update / warmup
    fraction = (update-warmup) / (total-warmup)
    return floor + 0.5*(peak-floor)*(1+math.cos(math.pi*fraction))


def optimizer_for(model, lr, decay):
    matrices, vectors = [], []
    for parameter in model.parameters():
        (matrices if parameter.ndim >= 2 else vectors).append(parameter)
    return torch.optim.AdamW([{"params": matrices, "weight_decay": decay},
                              {"params": vectors, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8, foreach=False)


@torch.no_grad()
def evaluate(model, x, y, batch=16, device="cpu"):
    training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for start in range(0, len(x), batch):
            loss, n = nll_sum(model(x[start:start+batch].to(device)), y[start:start+batch].to(device))
            total += loss.item()
            count += n
    finally:
        model.train(training)
    mean = total/count
    return {"nll": mean, "perplexity": math.exp(mean), "tokens": count}


def unigram_baseline(train_y, valid_y, vocab):
    counts = torch.bincount(train_y[train_y != PAD], minlength=vocab).double() + 1
    log_probs = (counts/counts.sum()).log()
    labels = valid_y[valid_y != PAD]
    return float(-log_probs[labels].mean())
