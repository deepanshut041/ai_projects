"""A fixed-batch fitting diagnostic, deliberately separate from corpus training."""
import json
import torch
from gpt_config import Config
from gpt_data import windows
from gpt_model import GPT
from gpt_objective import nll_sum
from gpt_tokenizer import ByteBPE


def fit(updates=300):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = GPT(Config(dropout=0.0))
    x, y = windows(["red cat", "blue dog"], ByteBPE(), 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    def loss():
        total, count = nll_sum(model(x), y)
        return total/count
    before = loss().item()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    after = loss().item()
    return {"updates": updates, "before_nll": before, "after_nll": after,
            "fitting_target": 0.1, "passed": after < 0.1,
            "scope": "two memorized strings, no held-out claim"}


if __name__ == "__main__":
    print(json.dumps(fit(), indent=2))
