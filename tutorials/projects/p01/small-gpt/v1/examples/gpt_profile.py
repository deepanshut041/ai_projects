"""Small reproducible CPU forward/backward timing; not a throughput leaderboard."""
import json
import platform
import statistics
import time
import torch
from gpt_config import Config
from gpt_model import GPT, parameter_formula
from gpt_objective import nll_sum


def profile(context=64, repeats=10):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = Config(context=context)
    model = GPT(config)
    x = torch.randint(0, 256, (8, context))
    y = torch.randint(0, 256, (8, context))
    def step():
        model.zero_grad(set_to_none=True)
        loss, count = nll_sum(model(x), y)
        (loss/count).backward()
    for _ in range(3):
        step()
    times = []
    for _ in range(repeats):
        begin = time.perf_counter()
        step()
        times.append(time.perf_counter()-begin)
    count = sum(p.numel() for p in model.parameters())
    assert count == parameter_formula(config)
    return {"context": context, "parameters": count, "weight_bytes_fp32": count*4,
            "weights_grad_adam_bytes_estimate": count*16,
            "one_attention_score_bytes": 8*config.heads*context*context*4,
            "median_forward_backward_seconds": statistics.median(times),
            "min_seconds": min(times), "max_seconds": max(times),
            "torch": str(torch.__version__), "python": platform.python_version(),
            "threads": 1, "batch": 8, "repeats": repeats,
            "scope": "CPU synthetic forward/backward; excludes optimizer/data/save; bytes exclude activations/temporaries"}


if __name__ == "__main__":
    print(json.dumps([profile(32), profile(64)], indent=2))
