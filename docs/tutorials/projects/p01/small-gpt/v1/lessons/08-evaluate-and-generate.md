## Lesson 8 — measure unseen text and generate a continuation

A low loss on two memorized strings proves the optimizer can fit them. It does not show that the model predicts a new chapter well. We need a held-out measure, a simple baseline and visible generated examples. Each answers a different question.

### Evaluate every valid label with dropout off

Append `evaluate` and `unigram_baseline` to `gpt_objective.py`:

<!-- file:gpt_objective.py mode:append -->
```python


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
```

`torch.no_grad()` avoids building a backward graph. `model.eval()` disables dropout. Neither substitutes for the other. The `finally` block restores the prior training/evaluation mode even if evaluation fails. The loop covers each validation window once in a fixed order and adds loss sums and token counts; its final value is independent of how a short final batch is grouped, apart from floating-point rounding.

The unigram baseline counts labels in the training partition, adds one to every vocabulary count and normalizes. Add-one smoothing gives every label positive probability. It ignores context completely. Beating it suggests learning useful predictive structure beyond marginal token frequency on this specific split. It does not establish reasoning, broad knowledge or useful prose.

Compute it before corpus training:

```bash
python - <<'PYCODE'
from gpt_data import load_data,windows
from gpt_objective import unigram_baseline
s,t,m=load_data('data/alice-byte')
_,train_y=windows(s['train'],t,64)
_,valid_y=windows(s['validation'],t,64)
print(unigram_baseline(train_y,valid_y,t.size))
PYCODE
```

Record the result with the split and tokenizer hashes. NLL is better when smaller; perplexity `exp(NLL)` is also better when smaller. Doubling perplexity adds ln(2)≈0.693147 nats to mean NLL.

[[practice:evaluation]]

### Compare tokenizations without fooling yourself

An 0.8 loss per byte-token and an 1.4 loss per BPE token have different denominators. A coarse token can represent several bytes. For the same raw held-out document set, a byte-normalized score can be computed as total negative log probability divided by UTF-8 byte count. State whether EOS losses are included and whether truncated-window context resets are used. Differences in that context policy can affect the comparison too.

Do not simply exponentiate a BPE-token average and compare it with byte perplexity. Keep original totals and counts so a report can reconstruct the metric. For this first run, compare only the same frozen byte tokenizer and windows.

### Sampling changes how scores become a choice

Greedy decoding chooses the largest logit. With temperature τ>0, sample from softmax(z/τ). For logits `[0,ln(3)]`, τ=1 gives probability 0.75 for the second item. At τ=0.5, its odds are 9:1, so probability becomes 0.9. Lower temperature concentrates probability; it does not improve the trained likelihood. Implement τ=0 as a separate greedy case to avoid division by zero.

Top-k keeps exactly k candidate logits before softmax. Structural PAD, BOS and SEP are excluded from generation; EOS remains eligible. This sampling restriction is a declared decoding policy, whereas training cross-entropy still normalizes over all V logits.

[[scenario:sampling]]
[[practice:temperature]]

### Write bounded, independent requests

Create `gpt_sample.py`:

<!-- file:gpt_sample.py mode:write -->
```python
"""Bounded uncached generation; positions reset within each cropped context."""
import argparse
import json
import math
from pathlib import Path
import torch
from gpt_config import Config
from gpt_model import GPT
from gpt_tokenizer import ByteBPE, BOS, EOS, PAD, SEP


def choose(logits, temperature, top_k, generator):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    scores = logits.clone()
    scores[[PAD, BOS, SEP]] = -float("inf")
    if temperature == 0:
        return int(scores.argmax())
    scores = scores / temperature
    if top_k is not None:
        if not 1 <= top_k <= scores.numel()-3:
            raise ValueError("top_k outside available content/EOS vocabulary")
        # Scatter actual top-k values; a threshold could retain extra tied values.
        values, indices = scores.topk(top_k)
        scores.fill_(-float("inf"))
        scores.scatter_(0, indices, values)
    if not bool(torch.isfinite(scores).any()):
        raise ValueError("No finite candidate")
    return int(torch.multinomial(scores.softmax(-1), 1, generator=generator))


@torch.no_grad()
def generate(model, tok, prompt, new_tokens=80, temperature=0.8, top_k=40, seed=29):
    if new_tokens < 0:
        raise ValueError("Nonnegative token budget required")
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    ids = [BOS] + tok.encode(prompt)
    created = []
    training = model.training
    model.eval()
    try:
        for _ in range(new_tokens):
            context = torch.tensor([ids[-model.config.context:]], dtype=torch.long, device=device)
            next_id = choose(model(context)[0, -1], temperature, top_k, rng)
            if next_id == EOS:
                break
            ids.append(next_id)
            created.append(next_id)
    finally:
        model.train(training)
    # Decode prompt and continuation together: a multibyte character can cross the seam.
    return {"text": tok.decode(ids[1:], errors="replace"), "generated_ids": created,
            "stopped_eos": len(created) < new_tokens, "utf8_policy": "replace"}


if __name__ == "__main__":
    from gpt_checkpoint import verified
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prompt", default="Alice ")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--tokens", type=int, default=80)
    args = p.parse_args()
    entries, rejected = verified(args.run)
    if not entries:
        raise ValueError("No verified state")
    state = entries[-1][2]
    tok = ByteBPE.load(args.data / "tokenizer.json")
    if tok.identity != state["identity"]["tokenizer"]:
        raise ValueError("Tokenizer differs from training")
    model = GPT(Config(**state["identity"]["model"]))
    model.load_state_dict(state["model"])
    print(json.dumps({"checkpoint_step": state["step"], "rejected": rejected,
                      **generate(model, tok, args.prompt, args.tokens, args.temperature, args.top_k)},
                     ensure_ascii=False))
```

Each call creates its own seeded generator and prefix. It never retains keys, values or old request state. At each iteration, run the entire most recent context window, take the last position's logits, choose one token, and stop at EOS or the new-token budget. There is no KV cache yet.

When a prefix exceeds T, this implementation crops to its last T tokens and resets learned position IDs to `0…T−1`. That is an explicit rolling-window baseline; it is not exact continuation of an unlimited absolute-position model. A later cache must reproduce the chosen position/window semantics or be compared only within the context bound.

We decode the combined prompt and generated bytes together, because a UTF-8 character can cross a token seam. Invalid generated bytes use replacement characters for display. Preserve generated IDs in evidence; a replacement character hides which byte sequence was invalid. The tokenizer's strict round-trip guarantee for valid input is a different property from arbitrary model output being valid UTF-8.

Before a trained checkpoint exists, call `generate(GPT(Config()),ByteBPE(),'hi',new_tokens=5)` from Python to verify the bound. Random-model text is expected to be poor. After Lesson 10 creates a run, the CLI becomes:

```bash
python gpt_sample.py --data data/alice-byte --run runs/alice-v1 --prompt "Alice " --tokens 80
```

Run three declared seeds and both greedy and temperature sampling after training. Save prompts, seeds, generation parameters, checkpoint identity and raw IDs. Do not keep only the most readable output. Note repeated characters, malformed bytes, broken word boundaries and whether the model stops at EOS.

[[practice:request-reset]]

### Separate four observations

| Observation | What it supports |
| --- | --- |
| Fixed batch loss falls | The chosen pipeline/optimizer can fit those examples |
| Held-out NLL improves | Better next-token probabilities on the frozen evaluation protocol |
| A sample looks readable | Qualitative behavior for that prompt/seed; susceptible to selection bias |
| Tests establish causal/gradient correctness | The implemented calculation obeys its checked contracts |

All four are useful; none replaces the others. Declare a small experiment before running: total updates, baseline, target, validation schedule and a few prompts. Keep the test chapters untouched until the final selected protocol.

Training now has meaningful outputs. The next problem is preserving progress when a process or runtime disappears.

[Continue to Lesson 9](lessons/09-persistent-checkpoints.md)

### Complete files from this lesson

[gpt_objective.py](examples/gpt_objective.py) · [gpt_sample.py](examples/gpt_sample.py)
