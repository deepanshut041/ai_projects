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
