import math
import torch

def attend(q, k, v, allowed=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    allowed = torch.broadcast_to(allowed, scores.shape)
    has_keys = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, -torch.inf)
    safe_scores = torch.where(has_keys, scores, torch.zeros_like(scores))
    shifted = safe_scores - safe_scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    weights = torch.where(has_keys, weights, torch.zeros_like(weights))
    return weights @ v, weights

torch.set_printoptions(precision=6, sci_mode=False)
src = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
tgt = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
w_q = torch.eye(2, dtype=src.dtype)
w_k = torch.eye(2, dtype=src.dtype)
w_v = 2 * torch.eye(2, dtype=src.dtype)

q = tgt @ w_q
k = src @ w_k
v = src @ w_v
context, weights = attend(q, k, v)
print(weights.shape)
print(context)

source_validity = torch.tensor([[True, True, True, False]])
masked_context, _ = attend(q, k, v, source_validity)
print(masked_context)
