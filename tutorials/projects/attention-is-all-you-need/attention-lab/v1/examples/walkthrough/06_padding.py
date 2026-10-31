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
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
valid_keys = torch.tensor([[True, True, True, False]])
context, weights = attend(x, x, 2 * x, valid_keys)
print(weights[0])
print(context)

changed = x.clone()
changed[3] = torch.tensor([1000., -1000.], dtype=x.dtype)
changed_context, _ = attend(changed, changed, 2 * changed, valid_keys)
assert torch.allclose(changed_context[:3], context[:3])

no_keys = torch.zeros((1, 4), dtype=torch.bool)
empty_context, empty_weights = attend(x[:1], x, 2 * x, no_keys)
print(empty_context)
assert torch.equal(empty_weights, torch.zeros_like(empty_weights))
