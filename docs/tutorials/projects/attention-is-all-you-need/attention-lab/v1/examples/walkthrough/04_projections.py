import math
import torch

def attend(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    return weights @ v, weights

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
w_q = torch.eye(2, dtype=x.dtype)
w_k = torch.eye(2, dtype=x.dtype)
w_v = 2 * torch.eye(2, dtype=x.dtype)

q = x @ w_q
k = x @ w_k
v = x @ w_v
context, weights = attend(q, k, v)
print(q)
print(k)
print(v)
print(context[0])

half_context, same_weights = attend(q, k, x @ (w_v / 2))
assert torch.allclose(same_weights, weights)
assert torch.allclose(half_context, context / 2)
