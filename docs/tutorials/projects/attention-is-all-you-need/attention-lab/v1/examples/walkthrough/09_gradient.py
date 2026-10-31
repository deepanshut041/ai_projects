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
w_q = torch.eye(2, dtype=x.dtype).requires_grad_()
w_k = torch.eye(2, dtype=x.dtype)
w_v = 2 * torch.eye(2, dtype=x.dtype)
target = torch.tensor([[2., 0.]], dtype=x.dtype)

q = x[:1] @ w_q
k, v = x @ w_k, x @ w_v
context, weights = attend(q, k, v)
loss = 0.5 * ((context - target) ** 2).sum()
loss.backward()

print(context.detach())
print(loss.item())
print(w_q.grad)

with torch.no_grad():
    new_w_q = w_q - 0.1 * w_q.grad
    new_context, new_weights = attend(x[:1] @ new_w_q, k, v)
    new_loss = 0.5 * ((new_context - target) ** 2).sum()

print(new_context)
print(new_loss.item())
assert new_loss < loss.detach()
