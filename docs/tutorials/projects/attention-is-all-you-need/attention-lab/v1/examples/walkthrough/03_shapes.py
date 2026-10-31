import math
import torch

def attend(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    context = weights @ v
    return context, weights

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
q = x
k = x
v = 2 * x
context, weights = attend(q, k, v)
print(weights)
print(context)

batch_x = x.unsqueeze(0)
batch_context, batch_weights = attend(batch_x, batch_x, 2 * batch_x)
print(batch_context.shape)
assert torch.allclose(batch_context[0], context)
