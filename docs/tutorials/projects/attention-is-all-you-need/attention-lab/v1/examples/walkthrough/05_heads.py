import math
import torch

def attend(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    return weights @ v, weights

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [0., 0.]]], dtype=torch.float64)
w_q = torch.eye(2, dtype=x.dtype)
w_k = torch.eye(2, dtype=x.dtype)
w_v = 2 * torch.eye(2, dtype=x.dtype)
w_out = torch.eye(2, dtype=x.dtype)
q, k, v = x @ w_q, x @ w_k, x @ w_v

batch, positions, width = x.shape
heads = 2
assert width % heads == 0
head_width = width // heads
q_heads = q.reshape(batch, positions, heads, head_width).transpose(1, 2)
k_heads = k.reshape(batch, positions, heads, head_width).transpose(1, 2)
v_heads = v.reshape(batch, positions, heads, head_width).transpose(1, 2)

head_context, head_weights = attend(q_heads, k_heads, v_heads)
joined = head_context.transpose(1, 2).reshape(batch, positions, width)
output = joined @ w_out

print(q_heads.shape)
print(head_weights[0, 0, 0])
print(head_weights[0, 1, 0])
print(output[0])
