import math
import torch

torch.set_printoptions(precision=6, sci_mode=False)
query = torch.tensor([1., 0.], dtype=torch.float64)
keys = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
values = torch.tensor([[2., 0.], [0., 2.], [2., 2.], [0., 0.]], dtype=torch.float64)

raw_scores = query @ keys.T
scores = raw_scores / math.sqrt(query.shape[-1])
shifted_scores = scores - scores.max()
positive_weights = torch.exp(shifted_scores)
weights = positive_weights / positive_weights.sum()
context = weights @ values

print(raw_scores)
print(weights)
print(context)
