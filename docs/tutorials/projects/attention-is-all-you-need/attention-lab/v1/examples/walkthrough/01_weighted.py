import torch

values = torch.tensor([[2., 0.], [0., 2.], [2., 2.], [0., 0.]], dtype=torch.float64)
weights = torch.tensor([0.4, 0.2, 0.3, 0.1], dtype=torch.float64)

contributions = weights[:, None] * values
context = contributions.sum(dim=0)

print(contributions)
print(context)
print(weights @ values)
