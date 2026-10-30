"""CPU float64 correctness probe, not a performance benchmark or trained GPT."""
import json
import math
import torch
from torch import nn


class PackedMHA(nn.Module):
    def __init__(self, width=2, heads=2):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('Positive width must be divisible by positive head count')
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x):
        batch, tokens, width = x.shape
        def split(layer):
            projected = layer(x)
            return projected.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.query), split(self.key), split(self.value)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        forbidden = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(forbidden, -torch.inf).softmax(dim=-1)
        contexts = weights @ v
        joined = contexts.transpose(1, 2).contiguous().reshape(batch, tokens, width)
        return self.output(joined), weights


def loop_reference(x, model):
    batch, tokens, width = x.shape
    mask = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).triu(1)
    outputs = []
    for index in range(model.heads):
        start, end = index * model.head_dim, (index + 1) * model.head_dim
        q = x @ model.query.weight[start:end].T
        k = x @ model.key.weight[start:end].T
        v = x @ model.value.weight[start:end].T
        scores = q @ k.transpose(-2, -1) / math.sqrt(model.head_dim)
        weights = scores.masked_fill(mask, -torch.inf).softmax(-1)
        outputs.append(weights @ v)
    return model.output(torch.cat(outputs, dim=-1))


if __name__ == '__main__':
    torch.manual_seed(17)
    model = PackedMHA().double().eval()
    with torch.no_grad():
        model.query.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.key.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.value.weight.copy_(torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64))
        model.output.weight.copy_(torch.tensor([[1., -0.5], [0.5, 1.]], dtype=torch.float64))
    x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]],
                      [[2., 0.], [0., 2.], [1., -1.]]], dtype=torch.float64, requires_grad=True)
    result, weights = model(x)
    looped = loop_reference(x, model)
    torch.testing.assert_close(result, looped, atol=1e-12, rtol=1e-12)
    loss = result.square().sum()
    parameter = model.query.weight
    (packed_gradient,) = torch.autograd.grad(loss, parameter, retain_graph=True)
    (looped_gradient,) = torch.autograd.grad(looped.square().sum(), parameter)
    torch.testing.assert_close(packed_gradient, looped_gradient, atol=1e-12, rtol=1e-12)
    changed = x.detach().clone()
    changed[:, -1] = 50.0
    torch.testing.assert_close(model(changed)[0][:, :2], result.detach()[:, :2])
    print(json.dumps({'shape': list(result.shape), 'weight_shape': list(weights.shape),
                      'first_sequence_output': [[round(v, 6) for v in row] for row in result[0].tolist()],
                      'max_output_difference': round((result - looped).abs().max().item(), 12),
                      'gradients_equal': True, 'future_invariant': True,
                      'torch_version': torch.__version__}))
