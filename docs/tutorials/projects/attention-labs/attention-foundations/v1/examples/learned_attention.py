"""Five-word projection snapshot, two equivalent classes and a variance probe."""
import math
import torch
from torch import nn


class ParameterAttention(nn.Module):
    def __init__(self, d_in, d_key, d_value):
        super().__init__()
        self.WQ = nn.Parameter(torch.rand(d_in, d_key, dtype=torch.float64))
        self.WK = nn.Parameter(torch.rand(d_in, d_key, dtype=torch.float64))
        self.WV = nn.Parameter(torch.rand(d_in, d_value, dtype=torch.float64))

    def forward(self, x):
        q, k, v = x @ self.WQ, x @ self.WK, x @ self.WV
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
        return weights @ v, weights


class LinearLayerAttention(nn.Module):
    # This name means linear projection layers, not recurrent linear attention.
    def __init__(self, d_in, d_key, d_value):
        super().__init__()
        self.query = nn.Linear(d_in, d_key, bias=False, dtype=torch.float64)
        self.key = nn.Linear(d_in, d_key, bias=False, dtype=torch.float64)
        self.value = nn.Linear(d_in, d_value, bias=False, dtype=torch.float64)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
        return weights @ v, weights


if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.tensor([[.5, 1.2, .8], [1.1, .3, .7], [.6, 1.5, .9],
                      [.9, .8, .2], [1.2, 1.7, .5]], dtype=torch.float64)
    wq = torch.tensor([[.6, .2], [.1, .7], [.5, .4]], dtype=torch.float64)
    wk = torch.tensor([[.3, .6], [.4, .2], [.7, .5]], dtype=torch.float64)
    wv = torch.tensor([[.2, .5], [.8, .1], [.3, .6]], dtype=torch.float64)
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = q @ k.T
    weights = torch.softmax(scores / math.sqrt(2), dim=-1)
    contexts = weights @ v
    for name, matrix in [('Q', q), ('K', k), ('V', v), ('scores', scores),
                         ('weights', weights), ('contexts', contexts)]:
        print(name, matrix.tolist())
    manual = sum((weights[-1, j] * v[j] for j in range(5)), torch.zeros(2, dtype=x.dtype))
    torch.testing.assert_close(manual, contexts[-1], atol=1e-14, rtol=0)
    explicit = ParameterAttention(3, 2, 2)
    linear = LinearLayerAttention(3, 2, 2)
    with torch.no_grad():
        for dest, source in [(explicit.WQ, wq), (explicit.WK, wk), (explicit.WV, wv),
                             (linear.query.weight, wq.T), (linear.key.weight, wk.T),
                             (linear.value.weight, wv.T)]:
            dest.copy_(source)
    torch.testing.assert_close(explicit(x)[0], contexts, atol=1e-14, rtol=0)
    torch.testing.assert_close(linear(x)[0], contexts, atol=1e-14, rtol=0)
    batch = torch.stack([x, x + .1])
    torch.testing.assert_close(explicit(batch)[0], linear(batch)[0], atol=1e-14, rtol=0)
    torch.manual_seed(123)
    for width in [5, 100]:
        queries = torch.randn(20000, width, dtype=torch.float64)
        keys = torch.randn(20000, width, dtype=torch.float64)
        raw = (queries * keys).sum(dim=-1)
        print('width', width, 'raw variance', raw.var(unbiased=False).item(),
              'scaled variance', (raw / math.sqrt(width)).var(unbiased=False).item())
    print('Both classes match the fixed snapshot, including the batch path.')
