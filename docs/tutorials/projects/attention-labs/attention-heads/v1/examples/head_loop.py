"""An explicit ModuleList of independent causal heads, followed by output mixing."""
import math
import torch
from torch import nn


class CausalHead(nn.Module):
    def __init__(self, width, head_dim, context_len, dropout=0.0):
        super().__init__()
        self.query = nn.Linear(width, head_dim, bias=False, dtype=torch.float64)
        self.key = nn.Linear(width, head_dim, bias=False, dtype=torch.float64)
        self.value = nn.Linear(width, head_dim, bias=False, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('future', torch.ones(context_len, context_len,
                             dtype=torch.bool).triu(1))

    def forward(self, x):
        length = x.shape[-2]
        if not 0 < length <= self.future.shape[0]:
            raise ValueError('Input is outside the prepared context length')
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        weights = scores.masked_fill(self.future[:length, :length], -torch.inf).softmax(-1)
        return self.dropout(weights) @ v


class LoopMHA(nn.Module):
    def __init__(self, width=6, heads=2, context_len=6, dropout=0.0):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('This layout requires a positive width divisible by heads')
        self.heads = nn.ModuleList([
            CausalHead(width, width // heads, context_len, dropout) for _ in range(heads)])
        self.output = nn.Linear(width, width, bias=False, dtype=torch.float64)

    def forward(self, x):
        joined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.output(joined)


if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.randn(2, 6, 6, dtype=torch.float64)
    layer = LoopMHA().eval()
    print('input shape', list(x.shape))
    print('each head shape', [list(head(x).shape) for head in layer.heads])
    print('output shape', list(layer(x).shape))
    print('registered parameter count', sum(p.numel() for p in layer.parameters()))
    assert layer(x).shape == x.shape
    assert sum(p.numel() for p in layer.parameters()) == 144
