"""Causal weights, mask-order equivalence, dropout and prefix invariance."""
import math
import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout=0.5):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError('Drop probability must be in [0, 1)')
        self.query = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
        self.key = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
        self.value = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('future', torch.triu(
            torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        length = x.shape[-2]
        if not 0 < length <= self.future.shape[0]:
            raise ValueError('Sequence length is outside the prepared context')
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(self.future[:length, :length], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        dropped = self.dropout(weights)
        return dropped @ v, weights, dropped


if __name__ == '__main__':
    torch.manual_seed(789)
    x = torch.tensor([[.72, .45, .31], [.75, .20, .55], [.30, .80, .40],
                      [.85, .35, .60], [.55, .15, .75], [.25, .20, .85]],
                     dtype=torch.float64)
    layer = CausalAttention(3, 2, 6, dropout=.5)
    layer.eval()
    output, weights, dropped = layer(x)
    torch.testing.assert_close(weights.sum(-1), torch.ones(6, dtype=x.dtype))
    assert torch.equal(weights.triu(1), torch.zeros_like(weights))
    changed = x.clone()
    changed[3:] = torch.tensor([99., -37., 12.], dtype=x.dtype)
    torch.testing.assert_close(layer(changed)[0][:3], output[:3], atol=1e-14, rtol=0)
    torch.testing.assert_close(layer(x[:3])[0], output[:3], atol=1e-14, rtol=0)
    q, k = layer.query(x), layer.key(x)
    scores = q @ k.T / math.sqrt(2)
    full = torch.softmax(scores, dim=-1)
    zeroed = full.masked_fill(layer.future, 0.0)
    repaired = zeroed / zeroed.sum(-1, keepdim=True)
    torch.testing.assert_close(repaired, weights, atol=1e-14, rtol=0)
    torch.testing.assert_close(full[-1], weights[-1], atol=1e-14, rtol=0)
    extreme = torch.tensor([0., 1000.], dtype=x.dtype)
    lost = torch.softmax(extreme, dim=-1)[0]
    assert lost.item() == 0  # Too late to recover this allowed probability.
    safe = torch.softmax(extreme.masked_fill(torch.tensor([False, True]), -torch.inf), dim=-1)
    assert safe.tolist() == [1., 0.]
    print('evaluation weights', weights.tolist())
    layer.train()
    torch.manual_seed(123)
    _, original, after_dropout = layer(x)
    print('dropout weights', after_dropout.tolist())
    assert torch.all((after_dropout == 0) | (after_dropout == 2 * original))
    layer.eval()
    assert torch.equal(layer(x)[1], layer(x)[2])
    print('Prefix interventions, short contexts, both mask orders and mode checks pass.')
