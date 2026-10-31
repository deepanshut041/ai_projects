"""Readable causal attention and an independent library comparison path."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def attend(q, k, v, allowed, *, optimized=False, dropout=0.0, training=False):
    probability = dropout if training else 0.0
    if optimized:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                              dropout_p=probability)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = scores.softmax(dim=-1)
    weights = F.dropout(weights, p=probability, training=training)
    return weights @ v


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)

    def forward(self, x, valid, *, optimized=False):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(tensor):
            return tensor.reshape(batch, time, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = map(heads, (q, k, v))
        causal = torch.ones(time, time, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        # Model input requires a real first token. Even padded queries see it;
        # they are discarded by the loss. No softmax row is entirely masked.
        out = attend(q, k, v, allowed, optimized=optimized,
                     dropout=self.dropout, training=self.training)
        out = out.transpose(1, 2).contiguous().view(batch, time, width)
        return F.dropout(self.output(out), p=self.dropout, training=self.training)
