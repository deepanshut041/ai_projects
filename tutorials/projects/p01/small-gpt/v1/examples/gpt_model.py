"""Complete small pre-norm GPT. Explicit attention, learned positions, tied head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from gpt_config import Config
from gpt_attention import CausalAttention
from gpt_tokenizer import PAD


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(nn.Linear(config.width, config.hidden),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(config.hidden, config.width),
                                 nn.Dropout(config.dropout))

    def forward(self, x, valid, *, optimized=False):
        x = x + self.attention(self.norm1(x), valid, optimized=optimized)
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.dropout = nn.Dropout(config.dropout)
        # F.linear below uses the SAME embedding parameter; no duplicate head.
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=0.02 / math.sqrt(2*config.layers))
            nn.init.normal_(block.ffn[2].weight, std=0.02 / math.sqrt(2*config.layers))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, ids, *, optimized=False):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("Token IDs must be a [B,T] int64 tensor")
        batch, time = ids.shape
        if batch < 1 or not 1 <= time <= self.config.context:
            raise ValueError("Empty input or context limit exceeded")
        if bool(((ids < 0) | (ids >= self.config.vocab)).any()):
            raise ValueError("Token ID outside frozen vocabulary")
        valid = ids != PAD
        if not bool(valid[:, 0].all()) or bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("Use nonempty, right-padded sequences")
        positions = torch.arange(time, device=ids.device)
        x = self.dropout(self.token(ids) + self.position(positions))
        for block in self.blocks:
            x = block(x, valid, optimized=optimized)
        return F.linear(self.final_norm(x), self.token.weight)


def parameter_formula(c):
    # Per block: QKV/output + two FFN matrices + all biases + two norms.
    return c.vocab*c.width + c.context*c.width + c.layers * (
        4*c.width*c.width + 2*c.width*c.hidden + 9*c.width + c.hidden) + 2*c.width
