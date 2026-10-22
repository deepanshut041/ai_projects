"""Learned attention with inspection and efficient execution paths."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def split_heads(x, num_heads):
    batch, length, width = x.shape
    if width % num_heads:
        raise ValueError("feature width must be divisible by num_heads")
    return x.reshape(batch, length, num_heads, width // num_heads).transpose(1, 2)


def merge_heads(x):
    return x.transpose(1, 2).contiguous().flatten(2)


def allowed_keys(key_mask, query_length, *, causal=False):
    """True means that this query may read this source key."""
    if key_mask.ndim != 2 or key_mask.dtype != torch.bool:
        raise ValueError("key_mask must be a Boolean [batch, source_length] tensor")
    source_length = key_mask.shape[-1]
    allowed = key_mask[:, None, None, :].expand(-1, 1, query_length, -1)
    if causal:
        if query_length != source_length:
            raise ValueError("causal self-attention requires equal query/key lengths")
        history = torch.ones(query_length, source_length, dtype=torch.bool,
                             device=key_mask.device).tril()
        allowed = allowed & history[None, None]
    return allowed


def attention_math(q, k, v, allowed=None):
    """Return context and probabilities; a wholly blocked row returns zero."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if allowed is None:
        weights = scores.softmax(dim=-1)
    else:
        scores = scores.masked_fill(~allowed, -torch.inf)
        nonempty = allowed.any(dim=-1, keepdim=True)
        safe_scores = torch.where(nonempty, scores, torch.zeros_like(scores))
        weights = safe_scores.softmax(dim=-1).masked_fill(~allowed, 0)
    return weights @ v, weights


class Attention(nn.Module):
    def __init__(self, embedding_dimension=32, num_heads=4, attn_dropout=0.0,
                 *, fused_attn=True):
        super().__init__()
        if embedding_dimension <= 0 or num_heads <= 0 or embedding_dimension % num_heads:
            raise ValueError("choose positive dimensions with width divisible by heads")
        if not 0 <= attn_dropout < 1:
            raise ValueError("attn_dropout must be in [0, 1)")
        self.num_heads = num_heads
        self.head_dim = embedding_dimension // num_heads
        self.attn_dropout = attn_dropout
        self.fused_attn = fused_attn
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, src, tgt=None, attention_mask=None, causal=False,
                *, need_weights=False):
        query_input = src if tgt is None else tgt
        if tgt is not None and causal:
            raise ValueError("this translation cross-attention reads all source positions")
        q = split_heads(self.q_proj(query_input), self.num_heads)
        k = split_heads(self.k_proj(src), self.num_heads)
        v = split_heads(self.v_proj(src), self.num_heads)
        if attention_mask is None:
            attention_mask = torch.ones(src.shape[:2], dtype=torch.bool, device=src.device)
        if tuple(attention_mask.shape) != tuple(src.shape[:2]):
            raise ValueError("attention_mask must describe the source keys")
        allowed = allowed_keys(attention_mask, q.shape[-2], causal=causal)
        drop = self.attn_dropout if self.training else 0.0
        if self.fused_attn and not need_weights:
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                     dropout_p=drop, is_causal=False)
            weights = None
        else:
            context, weights = attention_math(q, k, v, allowed)
            if drop:
                context = F.dropout(weights, p=drop, training=True) @ v
        output = self.out_proj(merge_heads(context))
        return (output, weights) if need_weights else output


def main():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    layer = Attention(2, 1, fused_attn=False)
    with torch.no_grad():
        layer.q_proj.weight.copy_(torch.tensor([[1., 1.], [0., 1.]]))
        layer.k_proj.weight.copy_(torch.tensor([[1., 0.], [1., -1.]]))
        layer.v_proj.weight.copy_(torch.tensor([[2., 0.], [0., 3.]]))
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj):
            projection.bias.zero_()
        layer.out_proj.weight.copy_(torch.eye(2))
        layer.out_proj.bias.zero_()
        q, k, v = layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)
        output, weights = layer(x, need_weights=True)
    for name, value in (("Q", q), ("K", k), ("V", v), ("Q @ K.T", q @ k.transpose(-2, -1)),
                        ("A", weights), ("A @ V", output)):
        print(name, value.squeeze().tolist())


if __name__ == "__main__":
    main()
