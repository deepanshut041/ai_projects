from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

def _dimensions(embed_dim: int, num_heads: int) -> int:
    """Check that equal-width heads fit the embedding and return head width."""
    if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
        raise ValueError("embedding dimension must be positive and divisible by num_heads")
    return embed_dim // num_heads

def _sequence(x: Tensor, embed_dim: int, name: str) -> None:
    """Validate a floating-point (batch, sequence, embedding) tensor."""
    if x.ndim != 3 or x.shape[-1] != embed_dim or x.shape[1] == 0:
        raise ValueError(f"{name} must have shape (batch, nonempty sequence, {embed_dim})")
    if not x.is_floating_point():
        raise TypeError(f"{name} must contain floating-point embeddings")

def make_attention_mask(
    attention_mask: Tensor | None,
    query_length: int,
    key_length: int,
    *,
    causal: bool = False,
    device: torch.device | str | None = None,
) -> Tensor | None:
    """Build the allowed query-key pairs from padding and causal constraints.

    attention_mask has shape (batch, keys), with True/1 for valid source keys.
    Padding produces a (batch, 1, 1, keys) mask that broadcasts across heads
    and queries. Causality combines it with a lower triangle, so query i can
    read only keys j <= i. The query and key lengths must then be equal.

    Return None when neither constraint is present. A padding mask excludes
    key columns; it does not zero the outputs of padded query positions.
    """
    allowed = None
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[1] != key_length:
            raise ValueError("attention_mask must have shape (batch, source sequence)")
        if attention_mask.dtype != torch.bool:
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask values must be boolean or 0/1")
        if device is None:
            device = attention_mask.device
        allowed = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
    if causal:
        if query_length != key_length:
            raise ValueError("causal attention requires self-attention with equal lengths")
        triangle = torch.ones(query_length, key_length, dtype=torch.bool, device=device).tril()
        triangle = triangle[None, None, :, :]
        allowed = triangle if allowed is None else allowed & triangle
    return allowed

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Tensor | None = None,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute softmax(Q K^T / sqrt(head_dim)) V and return context and weights.

    q has shape (..., queries, head_dim), k has (..., keys, head_dim), and
    v has (..., keys, value_dim). The result has (..., queries, value_dim)
    shape; weights have (..., queries, keys) shape.

    The boolean mask must broadcast to the scores, with True for allowed
    pairs. Softmax runs across keys. Fully blocked rows produce zero weights
    and context. Returned weights are before dropout, so valid rows sum to
    one even when dropout is applied to the weights used for the context.
    """
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    has_keys = None
    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        has_keys = attention_mask.any(dim=-1, keepdim=True)
        # softmax([-inf, ...]) is undefined. Replace only fully blocked rows.
        scores = torch.where(has_keys, scores, 0.0)
    weights = scores.softmax(dim=-1)
    if has_keys is not None:
        weights = torch.where(has_keys, weights, 0.0)
    context = F.dropout(weights, p=dropout_p, training=training) @ v
    return context, weights

def _heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """Split (batch, sequence, embedding) into (batch, heads, sequence, width)."""
    batch, length, _ = x.shape
    return x.reshape(batch, length, num_heads, head_dim).transpose(1, 2)

def _context(
    q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor | None,
    *, causal: bool, dropout_p: float, training: bool,
    fused_attn: bool, need_weights: bool,
) -> tuple[Tensor, Tensor | None]:
    """Select SDPA or explicit attention with the same masking and dropout rules.

    Requesting weights selects the explicit path. SDPA handles a causal mask
    internally when no padding mask is present; otherwise combine the masks
    before the call. Dropout applies only during training.
    """
    mask = make_attention_mask(
        attention_mask, q.shape[-2], k.shape[-2],
        # SDPA can build its own causal mask when there is no padding mask.
        causal=causal and not (fused_attn and not need_weights and attention_mask is None),
        device=q.device,
    )
    if fused_attn and not need_weights:
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=causal and attention_mask is None,
        )
        return context, None
    return scaled_dot_product_attention(q, k, v, mask, dropout_p=dropout_p, training=training)

def _validate_inputs(src, tgt, attention_mask, embed_dim):
    """Check embedding shapes, matching batches/devices/dtypes and source mask."""
    _sequence(src, embed_dim, "src")
    if tgt is not None:
        _sequence(tgt, embed_dim, "tgt")
        if tgt.shape[0] != src.shape[0] or tgt.dtype != src.dtype or tgt.device != src.device:
            raise ValueError("src and tgt must share batch size, dtype and device")
    if attention_mask is not None and attention_mask.shape != src.shape[:2]:
        raise ValueError("attention_mask must have shape (batch, source sequence)")

class Attention(nn.Module):
    """Multi-head attention supporting self-attention and cross-attention.

    src: (batch, source length, embedding_dimension)
    tgt: optional (batch, target length, embedding_dimension)
    attention_mask: optional (batch, source length), True/1 = allowed key

    Without tgt, all three projections use src. With tgt, queries come from
    tgt and keys/values come from src, so output length follows tgt. Split
    each projection into equal-width heads, compute attention, concatenate
    the head contexts and apply out_proj to mix them.

    causal=True applies only to self-attention. Padding masks always refer
    to source keys, with True/1 meaning allowed. The default fused path uses
    PyTorch SDPA; fused_attn=False computes scores and weights explicitly.
    Dropout applies only during training. need_weights=True uses the explicit
    path and returns (output, weights), with weights shaped
    (batch, heads, query_length, source_length).
    """

    def __init__(self, embedding_dimension=768, num_heads=12, attn_dropout=0.0,
                 *, fused_attn=True):
        super().__init__()
        self.head_dim = _dimensions(embedding_dimension, num_heads)
        if not 0 <= attn_dropout <= 1:
            raise ValueError("attn_dropout must lie between zero and one")
        self.embedding_dimension = embedding_dimension
        self.num_heads, self.attn_dropout = num_heads, attn_dropout
        self.fused_attn = fused_attn
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, src, tgt=None, attention_mask=None, causal=False, *, need_weights=False):
        _validate_inputs(src, tgt, attention_mask, self.embedding_dimension)
        if tgt is not None and causal:
            raise ValueError("causal masking is only supported for self-attention")
        query_input = src if tgt is None else tgt
        q = _heads(self.q_proj(query_input), self.num_heads, self.head_dim)
        k = _heads(self.k_proj(src), self.num_heads, self.head_dim)
        v = _heads(self.v_proj(src), self.num_heads, self.head_dim)
        context, weights = _context(
            q, k, v, attention_mask, causal=causal,
            dropout_p=self.attn_dropout, training=self.training,
            fused_attn=self.fused_attn, need_weights=need_weights,
        )
        output = self.out_proj(context.transpose(1, 2).flatten(2))
        return (output, weights) if need_weights else output

# Reproduce the same one-head example through the actual project class.
torch.set_printoptions(precision=6, sci_mode=False)
src = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [0., 0.]]], dtype=torch.float64)
layer = Attention(embedding_dimension=2, num_heads=1, fused_attn=False).double()
with torch.no_grad():
    layer.q_proj.weight.copy_(torch.eye(2, dtype=src.dtype))
    layer.k_proj.weight.copy_(torch.eye(2, dtype=src.dtype))
    layer.v_proj.weight.copy_(2 * torch.eye(2, dtype=src.dtype))
    layer.out_proj.weight.copy_(torch.eye(2, dtype=src.dtype))
    for projection in [layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj]:
        projection.bias.zero_()
layer.eval()
output, weights = layer(src, need_weights=True)
print(output[0])

fast_layer = Attention(embedding_dimension=2, num_heads=1, fused_attn=True).double()
fast_layer.load_state_dict(layer.state_dict())
fast_layer.eval()
fast_output = fast_layer(src)
assert torch.allclose(fast_output, output, atol=1e-12, rtol=1e-12)
