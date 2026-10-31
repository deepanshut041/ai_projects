"""Small, inspectable foundation probes, not a Transformer implementation.

All attention masks in this module use True = allowed. Empty attention rows
have zero weights and output. No dropout is applied inside attention. Matrix
weights use the mathematical row-vector orientation [input, output].

Torch imports are lazy so that merely importing the file needs only Python.
Running the probes requires PyTorch; the recorded checks use CPU float64.
"""

import json
import math


def _torch():
    import torch
    return torch


def _floating(*tensors):
    torch = _torch()
    if not tensors or any(not x.is_floating_point() for x in tensors):
        raise ValueError("use floating-point tensors")
    first = tensors[0]
    if any(x.dtype != first.dtype or x.device != first.device for x in tensors):
        raise ValueError("all tensors must share dtype and device")
    if any(not torch.isfinite(x).all() for x in tensors):
        raise ValueError("inputs must be finite")


def sinusoidal_positions(length, width, *, offset=0, base=10000.0, dtype=None):
    """Return [length,width]; an odd width ends with an unpaired sine."""
    torch = _torch()
    if any(type(n) is not int for n in (length, width, offset)):
        raise ValueError("length, width and offset must be integers")
    if length < 0 or width < 1 or offset < 0 or not math.isfinite(base) or base <= 1:
        raise ValueError("need length >= 0, width > 0, offset >= 0 and base > 1")
    dtype = torch.float64 if dtype is None else dtype
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("position coordinates must be floating point")
    positions = torch.arange(offset, offset + length, dtype=dtype).unsqueeze(1)
    pairs = torch.arange(0, width, 2, dtype=dtype)
    angles = positions * torch.exp(-math.log(base) * pairs / width)
    result = torch.empty((length, width), dtype=dtype)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :width // 2].cos()
    return result


def masked_softmax(scores, allowed=None):
    """Normalize keys (last axis); deliberately return zero for empty rows."""
    torch = _torch()
    if not scores.is_floating_point() or scores.ndim < 1 or scores.shape[-1] == 0:
        raise ValueError("scores need a nonempty floating key axis")
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    if allowed.dtype != torch.bool or allowed.device != scores.device:
        raise ValueError("allowed must be Boolean on the score device")
    try:
        allowed = torch.broadcast_to(allowed, scores.shape)
    except RuntimeError as error:
        raise ValueError("allowed must broadcast to score shape") from error
    if not torch.isfinite(scores[allowed]).all():
        raise ValueError("allowed scores must be finite")
    nonempty = allowed.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(nonempty, masked, torch.zeros_like(masked))
    return safe.softmax(dim=-1) * nonempty


def _attention_shapes(q, k, v):
    _floating(q, k, v)
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError("Q, K and V need sequence and feature axes")
    if not q.shape[:-2] == k.shape[:-2] == v.shape[:-2]:
        raise ValueError("Q, K and V must share explicit leading batch/head axes")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K widths and K/V source lengths must match")
    if min(q.shape[-1], k.shape[-2], v.shape[-1]) < 1:
        raise ValueError("feature widths and source length must be positive")


def explicit_attention(q, k, v, allowed=None):
    """Return (output,weights) for [...,T,d], [...,S,d], [...,S,dv]."""
    _attention_shapes(q, k, v)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    weights = masked_softmax(scores, allowed)
    return weights @ v, weights


def attention_backward(q, k, v, upstream, allowed=None):
    """Manual VJP through AV, row softmax and scaled QK^T, without autograd."""
    torch = _torch()
    _attention_shapes(q, k, v)
    _floating(q, upstream)
    if upstream.shape != (*q.shape[:-1], v.shape[-1]):
        raise ValueError("upstream must match the attention output")
    with torch.no_grad():
        _, weights = explicit_attention(q, k, v, allowed)
        d_weights = upstream @ v.transpose(-2, -1)
        row_average = (weights * d_weights).sum(dim=-1, keepdim=True)
        d_scores = weights * (d_weights - row_average)
        scale = 1 / math.sqrt(q.shape[-1])
        return {
            "q": (d_scores @ k) * scale,
            "k": (d_scores.transpose(-2, -1) @ q) * scale,
            "v": weights.transpose(-2, -1) @ upstream,
            "scores": d_scores,
        }


def causal_allowed(query_positions, key_positions):
    """Absolute-position alignment works for full sequences and cached prefixes."""
    torch = _torch()
    if query_positions.ndim != 1 or key_positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if query_positions.dtype != torch.long or key_positions.dtype != torch.long:
        raise ValueError("positions must be integer long tensors")
    if query_positions.device != key_positions.device:
        raise ValueError("positions must share a device")
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def multihead_attention(xq, xk, xv, wq, wk, wv, wo, *, heads,
                        allowed=None, biases=None, implementation="vectorized"):
    """Projected [B,T,D] -> [B,T,D], with either vectorized or head-loop execution.

    The loop uses the exact same output-column slices as the vectorized path;
    it provides a reference for axis mistakes, not a separate learned model.
    Biases, if given, are four [D] tensors. A final output bias can make an
    all-masked row nonzero even though its attended vector is zero.
    """
    torch = _torch()
    _floating(xq, xk, xv, wq, wk, wv, wo)
    if any(x.ndim != 3 for x in (xq, xk, xv)):
        raise ValueError("streams must be [batch,length,width]")
    width = xq.shape[-1]
    if type(heads) is not int or heads < 1 or width < 1 or width % heads:
        raise ValueError("positive width must be divisible by positive head count")
    if xq.shape[0] != xk.shape[0] or xk.shape != xv.shape or xk.shape[-1] != width:
        raise ValueError("batch, model widths and K/V shapes must match")
    if any(w.shape != (width, width) for w in (wq, wk, wv, wo)):
        raise ValueError("projection weights must be [D,D] in row orientation")
    if biases is not None:
        if len(biases) != 4 or any(b.shape != (width,) for b in biases):
            raise ValueError("supply four [D] projection biases")
        _floating(xq, *biases)
    else:
        biases = (0, 0, 0, 0)
    q, k, v = xq @ wq + biases[0], xk @ wk + biases[1], xv @ wv + biases[2]
    batch, queries, _ = q.shape
    sources = k.shape[1]
    d = width // heads
    if allowed is None:
        allowed = torch.ones((batch, heads, queries, sources), dtype=torch.bool, device=q.device)
    else:
        if allowed.dtype != torch.bool or allowed.device != q.device:
            raise ValueError("allowed must be Boolean on the stream device")
        try:
            allowed = torch.broadcast_to(allowed, (batch, heads, queries, sources))
        except RuntimeError as error:
            raise ValueError("mask must broadcast to [B,H,T,S]") from error
    if implementation == "vectorized":
        split = lambda x: x.reshape(batch, x.shape[1], heads, d).transpose(1, 2)
        output, weights = explicit_attention(split(q), split(k), split(v), allowed)
        merged = output.transpose(1, 2).contiguous().reshape(batch, queries, width)
    elif implementation == "loop":
        outputs, rows = [], []
        for head in range(heads):
            span = slice(head * d, (head + 1) * d)
            output, attention = explicit_attention(q[..., span], k[..., span], v[..., span], allowed[:, head])
            outputs.append(output)
            rows.append(attention)
        merged, weights = torch.cat(outputs, dim=-1), torch.stack(rows, dim=1)
    else:
        raise ValueError("implementation must be vectorized or loop")
    return merged @ wo + biases[3], weights


def positionwise_ffn(x, w1, b1, w2, b2):
    """Original ReLU feature expansion/contraction with position-shared weights."""
    torch = _torch()
    _floating(x, w1, b1, w2, b2)
    if x.ndim < 1 or w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("need feature rows and matrix weights")
    width, hidden = w1.shape
    if min(width, hidden) < 1 or x.shape[-1] != width or w2.shape != (hidden, width):
        raise ValueError("FFN must expand D to F then contract F to D")
    if b1.shape != (hidden,) or b2.shape != (width,):
        raise ValueError("biases must match hidden and model widths")
    return torch.relu(x @ w1 + b1) @ w2 + b2


def layer_norm(x, normalized_shape, *, scale=None, bias=None, eps=1e-5):
    """Population variance over exactly the supplied trailing dimensions."""
    torch = _torch()
    _floating(x)
    if type(normalized_shape) is int:
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)
    if not normalized_shape or any(type(n) is not int or n < 1 for n in normalized_shape):
        raise ValueError("normalized_shape needs positive trailing dimensions")
    if len(normalized_shape) > x.ndim or tuple(x.shape[-len(normalized_shape):]) != normalized_shape:
        raise ValueError("normalized_shape must match the actual trailing dimensions")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    for affine in (scale, bias):
        if affine is not None:
            _floating(x, affine)
            if tuple(affine.shape) != normalized_shape:
                raise ValueError("scale and bias must match normalized_shape")
    axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    centered = x - x.mean(dim=axes, keepdim=True)
    variance = centered.square().mean(dim=axes, keepdim=True)
    normalized = centered / (variance + eps).sqrt()
    return normalized * (1 if scale is None else scale) + (0 if bias is None else bias)


def residual_norm(x, sublayer, *, placement="post", eps=1e-5):
    """Compare original post-norm and a later pre-norm arrangement."""
    if placement == "post":
        branch = sublayer(x)
        if branch.shape != x.shape:
            raise ValueError("residual branch must preserve the full shape")
        return layer_norm(x + branch, x.shape[-1], eps=eps)
    if placement == "pre":
        branch = sublayer(layer_norm(x, x.shape[-1], eps=eps))
        if branch.shape != x.shape:
            raise ValueError("residual branch must preserve the full shape")
        return x + branch
    raise ValueError("placement must be pre or post")


def inverted_dropout(x, keep, probability, *, training=True):
    """Apply a supplied Bernoulli keep mask; no hidden global RNG mutation."""
    torch = _torch()
    _floating(x)
    if not math.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError("probability must be in [0,1)")
    if not training or probability == 0:
        return x
    if keep.dtype != torch.bool or keep.shape != x.shape or keep.device != x.device:
        raise ValueError("keep must be a Boolean mask matching x")
    return x * keep / (1 - probability)


def smoothed_targets(targets, classes, smoothing=0.1, *, ignore_index=-100):
    """(1-epsilon)*one_hot + epsilon/classes, including the correct class."""
    torch = _torch()
    if type(classes) is not int or classes < 2 or targets.dtype != torch.long:
        raise ValueError("need at least two classes and integer long targets")
    if not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
        raise ValueError("smoothing must be in [0,1]")
    valid = targets != ignore_index
    if ((targets[valid] < 0) | (targets[valid] >= classes)).any():
        raise ValueError("target outside the class vocabulary")
    safe = torch.where(valid, targets, torch.zeros_like(targets))
    one_hot = torch.nn.functional.one_hot(safe, classes).to(torch.float64)
    return ((1 - smoothing) * one_hot + smoothing / classes) * valid.unsqueeze(-1)


def transformer_learning_rate(step, *, width=512, warmup=4000):
    """Paper's schedule at positive optimizer-update index (no extra factor)."""
    if any(type(n) is not int or n < 1 for n in (step, width, warmup)):
        raise ValueError("step, width and warmup must be positive integers")
    return width ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def parameter_counts(*, vocabulary, width, hidden, heads, biases=True):
    if any(type(n) is not int or n < 1 for n in (vocabulary, width, hidden, heads)) or width % heads:
        raise ValueError("positive dimensions and an exactly divisible head width are required")
    return {
        "three_untied_vocabulary_matrices": 3 * vocabulary * width,
        "one_tied_vocabulary_matrix": vocabulary * width,
        "multihead": 4 * width * width + (4 * width if biases else 0),
        "ffn": 2 * width * hidden + (hidden + width if biases else 0),
    }


def embedding_gradient_probe():
    """PAD suppresses lookup gradients, but tied output use can update PAD."""
    torch = _torch()
    table = torch.tensor([[0., 0.], [1., 2.], [3., 4.], [-1., 1.]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([2, 1, 2, 0])
    upstream = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [9., 9.]], dtype=torch.float64)
    lookup = torch.nn.functional.embedding(ids, table, padding_idx=0)
    lookup_grad, = torch.autograd.grad((lookup * upstream).sum(), table)
    hidden = torch.tensor([[1., 0.]], dtype=torch.float64)
    tied_logits = hidden @ table.transpose(0, 1)
    output_grad, = torch.autograd.grad(torch.nn.functional.cross_entropy(tied_logits, torch.tensor([1])), table)
    return {"lookup": lookup.detach(), "lookup_grad": lookup_grad, "tied_output_grad": output_grad}


def report():
    """Compact deterministic manifest output; tests supply the deeper proof."""
    torch = _torch()
    q = torch.tensor([[1.]], dtype=torch.float64)
    k = torch.tensor([[0.], [math.log(2)], [math.log(3)]], dtype=torch.float64)
    v = torch.tensor([[0., 0.], [6., 0.], [4., 2.]], dtype=torch.float64)
    output, weights = explicit_attention(q, k, v)
    masked, _ = explicit_attention(q, k, v, torch.tensor([[True, False, True]]))
    empty, _ = explicit_attention(q, k, v, torch.zeros((1, 3), dtype=torch.bool))
    backward = attention_backward(q, k, v, torch.tensor([[1., 0.]], dtype=torch.float64))
    embedding = embedding_gradient_probe()
    x = torch.tensor([[2., -1.], [-1., 2.]], dtype=torch.float64)
    w1 = torch.tensor([[1., -1., 0.], [0., 1., 1.]], dtype=torch.float64)
    w2 = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    ffn = positionwise_ffn(x, w1, torch.zeros(3, dtype=torch.float64), w2, torch.tensor([1., -1.], dtype=torch.float64))
    rounded = lambda value: [[round(float(z), 6) for z in row] for row in value.tolist()]
    return {
        "attention": {"weights": rounded(weights), "output": rounded(output), "masked_output": rounded(masked), "empty_output": rounded(empty), "score_gradient": rounded(backward["scores"])},
        "embedding_lookup_gradient": rounded(embedding["lookup_grad"]),
        "tied_pad_output_gradient_nonzero": bool(embedding["tied_output_grad"][0].abs().sum() > 0),
        "position_width_5_at_zero": sinusoidal_positions(1, 5)[0].tolist(),
        "ffn_outputs": rounded(ffn),
        "small_counts": parameter_counts(vocabulary=4, width=2, hidden=3, heads=1),
        "schedule_peak_step_4_width_4": round(transformer_learning_rate(4, width=4, warmup=4), 6),
        "smoothed_target_class_1": smoothed_targets(torch.tensor([1]), 4).tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(report(), sort_keys=True, separators=(",", ":")))
