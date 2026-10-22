"""Applied arithmetic and information-flow probes for translation, MLM and QA.

This file supplies small calculations and checks against the actual reference
modules. It does not implement another model or trainer. Torch imports are lazy;
the deterministic report uses CPU float64 and restores global RNG state.
"""

from collections import Counter
import json
import math


def _torch():
    import torch
    return torch


def token_loss_and_gradient(logits, labels, *, ignore_index=-100):
    """Stable summed NLL and manual gradient of the valid-target mean.

    Class scores are on the last axis. An entirely ignored batch has zero loss
    and zero gradient. This is an arithmetic contract, not a forced optimizer
    update: the actual translation trainer rejects zero targets; MLM skips it.
    """
    torch = _torch()
    if not logits.is_floating_point() or logits.ndim < 2 or logits.shape[-1] < 1:
        raise ValueError("logits need rows and a floating class axis")
    if labels.shape != logits.shape[:-1] or labels.dtype != torch.long:
        raise ValueError("long targets must match every logit row")
    if logits.device != labels.device or not torch.isfinite(logits).all():
        raise ValueError("scores must be finite and share the target device")
    valid = labels.ne(ignore_index)
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[-1])).any():
        raise ValueError("target outside the class vocabulary")
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    log_probabilities = logits - logits.logsumexp(dim=-1, keepdim=True)
    losses = -log_probabilities.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    loss_sum = torch.where(valid, losses, torch.zeros_like(losses)).sum()
    count = int(valid.sum())
    one_hot = torch.nn.functional.one_hot(safe, logits.shape[-1]).to(logits.dtype)
    gradient = torch.where(valid.unsqueeze(-1), log_probabilities.exp() - one_hot,
                           torch.zeros_like(logits)) / max(count, 1)
    return {"loss_sum": loss_sum, "count": count, "loss": loss_sum / max(count, 1), "gradient": gradient}


def accumulated_logit_gradients(logits_batches, labels_batches):
    """Each microbatch contributes summed gradients / the SAME total count."""
    if not logits_batches or len(logits_batches) != len(labels_batches):
        raise ValueError("provide aligned nonempty microbatch lists")
    results = [token_loss_and_gradient(x, y) for x, y in zip(logits_batches, labels_batches)]
    total = sum(result["count"] for result in results)
    return [result["gradient"] * result["count"] / max(total, 1) for result in results], total


def projected_attention(query, source, wq, wk, wv, wo, *, heads=1, allowed=None):
    """Own rectangular learned projection probe; weights are [input,output]."""
    torch = _torch()
    if query.ndim != 3 or source.ndim != 3 or query.shape[0] != source.shape[0]:
        raise ValueError("streams must be [B,T,D] and [B,S,D]")
    batch, queries, width = query.shape
    sources = source.shape[1]
    if type(heads) is not int or heads < 1 or width < 1 or width % heads:
        raise ValueError("positive width must divide into the declared heads")
    if source.shape[-1] != width or sources < 1 or any(w.shape != (width, width) for w in (wq, wk, wv, wo)):
        raise ValueError("all four maps must be [D,D] and source width must match")
    tensors = (query, source, wq, wk, wv, wo)
    if any(not x.is_floating_point() or x.dtype != query.dtype or x.device != query.device or not torch.isfinite(x).all() for x in tensors):
        raise ValueError("use finite floating inputs with shared dtype/device")
    head_width = width // heads
    split = lambda x: x.reshape(batch, x.shape[1], heads, head_width).transpose(1, 2)
    q, k, v = split(query @ wq), split(source @ wk), split(source @ wv)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    if allowed.dtype != torch.bool or allowed.device != scores.device:
        raise ValueError("allowed is Boolean True=readable on the stream device")
    try:
        allowed = torch.broadcast_to(allowed, scores.shape)
    except RuntimeError as error:
        raise ValueError("allowed must broadcast to [B,H,T,S]") from error
    if not torch.isfinite(scores[allowed]).all():
        raise ValueError("allowed attention scores overflowed")
    nonempty = allowed.any(-1, keepdim=True)
    safe = torch.where(nonempty, scores.masked_fill(~allowed, -torch.inf), torch.zeros_like(scores))
    weights = safe.softmax(-1) * nonempty
    context = weights @ v
    merged = context.transpose(1, 2).contiguous().reshape(batch, queries, width)
    return {"output": merged @ wo, "weights": weights, "q": q, "k": k, "v": v, "merged": merged}


def projected_attention_backward(query, source, wq, wk, wv, wo, upstream, *, heads=1, allowed=None):
    """Manual VJP, including the two K/V paths back into source features."""
    torch = _torch()
    if (upstream.shape != query.shape or upstream.dtype != query.dtype or
            upstream.device != query.device or not torch.isfinite(upstream).all()):
        raise ValueError("finite upstream must match projected output shape/dtype/device")
    with torch.no_grad():
        result = projected_attention(query, source, wq, wk, wv, wo, heads=heads, allowed=allowed)
        batch, queries, width = query.shape
        head_width = width // heads
        d_context = (upstream @ wo.T).reshape(batch, queries, heads, head_width).transpose(1, 2)
        d_weights = d_context @ result["v"].transpose(-2, -1)
        a = result["weights"]
        d_scores = a * (d_weights - (a * d_weights).sum(-1, keepdim=True))
        merge = lambda values: values.transpose(1, 2).contiguous().reshape(batch, values.shape[2], width)
        dq = merge((d_scores @ result["k"]) / math.sqrt(head_width))
        dk = merge((d_scores.transpose(-2, -1) @ result["q"]) / math.sqrt(head_width))
        dv = merge(a.transpose(-2, -1) @ d_context)
        matrix_grad = lambda x, dy: torch.einsum("bti,btj->ij", x, dy)
        return {"query": dq @ wq.T, "source": dk @ wk.T + dv @ wv.T,
                "wq": matrix_grad(query, dq), "wk": matrix_grad(source, dk),
                "wv": matrix_grad(source, dv), "wo": matrix_grad(result["merged"], upstream)}


def learned_attention_fixture():
    """Two learned queries read three sources: probabilities have exact ratios."""
    torch = _torch()
    tensor = lambda values: torch.tensor(values, dtype=torch.float64)
    query = tensor([[[1., 0.], [0., 1.]]])
    source = tensor([[[1., 0.], [0., 1.], [1., 1.]]])
    wq = tensor([[math.sqrt(2), 0.], [0., math.sqrt(2)]])
    wk = tensor([[0., math.log(2)], [math.log(3), 0.]])
    wv, wo = tensor([[2., 0.], [0., 3.]]), tensor([[1., -1.], [0., 1.]])
    return (query, source, wq, wk, wv, wo)


def translation_parameter_budget(*, source_vocabulary, target_vocabulary, width, heads,
                                 encoder_depth=1, decoder_depth=1, ratio=4):
    """Exact trainable-parameter count for the inspected untied translation model."""
    dimensions = (source_vocabulary, target_vocabulary, width, heads, encoder_depth, decoder_depth, ratio)
    if any(type(n) is not int or n < 1 for n in dimensions) or width % heads:
        raise ValueError("positive integer dimensions and divisible head width required")
    hidden = width * ratio
    attention = 4 * width * width + 4 * width
    ffn = 2 * width * hidden + hidden + width
    embeddings = (source_vocabulary + target_vocabulary) * width
    head = width * target_vocabulary + target_vocabulary
    encoder = encoder_depth * (attention + ffn + 4 * width)
    decoder = decoder_depth * (2 * attention + ffn + 6 * width)
    return {"embeddings": embeddings, "head": head, "encoder": encoder, "decoder": decoder,
            "total": embeddings + head + encoder + decoder}


def roberta_parameter_budget(*, vocabulary, width, layers, heads, context_length, ratio=4, pad_id=1, task="mlm"):
    """Count unique parameters: the MLM decoder table is genuinely tied."""
    dimensions = (vocabulary, width, layers, heads, context_length, ratio)
    if any(type(n) is not int or n < 1 for n in dimensions) or width % heads or type(pad_id) is not int or pad_id < 0:
        raise ValueError("positive dimensions, nonnegative PAD and divisible heads required")
    hidden = width * ratio
    embedding = vocabulary * width + (context_length + pad_id + 1) * width + 2 * width
    per_layer = (4 * width * width + 4 * width) + (2 * width * hidden + hidden + width) + 4 * width
    backbone = embedding + layers * per_layer
    if task == "mlm":
        head = width * width + 3 * width + vocabulary
    elif task == "qa":
        head = 2 * width + 2
    else:
        raise ValueError("task must be mlm or qa")
    return {"embeddings": embedding, "encoder": layers * per_layer, "backbone": backbone, "head": head, "total": backbone + head}


def attention_score_elements(*, batch, heads, source_length, target_length, encoder_depth=1, decoder_depth=1):
    """Explicit score-array elements; this is not measured fused-kernel memory."""
    if any(type(n) is not int or n < 1 for n in (batch, heads, source_length, target_length, encoder_depth, decoder_depth)):
        raise ValueError("dimensions must be positive integers")
    return batch * heads * (encoder_depth * source_length ** 2 + decoder_depth * (target_length ** 2 + source_length * target_length))


def corruption_expectations(eligible_tokens, selection_probability=.15):
    """80/10/10 are conditional on selection; no forced-target policy is added."""
    if type(eligible_tokens) is not int or eligible_tokens < 0 or not 0 <= selection_probability <= 1:
        raise ValueError("nonnegative count and selection probability in [0,1] required")
    selected = eligible_tokens * selection_probability
    return {"selected": selected, "mask": .8 * selected, "random": .1 * selected,
            "unchanged": .1 * selected, "zero_selection_probability": (1 - selection_probability) ** eligible_tokens}


def controlled_mlm_probe():
    """Call the actual collator with independent selection and branch draws."""
    torch = _torch()
    from roberta_data import ByteBPETokenizer, corrupt_mlm
    tokenizer = ByteBPETokenizer()
    ids = torch.tensor([[0, 6, 7, 8, 9, 2, 1]])
    selection = torch.tensor([[0., .1, .1, .1, .9, 0., 0.]])
    branches = torch.tensor([[0., .2, .85, .95, 0., 0., 0.]])
    output, labels = corrupt_mlm(ids, tokenizer, torch.Generator().manual_seed(17), probability=.35,
                                 selection_draws=selection, replacement_draws=branches)
    selected = labels.ne(-100)
    return {"labels": labels.tolist(), "selected": int(selected.sum()),
            "mask_branch": int((selected & branches.lt(.8)).sum()),
            "random_branch": int((selected & branches.ge(.8) & branches.lt(.9)).sum()),
            "unchanged_branch": int((selected & branches.ge(.9)).sum()),
            "original_ids_unchanged": torch.equal(ids, torch.tensor([[0, 6, 7, 8, 9, 2, 1]])),
            "random_token_ordinary": bool(output[0, 2] >= 5),
            "unchanged_selected_label_preserved": bool(output[0, 3] == labels[0, 3])}


def utf8_byte_character_map(text):
    """Map every UTF-8 byte to its original half-open character interval."""
    return [span for index, character in enumerate(text)
            for span in [(index, index + 1)] * len(character.encode("utf-8"))]


def qa_endpoint_loss_and_gradient(start_logits, end_logits, starts, ends):
    """Source-style mean of two endpoint CEs, over ALL packed positions."""
    torch = _torch()
    if start_logits.ndim != 2 or start_logits.shape != end_logits.shape:
        raise ValueError("start/end scores must share [batch,positions]")
    if starts.shape != (start_logits.shape[0],) or ends.shape != starts.shape:
        raise ValueError("endpoint labels must describe every window")
    if not torch.equal(starts.ne(-100), ends.ne(-100)):
        raise ValueError("missing-window endpoints must be ignored together")
    valid = starts.ne(-100)
    if (ends[valid] < starts[valid]).any():
        raise ValueError("supervised span must have ordered endpoints")
    left, right = token_loss_and_gradient(start_logits, starts), token_loss_and_gradient(end_logits, ends)
    return {"loss": (left["loss"] + right["loss"]) / 2, "count": left["count"],
            "start_gradient": left["gradient"] / 2, "end_gradient": right["gradient"] / 2}


def valid_span_candidates(start_scores, end_scores, offsets, *, max_tokens=16):
    """List context-only ordered candidates, preserving the source's raw score."""
    if len(start_scores) != len(end_scores) or len(start_scores) != len(offsets) or max_tokens < 1:
        raise ValueError("aligned score/offset lists and a positive span limit required")
    candidates = []
    for start, left in enumerate(offsets):
        if left is None or left[0] >= left[1]:
            continue
        for end in range(start, min(len(offsets), start + max_tokens)):
            right = offsets[end]
            if right is None or right[0] >= right[1] or any(span is None for span in offsets[start:end + 1]):
                continue
            candidates.append({"start": start, "end": end, "left": left[0], "right": right[1],
                               "score": float(start_scores[start]) + float(end_scores[end])})
    return candidates


def multiset_f1(predicted_tokens, gold_tokens):
    """Independent metric arithmetic after a separately specified normalization."""
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    return 2 * overlap / (len(predicted_tokens) + len(gold_tokens))


def bleu_from_counts(matches, totals, candidate_length, reference_length):
    """Combine corpus-clipped counts; one-reference, equal weights, no smoothing."""
    if not matches or len(matches) != len(totals) or min(candidate_length, reference_length) < 0:
        raise ValueError("aligned counts and nonnegative lengths required")
    if any(type(n) is not int or n < 0 for n in [*matches, *totals]) or any(m > n for m, n in zip(matches, totals)):
        raise ValueError("clipped matches must be integer counts <= totals")
    precision = [m / n if n else 0 for m, n in zip(matches, totals)]
    brevity = 0 if not candidate_length else min(1., math.exp(1 - reference_length / candidate_length))
    score = 0 if min(precision) == 0 else 100 * brevity * math.exp(sum(map(math.log, precision)) / len(precision))
    return {"precision": precision, "brevity_penalty": brevity, "score": score}


def model_control_probe():
    """Use actual complete modules for teacher forcing and visibility controls."""
    torch = _torch()
    from translation_model import Transformer, TransformerConfig
    from roberta_model import RobertaConfig, RobertaForMaskedLM
    # Restore global RNG consumed by model construction; no learner model is read.
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(11)
        translation = Transformer(TransformerConfig(src_vocab_size=7, tgt_vocab_size=9,
                                  embedding_dimension=8, num_attention_heads=2, mlp_ratio=2,
                                  max_src_len=16, max_tgt_len=12, fused_attn=False)).double().eval()
        source, target = torch.tensor([[2, 4, 5, 3, 0]]), torch.tensor([[2, 4, 5, 6]])
        mask = source.ne(0)
        baseline = translation(source, target, mask)
        future = target.clone()
        future[:, 2:] = torch.tensor([7, 8])
        future_logits = translation(source, future, mask)
        padding = source.clone()
        padding[:, -1] = 6
        padded_logits = translation(padding, target, mask)
        changed_source = source.clone()
        changed_source[:, 1] = 6
        source_logits = translation(changed_source, target, mask)
        roberta = RobertaForMaskedLM(RobertaConfig(vocab_size=20, embedding_dimension=8,
                                   num_transformer_blocks=1, num_attention_heads=2,
                                   mlp_ratio=2, context_length=16)).double().eval()
        ids = torch.tensor([[0, 6, 7, 2, 1]])
        mlm_mask = ids.ne(1)
        initial = roberta(ids, mlm_mask)["hidden_states"]
        right = ids.clone()
        right[:, 2] = 9
        right_hidden = roberta(right, mlm_mask)["hidden_states"]
        pad = ids.clone()
        pad[:, -1] = 10
        pad_hidden = roberta(pad, mlm_mask)["hidden_states"]
        return {"translation_future_prefix_invariant": bool(torch.allclose(baseline[:, :2], future_logits[:, :2], atol=1e-12, rtol=0)),
                "translation_source_padding_invariant": bool(torch.allclose(baseline, padded_logits, atol=1e-12, rtol=0)),
                "translation_depends_on_real_source": bool((baseline - source_logits).abs().max() > 1e-10),
                "roberta_can_use_right_context": bool((initial[:, :2] - right_hidden[:, :2]).abs().max() > 1e-10),
                "roberta_tail_padding_invariant": bool(torch.allclose(initial[:, :4], pad_hidden[:, :4], atol=1e-12, rtol=0)),
                "scope": "untrained visibility controls; not held-out task quality"}


def report():
    torch = _torch()
    logits = torch.tensor([[[0., math.log(3)], [0., math.log(3)], [0., 0.]]], dtype=torch.float64)
    labels = torch.tensor([[1, 0, -100]])
    nll = token_loss_and_gradient(logits, labels)
    learned = projected_attention(*learned_attention_fixture())
    endpoint_logits = torch.tensor([[math.log(n) for n in [1, 2, 3, 4]]], dtype=torch.float64)
    endpoint = qa_endpoint_loss_and_gradient(endpoint_logits, endpoint_logits, torch.tensor([2]), torch.tensor([3]))
    rounded = lambda value: [round(float(z), 6) for z in value.flatten().tolist()]
    return {"valid_nll": {"count": nll["count"], "mean": round(float(nll["loss"]), 6), "logit_gradient": rounded(nll["gradient"])},
            "learned_rectangular_attention": {"weights": rounded(learned["weights"]), "output": rounded(learned["output"])},
            "controlled_mlm": controlled_mlm_probe(),
            "mlm_expected_of_100_at_p15": corruption_expectations(100, .15),
            "qa_endpoint": {"mean_loss": round(float(endpoint["loss"]), 6), "start_gradient": rounded(endpoint["start_gradient"]), "end_gradient": rounded(endpoint["end_gradient"])},
            "utf8_characters_for_cafe": utf8_byte_character_map("café"),
            "translation_counts": translation_parameter_budget(source_vocabulary=7, target_vocabulary=9, width=8, heads=2, ratio=2),
            "roberta_mlm_counts": roberta_parameter_budget(vocabulary=20, width=8, layers=1, heads=2, context_length=16, ratio=2),
            "visibility": model_control_probe()}


if __name__ == "__main__":
    print(json.dumps(report(), sort_keys=True, separators=(",", ":")))
