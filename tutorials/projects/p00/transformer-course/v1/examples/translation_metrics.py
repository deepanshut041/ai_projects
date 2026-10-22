"""Clipped corpus n-gram precision and a reproducible metric adapter."""

from collections import Counter
import math


def ngrams(tokens, order):
    return Counter(tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1))


def corpus_bleu_tokens(predictions, references, *, max_order=4):
    """One reference per sentence, no smoothing, score on a 0..100 scale."""
    if len(predictions) != len(references) or not predictions or max_order < 1:
        raise ValueError("BLEU needs aligned nonempty corpora and a positive n-gram order")
    matches, totals = [0] * max_order, [0] * max_order
    candidate_length = reference_length = 0
    for prediction, reference in zip(predictions, references):
        candidate_length += len(prediction)
        reference_length += len(reference)
        for order in range(1, max_order + 1):
            proposed, expected = ngrams(prediction, order), ngrams(reference, order)
            matches[order - 1] += sum((proposed & expected).values())
            totals[order - 1] += sum(proposed.values())
    precision = [m / n if n else 0.0 for m, n in zip(matches, totals)]
    brevity = 0.0 if candidate_length == 0 else min(1.0, math.exp(1 - reference_length / candidate_length))
    score = 0.0 if min(precision) == 0 else 100 * brevity * math.exp(sum(math.log(p) for p in precision) / max_order)
    return {"score": score, "matches": matches, "totals": totals,
            "precisions": [100 * p for p in precision], "brevity_penalty": brevity,
            "candidate_length": candidate_length, "reference_length": reference_length,
            "settings": f"one-reference|tokenized|order:{max_order}|smooth:none"}


def sacrebleu_corpus(predictions, references):
    """Use raw decoded strings and return the library's scoring signature."""
    from sacrebleu.metrics import BLEU
    metric = BLEU(tokenize="13a", lowercase=False, smooth_method="exp")
    result = metric.corpus_score(predictions, [references])
    return {"score": result.score, "signature": str(metric.get_signature())}


def main():
    print(corpus_bleu_tokens([["je", "vois", "les", "chats"]],
                             [["je", "vois", "les", "chats"]]))
    repeated = corpus_bleu_tokens([["chat", "chat", "chat"]], [["chat", "est", "ici"]], max_order=1)
    print("repetition clipped", repeated["matches"], repeated["totals"])


if __name__ == "__main__":
    main()
