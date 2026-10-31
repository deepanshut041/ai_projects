## 7. Establish what the floating-point reference actually does

Frozen questions are useful only if each candidate receives the same token sequence and decoding policy. We first preserve an uncompressed reference, including raw output and failures. “Floating point” names a family of representations; BF16 and FP32 have different precision and range behavior. Record the actual source, storage/compute dtype and runtime rather than treating all floating implementations as identical.

### Follow the next-token operation

After a formatted prompt, a decoder produces one logit $z_i$ per vocabulary token $i$. A logit is an unnormalized score, not a probability. Softmax turns finite scores into probabilities:

$$p_i=\frac{\exp(z_i-m)}{\sum_j\exp(z_j-m)},\qquad m=\max_j z_j.$$

Subtracting $m$ cancels from numerator and denominator, so the exact mathematical distribution is unchanged while large exponentials are avoided. For scores $[0,\log3]$, exponentials are proportional to $[1,3]$ and probabilities are $[1/4,3/4]$. Adding 1000 to both scores changes no probability mathematically. Naively evaluating `exp(1000)` would overflow ordinary finite precision; the stable form avoids that particular failure.

Greedy decoding chooses the largest logit. A deterministic tie rule still matters; this reference chooses the first index. Sampling instead draws according to a declared policy, potentially with temperature and truncation. A random seed is not a promise of identical output across kernels/devices. Our first comparison chooses greedy decoding to remove one avoidable source of variation, while still checking repeated outputs under the chosen runtime.

### Autoregression makes small shifts consequential

The model consumes the prompt, chooses a token, appends that token and calculates the next distribution conditional on the enlarged sequence. In our teaching vocabulary, token 0 is `3`, token 1 is ` hours`, and token 2 is EOS. The supplied logit sequence picks `[0,1,2]`; we return only content IDs `[0,1]` and finish reason `eos`. The logits are invented, not a tiny trained language model. A real tokenizer may divide these words differently.

If the first choice changes from “3” to “4,” later logits are evaluated on a different history. This is why low average tensor error does not guarantee identical answers. If a decode reaches its output cap first, mark `length`; do not silently treat a truncated JSON object as a valid refusal. End markers and stop policies must match the template/backend probe from Lesson 3.

### Run the mechanics without an inference framework

Create `07_reference_decode.py`. Read `softmax()` before `decode()`: the former explains probabilities, while greedy decoding can choose the maximum directly. There is no need to evaluate softmax merely to take argmax. `decode()` bounds generated content and distinguishes EOS from the length cap; a production adapter must additionally handle backend exhaustion/cancellation/errors explicitly.

[Download the complete reference](examples/07_reference_decode.py)

```python
"""Autoregressive mechanics with explicitly supplied logits; no LLM."""
import json
import math


def softmax(logits):
    maximum = max(logits)
    exps = [math.exp(x - maximum) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def decode(steps, eos, limit):
    ids, reason = [], "length"
    for logits in steps[:limit]:
        token = max(range(len(logits)), key=lambda i: logits[i])
        if token == eos:
            reason = "eos"
            break
        ids.append(token)
    return ids, reason


def main():
    probabilities = softmax([0.0, math.log(3.0)])
    shifted = softmax([1000.0, 1000.0 + math.log(3.0)])
    assert all(abs(a - b) < 1e-12 for a, b in zip(probabilities, shifted))
    # IDs 0='3', 1=' hours', 2=EOS. The model would produce each next vector.
    tokens, reason = decode([[4, 1, 0], [0, 4, 1], [0, 1, 4]], eos=2, limit=4)
    assert (tokens, reason) == ([0, 1], "eos")
    assert decode([[0, 4, 1]], eos=2, limit=1)[1] == "length"
    print(json.dumps({"probabilities": [round(x, 6) for x in probabilities],
                      "ids": tokens, "finish_reason": reason}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 07_reference_decode.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "probabilities": [
    0.25,
    0.75
  ],
  "ids": [
    0,
    1
  ],
  "finish_reason": "eos"
}
```

The probability check tests invariance under a large common shift. The token check tests EOS handling, and the one-step fixture tests truncation. These detect different mistakes. Returning EOS in answer text would indicate an output-slicing issue. Returning prompt IDs along with generated IDs would inflate answer token counts and corrupt both evaluation and timing.

### Preserve a reference before any compression

For the real pilot, retain rendered prompt bytes/token IDs, output token IDs, raw text, parsed structure, finish reason, source artifact hash, runtime build and placement. Run at least the charging question, unavailable warranty and an over-budget input. Repeat in a fresh process. A plausible answer on one question proves neither the dataset score nor support correctness; record the bounded probe precisely.

If conversion is required for the chosen runtime, compare the source-framework floating output and the converted floating artifact on a bounded agreement probe. Exact logits may differ numerically; declare the acceptance test rather than deciding after seeing differences. The main weight study should then keep the runtime and converted source fixed where possible. Cross-runtime quality references remain useful, but timing differences cannot be attributed solely to quantization.

[[practice:reference-transfer]]

The [learning-points record](project/learning-points.md) gathers these connections without inventing learner answers. We can now ask how a finite integer grid approximates the floating weights that produced this behavior.
