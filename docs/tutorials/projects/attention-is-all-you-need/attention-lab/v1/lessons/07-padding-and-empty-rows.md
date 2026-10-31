## 6. Padding: exclude storage rows from the information flow

We have built the gathering calculation for every position and allowed several independently weighted mixtures. So far every stored source row has been treated as available information. We now return to the translator around that calculation: batching introduces stored rows that do not belong to a sentence, so the information flow needs a boundary.

The encoder will process batches of sentences. Our source has four displayed tokens; another source in the batch might have only three. A rectangular `(batch,positions,features)` tensor still needs the same stored position count for both examples, so the shorter one receives a padding row.

```text
Example A: real token | real token | real token | real token
Example B: real token | real token | real token | PAD
Validity B:    True   |    True    |    True    | False
```

The fourth row in example A is real even though our chosen vector is `[0,0]`; its validity is True. In example B, slot 3 stores padding with validity False. The validity metadata tells us which case we have; the numerical vector alone cannot.

Padding occupies memory but is not part of the source sentence. If attention treats it as another source position, it can alter the representations of real tokens. We must prevent that dependency.

### Even a zero value can contaminate an output

The problem is not solved by making the padded value zero. Its key still contributes a score and an exponential to the softmax denominator. Some weight is spent on the padded position, reducing the weights assigned to real values. The output can change even though the padded value contributes no nonzero features directly.

The right boundary is therefore **before normalization**. A boolean allowed mask states which query/key pairs exist for the calculation. This project uses `True` for allowed. For example B, `[True,True,True,False]` excludes source key 3 for every receiver.

Replace forbidden scores with negative infinity. Their exponentials are zero, leaving only allowed positions in the denominator. For receiver 0, the scaled row changes from `[0.707107,0,0.707107,0]` to `[0.707107,0,0.707107,-inf]`. The weights become `[0.401112,0.197776,0.401112,0]` and context becomes `[1.604448,1.197776]`.

Masking after softmax without renormalizing would keep the old weights on valid keys; masking the final context would be later still. Neither implements this normalized distribution over the allowed positions.

### Define what happens when nothing is allowed

Softmax of an all-negative-infinity row is undefined: subtracting its maximum encounters `-inf - (-inf)`. The project's explicit policy is zero weights and zero gathered context for a completely blocked receiver.

The small function below implements the same policy. It substitutes safe temporary scores for empty rows, performs a finite softmax, then replaces those weight rows with zeros. It also accepts a different allowed row per receiver, which we will need for decoder causality.

```python
import math
import torch

def attend(q, k, v, allowed=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    allowed = torch.broadcast_to(allowed, scores.shape)
    has_keys = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, -torch.inf)
    safe_scores = torch.where(has_keys, scores, torch.zeros_like(scores))
    shifted = safe_scores - safe_scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    weights = torch.where(has_keys, weights, torch.zeros_like(weights))
    return weights @ v, weights

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
valid_keys = torch.tensor([[True, True, True, False]])
context, weights = attend(x, x, 2 * x, valid_keys)
print(weights[0])
print(context)

changed = x.clone()
changed[3] = torch.tensor([1000., -1000.], dtype=x.dtype)
changed_context, _ = attend(changed, changed, 2 * changed, valid_keys)
assert torch.allclose(changed_context[:3], context[:3])

no_keys = torch.zeros((1, 4), dtype=torch.bool)
empty_context, empty_weights = attend(x[:1], x, 2 * x, no_keys)
print(empty_context)
assert torch.equal(empty_weights, torch.zeros_like(empty_weights))
```

For example B with source position 3 masked, contexts are `[[1.604448,1.197776],[1.197776,1.604448],[1.503490,1.503490],[1.333333,1.333333]]`. The final empty-memory check returns `[[0,0]]`.

### Prove the excluded row cannot affect real receivers

The program changes the padded row to `[1000,-1000]`, then recomputes attention. The first three outputs must stay unchanged. This checks the intended absence of influence, not merely the presence of a triangular picture or a zero in a printed tensor.

Only the first three receivers are compared because changing X[3] also changes query 3. Masking **key 3** removes that row as a source of information; it does not freeze or delete its query. The original fourth context `[1.333333,1.333333]` is therefore consistent: the padded query still reads the three real keys.

Keep three responsibilities separate. A key mask excludes source contributions. A target-loss mask later excludes padded prediction positions from the objective. An output projection can add its bias even when gathered context is zero. Confusing those boundaries leads to tests that look reassuring while checking the wrong behavior.

**Runnable copy:** [padding, perturbation and empty-row checks](examples/walkthrough/06_padding.py).

[[practice:padding-check]]
