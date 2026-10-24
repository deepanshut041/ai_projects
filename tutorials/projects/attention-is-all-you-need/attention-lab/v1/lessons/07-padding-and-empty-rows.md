## Keep padding out of the retrieved information

Sentences can have different lengths, but a batch tensor needs a rectangular layout. Padding fills the unused positions. Even a zero input can become a nonzero key or value after a biased projection, so padding cannot be made harmless by assuming its numeric entries are zero.

### Give the mask one unambiguous meaning

Our public mask has shape `(B,S)` and uses `True` or `1` for a **valid source key**. For a five-position source with three real tokens, one row is `[True,True,True,False,False]`. The mask names positions, not embedding coordinates.

The builder converts it into `(B,1,1,S)` using `[:,None,None,:]`. Compared with scores `(B,H,T,S)`, the singleton head/query axes broadcast the same key validity across all heads and all queries in that example. Different examples can have different valid source lengths.

```text
source validity:          [1, 1, 1, 0, 0]
allowed key columns:
query 0                  [1, 1, 1, 0, 0]
query 1                  [1, 1, 1, 0, 0]
query 2                  [1, 1, 1, 0, 0]
```

![Padding excludes key columns across query rows.](images/10-padding-mask-and-weights.png)

### Block scores before normalization

Suppose a query has unmasked scores `[0,0,10]`, but the third key is padding. If we apply softmax first, the padded position takes almost all the weight. Setting that weight to zero afterwards leaves a tiny sum, not a properly normalized distribution over the valid positions.

Instead replace blocked scores with negative infinity before softmax: `[0,0,-∞]` becomes weights `[0.5,0.5,0]`. In the limit `exp(-∞)=0`, so blocked positions contribute nothing and the remaining valid keys share total mass one.

The explicit code uses `scores.masked_fill(~attention_mask, float('-inf'))`. The inversion matters because `masked_fill` receives positions to replace, whereas our incoming mask marks positions to allow. The comparison with `nn.MultiheadAttention` also inverts masks because its `key_padding_mask` marks blocked keys. Copying the same boolean tensor between these APIs without checking semantics reverses the contract.

### A key mask does not clear padded query rows

For self-attention, padded source positions are also potential query positions. The current key mask stops others from retrieving their keys/values; it does not force their own output vectors to zero. A padded query can still retrieve real keys and pass through the output projection.

That behavior is visible in the source and tests. The padding-isolation test compares real query outputs for self-attention; it does not demand zeros at padded query positions. A later translation loss must ignore padded target labels using its own policy. Key visibility and loss masking solve different problems.

### Define the entirely blocked case explicitly

For a row containing only blocked keys, ordinary softmax would see `[-∞,-∞,…]`, whose normalization is undefined. Our explicit function first detects `has_keys`, substitutes a finite score row for safe arithmetic and then replaces that row's weights with zero. Its context is zero, and the tests require finite outputs and gradients.

There is another operation after context. If the merged context is zero and `out_proj` has bias b, the final output is `0@W_O+b=b`. Therefore “zero context” does not imply “zero module output”. In the all-blocked-row test, the final output equals the output projection's bias. This distinction avoids treating a valid implementation result as a bug.

### Use a perturbation to test the information boundary

Take the same layer and valid inputs twice. In one copy add 100 to every padded input vector. The outputs at real self-attention queries must agree; cross-attention outputs must all agree when its target queries are unchanged. This tests that padded keys/values cannot influence allowed results.

```python
changed = src.clone()
changed[~mask] += 100
first = layer(src, attention_mask=mask)
second = layer(changed, attention_mask=mask)
torch.testing.assert_close(first[mask], second[mask])
```

Use evaluation mode or otherwise control randomness when comparing. With active dropout, two calls can differ even if padding is correctly excluded. Hold parameters, valid data and the query mask interpretation fixed; change only the forbidden information.

[[practice:padding-check]]

Next we add a different restriction: a decoder position must not retrieve future target positions, even when those positions contain real tokens.
