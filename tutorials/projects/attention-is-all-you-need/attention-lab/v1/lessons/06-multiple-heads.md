## Run several heads and reassemble their information

One head uses one learned query/key comparison space and one value transformation. Multiple heads let the layer compute several such retrievals in parallel. This is architectural capacity; it does not guarantee that different heads acquire distinct linguistic specialties.

### Split projected features while keeping all positions

Choose `B=2`, `S=5`, model width `D=12` and `H=3` heads. Each head has `d=4` features. The public layer first projects X to three `(2,5,12)` tensors. `_heads` then performs:

```python
x.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
```

The intermediate shape is `(2,5,3,4)` and the result is `(2,3,5,4)`. We have separated head and feature axes, then moved the head axis before sequence. Every head still receives all five positions. We are not giving tokens 0–1 to head zero and tokens 2–3 to head one.

![Head packing separates feature groups and preserves sequence positions.](images/07-head-packing.png)

The constructor requires `D` to be positive and divisible by positive `H`. With `D=10,H=3`, equal-width heads do not fit, so this implementation raises `ValueError`. Some alternative architecture could use unequal heads; that is not the contract implemented here.

### Compare the readable loop with vectorized heads

The earlier `MultiHeadAttention` class builds separate Q/K/V linear layers for each head and runs them in a Python loop. Each projection maps the full D-dimensional input to d features. The vectorized version stores all corresponding projection rows in one `nn.Linear(D,D)`, then splits its output.

These are two organizations of compatible calculations. Stacking matching per-head parameter rows into the vectorized layer produces matching outputs and weights; the test `test_loop_heads_equal_vectorized_heads_with_matching_parameters` verifies that comparison. Comparing two separately initialized layers without copying their parameters would test different functions, not the loop/vectorization choice.

### Trace attention inside each head

For each batch/head, Q and K have shape `(5,4)`. Their score matrix has shape `(5,5)`, scaled by `√4`. Multiplying normalized weights by `(5,4)` values yields one `(5,4)` context. All heads together form `(B,H,S,d)=(2,3,5,4)`.

For output query position zero, take its four-coordinate result from head zero, then from head one, then from head two. Concatenating those features produces a 12-coordinate row. Transpose the context back to `(B,S,H,d)` before flattening the final two axes:

```python
merged = context.transpose(1, 2).flatten(2)
output = self.out_proj(merged)
```

Flattening `(B,H,S,d)` directly would combine the wrong logical axes. A shape that eventually reshapes to `(B,S,D)` is not proof that the elements retain their correct token/head ownership.

### Work a small concatenation and output projection

The separate two-head visualization uses two query positions. Head zero returns `[[1,0],[0,1]]`; head one returns `[[2,1],[1,2]]`. Concatenating along features yields:

```text
merged = [[1,0,2,1],
          [0,1,1,2]]
```

The existing display fixture applies this controlled output map, with zero bias:

```text
W_O = [[1,0,1,0],
       [0,1,0,1],
       [1,0,0,1],
       [0,1,1,0]]
merged @ W_O = [[3,1,2,2],
                [1,3,2,2]]
```

For example, the first row's third output is `1×1 + 0×0 + 2×0 + 1×1 = 2`. The operation combines head features at the same query position, rather than attending across positions again. `output_projection_example()` computes this through the actual layer's output projection, and the known-answer test verifies both merged and projected values.

![The existing fixture concatenates head contexts before the output projection.](images/25-head-context-concatenation.png)

### Count the final layer's parameters

In the final `Attention` class, Q/K/V/output are each biased D-to-D linear maps. Each owns `D²` weights and `D` biases. Total count is `4(D²+D)`. The demo uses D=9, so the count is `4(81+9)=360`. Changing H while preserving D changes how features are partitioned and the scale used per head, but not this parameter count. Attention-score storage still grows with head count and the product of query/key lengths.

[[practice:heads-check]]

The layer now handles many positions and heads. Before applying it to variable-length batches, we must ensure that added padding positions cannot become useful keys.
