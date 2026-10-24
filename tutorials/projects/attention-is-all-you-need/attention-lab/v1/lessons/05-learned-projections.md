## Learn separate query, key and value representations

Raw input attention forces each input coordinate to do three jobs. The implemented layers instead create Q, K and V using separate learned transformations. We will use the same three-token, two-feature fixture as `projection_example()` in `src/utils/attention_visuals.py`, and follow it through to retrieved values.

### Begin with the input rows and an affine map

Use `X=[[1,0],[0,1],[1,1]]`. Each row is one position with two input features. An affine projection multiplies by a weight matrix and adds a bias vector to every row. For the teaching convention:

$$Q=XW_Q+b_Q,\qquad K=XW_K+b_K,\qquad V=XW_V+b_V.$$

The bias broadcasts across positions; it is not a separately learned number for each sentence position. Q and K determine comparisons; V determines the content mixed by those comparisons. Their parameter sets are distinct, even when their shapes are equal.

The fixture uses the following right-multiplication matrices:

```text
W_Q = [[1,0],     W_K = [[1,1],     W_V = [[2,0],
       [1,1]]            [0,-1]]            [0,3]]
b_Q = [0,0]      b_K = [0,0]        b_V = [0.5,-0.5]
```

### Derive all three projected matrices

The first query row is `[1,0]@W_Q=[1,0]`; the second is `[0,1]@W_Q=[1,1]`; the third adds those rows and becomes `[2,1]`.

```text
Q = [[1,0],      K = [[1, 1],      V = [[2.5,-0.5],
     [1,1],           [0,-1],           [0.5, 2.5],
     [2,1]]           [1, 0]]           [2.5, 2.5]]
```

For V's first row, matrix multiplication produces `[2,0]`; adding `[0.5,-0.5]` gives `[2.5,-0.5]`. The same bias is added to all three rows. A negative value coordinate is valid: probabilities must be nonnegative, but representation coordinates need not be.

![The actual controlled query projection, with a highlighted row and column.](images/18-query-projection.png)

![The value projection includes a row-broadcast bias.](images/20-value-projection.png)

### Connect the written matrix to nn.Linear storage

PyTorch stores `nn.Linear` weights as `(output_features,input_features)`, and evaluates `x @ weight.T + bias`. Our displayed `W_Q` is the transpose of `layer.query.weight`. The fixture deliberately returns `projection.weight.T` for the matrix walkthrough. Confusing storage orientation with the mathematical right-multiplication convention is an easy error when both dimensions are two.

The fixture writes explicit parameters into an instance of the actual `SingleHeadAttention` class, computes the intermediate projections and checks its output against the same low-level attention operation. It constructs the temporary layer inside `torch.random.fork_rng`, so displaying a controlled example does not consume the caller's model-initialization RNG sequence.

### Use those queries and keys to obtain scores

Transpose K into columns, then multiply:

```text
K.T = [[1, 0,1],
       [1,-1,0]]
Q @ K.T = [[1, 0,1],
           [2,-1,1],
           [3,-1,2]]
```

For example, query row `[1,1]` against key row `[0,-1]` gives `1×0 + 1×(-1)=-1`. Divide each score by `√2` and apply row-wise softmax. For the first query, the scaled scores are `[1/√2,0,1/√2]`.

Let `α=exp(1/√2)≈2.028115`. Its first weight row is `[α,1,α]/(2α+1)`, approximately `[0.401112,0.197776,0.401112]`.

### Mix the projected values, not the original inputs

The first context coordinate is `(5α+0.5)/(2α+1)≈2.104448`. Its second coordinate is `(2α+2.5)/(2α+1)≈1.296664`. Work the numerator from the actual V rows:

```text
c_0[0] = a_00×2.5 + a_01×0.5 + a_02×2.5
c_0[1] = a_00×(-0.5) + a_01×2.5 + a_02×2.5
```

![Normalized weights multiply V to create context rows.](images/24-weights-times-values.png)

The two high-weight positions do not have the same values, so equal retrieval weight does not mean identical contributions in each coordinate. A contribution is weight times value. The [numerical probes](examples/arithmetic.py) check this arithmetic independently; the project visualization tests verify the fixture's Q, K, V, scores, weights and first context.

### Predict a controlled change

Hold Q and K fixed and add `[1,-2]` to every V row. Each ordinary normalized context row shifts by `[1,-2]`, because its weights sum to one. The attention weights do not change. This isolates comparison from content; changing the original X instead can affect all three projections at once.

Changing a Q/K parameter can redistribute weights and therefore affect many output coordinates. Changing a V parameter alters the available content. All three can receive gradients from a later loss, but they are not learned by this hand-set demonstration.

[[practice:projections-check]]

We now have one complete head. The next lesson shows how multiple heads share the same input positions while producing separate comparisons and contexts.
