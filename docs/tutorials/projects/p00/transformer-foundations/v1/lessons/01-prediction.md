## Lesson 1 — Linear algebra and neural maps

**Goal:** read and calculate the maps that appear throughout a Transformer. Prerequisite: comfortable Python programming. No previous deep-learning course is assumed. The [course outline](project/course-outline.md) gives the topic order; [study notes](project/study-notes.md) provide a personal workbook. Runnable reference files live in this package's `examples/` directory. Code blocks with imports are standalone probes; blocks labelled as implementation exercises belong in the existing learner lab.

### Arrays, coordinates and notation

A scalar is one number. A vector is an ordered list of coordinates; its order matters. A matrix has two axes, and a tensor generalizes this to any number of axes. Rank counts axes; shape gives their lengths. An array shaped `[2,3,4]` contains 24 numbers. It is not automatically a batch or a sequence: those meanings come from our contract.

We use row vectors. `X[N,D]` means N examples with D features each. In later lessons `X[B,T,D]` means batch, position, feature. A subscript is an index: xᵢ is a coordinate, not multiplication by i. Σ means addition over the stated index. All numerical indices here start at zero.

The dot product of equal-width vectors is

$$x\cdot w=\sum_{i=0}^{D-1}x_iw_i.$$

For `[1,2]` and `[2,3]` it is `1×2+2×3=8`. Multiplying corresponding coordinates gives a vector; summing those products gives a scalar. Dot products can be positive, negative or zero. Their magnitude depends on both vector lengths and direction, so they are not automatically normalized similarity scores.

The Euclidean norm is $\|x\|_2=\sqrt{\sum_i x_i^2}$. For `[3,4]`, it is 5. Cosine similarity divides a dot product by both norms and needs a policy for a zero vector. Transformer attention uses scaled dot products; it does not generally normalize them to cosine similarity.

### Matrix multiplication is a collection of dot products

Let

$$X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
W=\begin{bmatrix}1&0&2\\0&1&3\end{bmatrix},\quad b=[0,0,1].$$

The weights have one column per output recipe. Compute

$$\begin{aligned}
Y_{nf}&=\sum_{d=0}^{D-1}X_{nd}W_{df}+b_f,\\
Y&=XW+b=\begin{bmatrix}1&2&9\\3&4&19\end{bmatrix}.
\end{aligned}$$

For example, `Y[1,2]=3×2+4×3+1=19`. Shapes `[N,D]@[D,F]→[N,F]` name the contracted dimension D. Matrix multiplication is not commutative: W X may be undefined even when X W exists. Transpose exchanges axes: $(XW)^T=W^TX^T$. It does not reverse the order of values within each row.

[[diagram:prereq-linear]]

One set of weights and biases is reused for all examples. A batch of 10,000 examples does not need 10,000 independent weight matrices. The number of learned parameters here is `D×F+F=9`; the two input rows are data, not parameters. A dense multiply uses about NDF multiply–accumulate operations. When counting FLOPs as separate multiplication and addition, the leading count is approximately 2NDF; state the convention.

### Linear versus affine

A linear map preserves addition and scalar multiplication and sends zero to zero. `xW+b` is affine when b is nonzero. A library's “linear layer” usually includes a bias. That is useful: the targets `x=0→1` and `x=1→3` are fitted by `2x+1`, while `wx` cannot fit the first target.

PyTorch's `nn.Linear(D,F)` stores weights `[F,D]`, and calculates `X @ layer.weight.T + layer.bias`. Our mathematical W is `[D,F]`. Both conventions describe the same map; never transpose twice to make a shape error disappear. Its operation on `[B,T,D]` produces `[B,T,F]`, reusing parameters at each position. This operation transforms features independently; it does not mix tokens.

### Implementation exercise — the existing linear() contract

Work in `ai_projects/projects/transformer-labs/transformer-foundations-lab/`, inspect the existing `src/transformer_foundations_lab/numerics.py`, and preserve any edits. Implement or complete `linear(rows, weights, bias)` using the row convention above. Require nonempty rectangular rows/weights, matching contracted widths, and one bias per output. Raise `ValueError` for an invalid shape rather than silently truncating a `zip`.

A complete reference is [prerequisite_math.py](examples/prerequisite_math.py); attempt the exercise before reading its implementation. From the package directory, the arithmetic reference is runnable without Torch:

```bash
python3 examples/prerequisite_math.py
python3 -m unittest discover -s examples -p test_prerequisite_math.py
```

Known answers are `[[1,2,9],[3,4,19]]`. Replacing the second input row with `[4,4]` changes only the second output row to `[4,4,21]`. A zero input gives b. Do not interpret a passing author reference as completion of your own learner exercise.

[[practice:linear-transfer]]

### Failure cases and notes

A legal multiply can still join the wrong axes. Using equal dimensions hides mistakes, so debug with unequal values for N, D and F. An elementwise product is not a matrix multiply. Bias belongs to output features; a bias per example would depend on batch organization.

In my notes: write the indexed affine formula, calculate every cell of one result row, and label which numbers are data and which are shared parameters. Explain how the same map will produce Q, K and V in attention. Stop when you can derive the output and parameter count without running the reference.

Source contract: [PyTorch Linear](source:torch-linear). The arithmetic, exercise and changed-case checks are original. Continue with [tensor contracts](lessons/02-workspace.md).
