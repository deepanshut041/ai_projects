## 5. Simplified self-attention: every word gets a context

We know how a decoder can retrieve a changing mixture from an encoder. Now change the relationship between the inputs: let a sentence retrieve from **itself**. Each position asks its own question and gets its own contextual vector.

![Translation retrieval uses one sequence to ask about another](images/original-slides/lecture-05-slide-02.png)

The earlier picture had two different sides: decoder queries and source memory. The new operation will supply both from the same input sequence. We keep the three-step recipe: scores, row-wise softmax, weighted sum.

![The attention map now highlights connectivity](images/original-slides/lecture-05-slide-03.png)

This is a change in **who retrieves from whom**. Self-attention does not, by itself, specify the score formula, a causal mask or the number of heads.

![Four steps: parameter-free self-attention, learned projections, causal masking and multiple heads](images/original-slides/lecture-05-slide-05.png)

Work up the staircase one addition at a time. Today we use the input vectors directly. Next we introduce separate query, key and value projections. Then we restrict visibility. Multiple heads have their own Attention Heads route after that. With the input vectors fixed, today's isolated operation has no parameters to train.

### First change the axes of the attention matrix

![Cross-sequence attention can be rectangular, while a complete self-attention pass is square](images/original-slides/lecture-05-slide-08.png)

In the left panel, two target queries consult three source positions, giving two rows and three columns. In the right panel, each position of a four-position sequence consults that same sequence, giving four rows and four columns. In both panels, normalization is across each query's source columns.

A square matrix does not by itself prove self-attention: two different sequences could have equal lengths. The defining property is input origin. Later, cached causal generation can also use a subset of query positions, so even a self-attention computation need not materialize a square matrix in every execution mode.

### Why a fixed token embedding needs context

![The pronoun it can refer to cat in one sentence and milk in another](images/original-slides/lecture-05-slide-09.png)

Compare “The cat drank the milk because it was thirsty” with “…because it was sweet”. In the first, **it** naturally refers to the cat; in the second, to the milk. A vocabulary lookup gives the token the same starting embedding in both sentences. A contextual representation can incorporate the surrounding words and distinguish the uses.

The picture motivates what a learned model should represent. The small arithmetic pass below has no trained language understanding and does not demonstrate that it resolves pronouns correctly.

![For the word big, transform its starting embedding x(2) into a context vector z(2)](images/original-slides/lecture-05-slide-10.png)

Our sentence is **Dream big and work for it**. Follow one position, **big**, from its input $x^{(2)}$ to its output $z^{(2)}$. Superscript $(2)$ means the second sequence position, not a square or a second feature coordinate. In Python it will be row index `1`.

### Write down all six input vectors

These are hand-chosen teaching coordinates, not vectors measured from a trained embedding model.

| Position | Token | Input vector |
| --- | --- | --- |
| 1 | Dream | $[0.72,0.45,0.31]$ |
| 2 | big | $[0.75,0.20,0.55]$ |
| 3 | and | $[0.30,0.80,0.40]$ |
| 4 | work | $[0.85,0.35,0.60]$ |
| 5 | for | $[0.55,0.15,0.75]$ |
| 6 | it | $[0.25,0.20,0.85]$ |

Stack them into $X\in\mathbb R^{6\times3}$. Six counts positions; three counts features at each position.

![The six three-coordinate vectors are drawn in a three-dimensional space](images/original-slides/lecture-05-slide-13.png)

Each arrow starts at the origin and ends at one input vector. Before calculating, compare directions and lengths. The picture makes the numerical geometry visible. Labels such as “content words” and “connectors” do not turn these manually chosen numbers into evidence of learned semantic clusters.

### Step 1: big scores every position, including itself

![Six dot products compare big with every source vector](images/original-slides/lecture-05-slide-14.png)

Use $x^{(2)}=[0.75,0.20,0.55]$ as the query. For source position $j$, define $\omega_{2j}=x^{(2)}\cdot x^{(j)}$. The first subscript identifies who asks; the second identifies the available source.

For Dream,

$$\omega_{21}=0.75(0.72)+0.20(0.45)+0.55(0.31)=0.8005.$$

For big itself,

$$\omega_{22}=0.75^2+0.20^2+0.55^2=0.9050.$$

For work,

$$\omega_{24}=0.75(0.85)+0.20(0.35)+0.55(0.60)=1.0375.$$

The complete score row is

$$[0.8005,0.9050,0.6050,1.0375,0.8550,0.6950].$$
 There are six scores because there are six source positions. No masking excludes big from looking at itself.

### Why can work beat big's score with itself?

![The vector magnitudes differ, with work longer than big](images/original-slides/lecture-05-slide-15.png)

The quantities in this picture are **lengths**, not the previous six attention scores. In particular,

$$\lVert x^{(2)}\rVert=\sqrt{0.905}\approx0.951315,\qquad
\lVert x^{(4)}\rVert=\sqrt{1.205}\approx1.097725.$$

A self-comparison has angle zero, but another longer vector can still score more because $q\cdot k=\lVert q\rVert\lVert k\rVert\cos\theta$. Self-alignment does not guarantee the largest raw dot product when lengths vary.

The other lengths are about 0.903881 for Dream, 0.943398 for and, 0.942072 for for, and 0.908295 for it. None was constrained to be a unit vector.

### Step 2: normalize the six scores

![A separate four-score example compares division by the sum with softmax](images/original-slides/lecture-05-slide-16.png)

Pause the six-word calculation and inspect the separate scores $[1,2,3,6]$. Dividing by their sum, 12, gives $[0.0833,0.1667,0.25,0.5]$. Softmax gives approximately

$$[0.006269,0.017040,0.046320,0.930370].$$

Here exponentiation makes the largest score much more dominant.

This is a separate four-item example, not a row with two missing words from our sentence. It exposes the difference more clearly than the gentle six-word scores.

![The interactive comparison shows every division and softmax bar for scores 1, 2, 3 and 6](images/original-slides/lecture-05-lab-normalize.png)

Both rows add to one, but only softmax handles arbitrary finite signed scores while producing positive normalized weights. Change the first score to $-1$:

![With a negative first score, direct division gives a negative weight while softmax remains positive](images/original-slides/lecture-05-lab-negative.png)

The direct sum is now ten, so its first normalized entry is $-0.1$. A signed coefficient is not a probability weight. Scores $[1,-1]$ would be worse: direct normalization divides by zero. Softmax avoids these failures by exponentiating first. This explains our choice; it is not a claim that every attention variant uses softmax.

Return to the six-word row. Using its **unscaled** scores,

$$\alpha_{2j}=\frac{\exp(\omega_{2j})}{\sum_{r=1}^{6}\exp(\omega_{2r})}.$$

| Source | Dream | big | and | work | for | it |
| --- | --- | --- | --- | --- | --- | --- |
| Score | 0.8005 | 0.9050 | 0.6050 | 1.0375 | 0.8550 | 0.6950 |
| Weight | 0.162446 | 0.180340 | 0.133599 | 0.205890 | 0.171545 | 0.146181 |

The exact computed weights sum to one. Rounded displayed entries may not. Work still receives the largest weight, about 0.20589; normalization preserves the row's ranking.

### Step 3: combine the complete input vectors

![All six embeddings contribute to big's returned context](images/original-slides/lecture-05-slide-18.png)

Compute $z^{(2)}=\sum_{j=1}^{6}\alpha_{2j}x^{(j)}$. For the first coordinate,

$$z^{(2)}_1\approx0.116961+0.135255+0.040080+0.175007+0.094350+0.036545
=0.598197.$$

Each contribution is one source's first coordinate multiplied by its weight. Do the same for coordinates two and three to obtain

$$z^{(2)}\approx[0.598197,0.343077,0.579431].$$

Use unrounded weights for the computation. Multiplying only the displayed two-decimal weights gives a slightly different answer. The context has three features; it is not the six-entry distribution of weights.

![The big query shows its score bars, weight bars and three-coordinate context together](images/original-slides/lecture-05-lab-big.png)

Follow the panels from top to bottom. Six comparison scores become six weights; those weights mix six three-coordinate vectors into **one three-coordinate result**. The number of weights and the number of output features answer different shape questions.

### Repeat the same calculation for every position

![Matrix shapes connect X, all pairwise scores, row-wise weights and every returned context](images/original-slides/lecture-05-slide-20.png)

One big row is only one sixth of the pass. All positions at once use

$$S=XX^\top,\qquad A=\operatorname{softmax}_{\rm rows}(S),\qquad Z=AX.$$

The shapes are $(6\times3)(3\times6)=6\times6$ for scores and $(6\times6)(6\times3)=6\times3$ for output. There are 36 score entries and six returned vectors. Softmax runs along the source/key axis of each row, not over all 36 entries together.

![The complete weight matrix contains one distribution per query word](images/original-slides/lecture-05-slide-21.png)

The second row is exactly the big row above. Several rows put their largest weight on work, whose magnitude affects comparisons. The and and it rows place their largest weights on themselves. The matrix describes this chosen geometry; it is not a learned linguistic result.

![An interactive heatmap lets each of the six query rows be inspected](images/original-slides/lecture-05-lab-matrix.png)

Read a row horizontally to understand one retrieval. A dark column is not a separate vocabulary prediction. Even though $XX^\top$ is symmetric, row-wise softmax generally is not: different rows use different denominators. Here $A_{12}\approx0.173758$ while $A_{21}\approx0.162446$.

![Choosing it instead of big changes its entire score row and returned mixture](images/original-slides/lecture-05-lab-it.png)

Now it is the querying position. Its weight row is approximately

$$[0.139594,0.164061,0.145509,0.180862,0.183137,0.186837],$$

and its context is $[0.568374,0.340176,0.596393]$. Nothing about the algorithm changed; we selected another input row to ask from.

### See the input-to-context movement in space

![The input embedding of big and its returned context point in different directions](images/original-slides/lecture-05-slide-22.png)

Plot $x^{(2)}=[0.75,0.20,0.55]$ and $z^{(2)}=[0.598197,0.343077,0.579431]$ from the same origin. The output lies at a weighted combination of all six endpoints, so it need not point in the original direction. Because the weights are nonnegative and sum to one, the output lies in the convex hull of the input endpoints in this parameter-free pass.

This is a weighted mixture, not a guaranteed rotation toward one semantically correct word. Changing the available vectors changes both what can be retrieved and how the comparisons are scored.

### Reproduce all numbers in Python and Colab

Complete file: [six_word_attention.py](examples/six_word_attention.py). Keep it beside `row_math.py`, which was displayed in Lesson 3. It prints every norm, score, weight and output, verifies row sums and symmetry conditions, then calculates the scaled variant separately. In Colab, upload those two files and run `!python six_word_attention.py`.

```python
"""The six-word, three-coordinate walkthrough, without trainable projections."""
import math
from row_math import dot, matmul, softmax, transpose

TOKENS = ['Dream', 'big', 'and', 'work', 'for', 'it']
INPUTS = [[0.72, 0.45, 0.31], [0.75, 0.20, 0.55],
          [0.30, 0.80, 0.40], [0.85, 0.35, 0.60],
          [0.55, 0.15, 0.75], [0.25, 0.20, 0.85]]


def attend(inputs, scale=False):
    scores = matmul(inputs, transpose(inputs))
    divisor = math.sqrt(len(inputs[0])) if scale else 1.0
    weights = [softmax([x / divisor for x in row]) for row in scores]
    return scores, weights, matmul(weights, inputs)


if __name__ == '__main__':
    scores, weights, contexts = attend(INPUTS)
    for label, vector in zip(TOKENS, INPUTS):
        print(label, 'norm', round(math.sqrt(dot(vector, vector)), 6))
    for label, matrix in [('scores', scores), ('weights', weights), ('contexts', contexts)]:
        print(label)
        for row in matrix:
            print(' '.join(f'{x:.6f}' for x in row))
    assert all(abs(sum(row) - 1.0) < 1e-12 for row in weights)
    assert all(abs(scores[i][j] - scores[j][i]) < 1e-12
               for i in range(6) for j in range(6))
    assert abs(weights[0][1] - weights[1][0]) > 1e-4
    assert abs(contexts[1][0] - 0.5981968369218149) < 1e-7
    scaled = attend(INPUTS, scale=True)
    print('scaled big weights', [round(x, 6) for x in scaled[1][1]])
    print('scaled big context', [round(x, 6) for x in scaled[2][1]])
    print('divide by sum toy', [x / 12 for x in [1, 2, 3, 6]])
    print('softmax toy', [round(x, 6) for x in softmax([1, 2, 3, 6])])
```

The same central operations in PyTorch are:

```python
import torch

X = torch.tensor([[0.72, 0.45, 0.31], [0.75, 0.20, 0.55],
                  [0.30, 0.80, 0.40], [0.85, 0.35, 0.60],
                  [0.55, 0.15, 0.75], [0.25, 0.20, 0.85]], dtype=torch.float64)
scores = X @ X.T
weights = torch.softmax(scores, dim=-1)
Z = weights @ X
print(weights[1])
print(Z[1])
```

`dim=-1` normalizes each row's source columns. It does not normalize each input feature row of $X$. This cell is a self-contained calculation you can paste into Colab.

### Keep unscaled and scaled arithmetic separate

The numerical route above uses plain dot products, so it reproduces the weight matrix and $[0.598197,0.343077,0.579431]$ context. To include the scale developed in Lesson 4, calculate instead

$$A_{\rm scaled}=\operatorname{softmax}_{\rm rows}(XX^\top/\sqrt3),\qquad Z_{\rm scaled}=A_{\rm scaled}X.$$

The big row becomes approximately

$$[0.164610,0.174847,0.147040,0.188748,0.169872,0.154883],$$

with context $[0.586353,0.349195,0.578314]$. Both are parameter-free self-attention; they are different numerical operations. Do not add $\sqrt3$ to a formula while retaining the unscaled expected outputs.

### Check additional properties with a smaller fixture

The following check preserves the compact fixture used by the earlier scoring examples and the later projection reference. It tests symmetry, zero queries and changed ordering without assigning semantic meanings to the binary coordinates.

For a compact transfer check, return to the three hand-set rows from the earlier scoring examples,

$$X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

Associate them with the positions in “Quiet cats nap” only as an indexing aid. The coordinates are synthetic and have no measured linguistic interpretation. For this first pass, use each row directly for all three roles: query, key, and returned value. Also leave scores unscaled to expose the raw dot-product calculation; scaled scores can be substituted without changing the self-attention relationship.

All pairwise scores fit into

$$S=XX^\top=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}.$$

Row $i$ asks from position $i$; column $j$ identifies the potential source position. The first score is $[1,0]\cdot[1,0]=1$, the second is $[1,0]\cdot[0,1]=0$, and the bottom-right entry is $[1,1]\cdot[1,1]=2$.

Normalize independently across each row and multiply by the value rows:

Dividing raw scores by their sum is not a suitable general replacement for softmax. Scores $[1,-1]$ have a zero sum, while $[-1,2]$ would give a negative mixing weight. Exponentiation followed by normalization gives positive weights for finite unmasked scores and preserves their ranking. It is a particular useful choice of mixture, not the only normalization ever used in attention research.

$$A=\operatorname{softmax}_{\mathrm{keys}}(S),\qquad Z=AX.$$

![The query and source axes of the weight matrix](images/05-self-matrix.svg)

The computed matrices, rounded only for display, are

$$A\approx\begin{bmatrix}
0.422319&0.155362&0.422319\\
0.155362&0.422319&0.422319\\
0.211942&0.211942&0.576117
\end{bmatrix},\qquad
Z\approx\begin{bmatrix}
0.844638&0.577681\\
0.577681&0.844638\\
0.788058&0.788058
\end{bmatrix}.$$

For example, the first output's first coordinate is $0.422319+0.422319$, because source rows 1 and 3 carry a one in that coordinate. Its second coordinate is $0.155362+0.422319$. Every input position gets an output row; there is no single pooled vector for the entire sentence unless another operation explicitly pools these rows.

#### Symmetry and shapes in the small check




$X$ is $3\times2$, $X^\top$ is $2\times3$, scores and weights are $3\times3$, and $Z$ is $3\times2$. Normalizing along the feature axis of $X$ would not produce distributions over source positions. Normalizing the columns of $S$ would answer a different question and would not ensure each query's weights sum to one.

Although $S$ is symmetric, $A$ need not be. The first and third score rows have different denominators, so $A_{13}=0.422319$ while $A_{31}=0.211942$. Softmax preserves a row's score ordering; it does not preserve arbitrary matrix symmetries.

#### Change the inputs and predict which outputs can move




Change one source row and it can affect its own query, its key in every other query's comparisons, and its value in every mixture. That is why contextualization is a collective computation. It is not a rule that only the modified token's output changes.

For a zero query `[0,0]` and three finite allowed keys, all dot products are zero and the weights are `[1/3,1/3,1/3]`. A genuine zero row is therefore meaningful input in this toy computation. Padding is a storage convention that requires an explicit mask; zeros alone do not declare a row invalid.

If we permute all input rows using the same permutation matrix $P$, then unmasked, position-free self-attention obeys

$$Z(PX)=PZ(X).$$

The score matrix becomes $PSP^\top$, row-wise softmax permutes with it, and multiplication by $PX$ produces the permuted output. This is **permutation equivariance**, not a statement that ordered output arrays are unchanged. A causal mask or positional information changes the conditions. We identify the limitation here without teaching Lecture 9's positional methods.

<!-- example:05_self_attention.py:start -->

Complete file: [05_self_attention.py](examples/05_self_attention.py).

```python
import json
from row_math import X, dot, matmul, rounded, softmax, transpose


def self_attention(inputs):
    scores = matmul(inputs, transpose(inputs))
    weights = [softmax(row) for row in scores]
    return scores, weights, matmul(weights, inputs)


if __name__ == '__main__':
    scores, weights, output = self_attention(X)
    permutation = [2, 0, 1]
    permuted = [X[i] for i in permutation]
    _, _, changed = self_attention(permuted)
    error = max(abs(changed[i][j] - output[old][j])
                for i, old in enumerate(permutation) for j in range(2))
    zero_weights = [softmax([dot([0.0, 0.0], key) for key in X])]
    assert error < 1e-14
    assert zero_weights[0] == [1 / 3] * 3
    assert weights[0][2] != weights[2][0]
    print(json.dumps({'scores': scores, 'weights': rounded(weights),
                      'output': rounded(output), 'permutation_equivariant': error < 1e-14,
                      'zero_query_weights': rounded(zero_weights[0])}))
```

<!-- example:05_self_attention.py:end -->

Run `python3 examples/05_self_attention.py`. It prints the full matrices, tests the zero query and verifies the permutation relation. The equality checks use unrounded values; displayed six-decimal row sums can differ slightly from one due to rounding.

### What is missing before the model can learn its retrieval?

![One input is transformed into separate query, key and value representations](images/original-slides/lecture-05-slide-27.png)

Today each input row filled all three roles directly: $Q=K=V=X$. With the six input vectors fixed, there is no attention parameter that can learn a different preference. A larger system could still train the upstream embedding table; that is a different claim from training projections inside this operation.

Next, three learned matrices let a token ask for information, expose comparison features and deliver content through different representations. Learning can change the mixture; it does not guarantee that every useful mixture becomes sharper.

[[practice:self-check]]

### References

[Vizuara — The Attention Family, Lecture 5](https://www.youtube.com/watch?v=vQPCjniSfNU) · [Presentation and diagrams](https://lecture5-attention-mechanisms.vercel.app/) · [Interactive six-word lab](https://lecture5-attention-lab.vercel.app/) · [Lecture Colab notebook](https://colab.research.google.com/drive/1yojm4UsLKyNvNPTmbY2qKZKMHPV0F4CZ?usp=sharing).
