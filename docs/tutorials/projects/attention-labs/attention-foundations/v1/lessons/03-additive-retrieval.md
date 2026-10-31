## 3. Additive attention: let the decoder look back

The baseline translation machine passed one final encoder state to its decoder. Keep that machine in view while we change one interface.

![A fixed context is the only source handoff in the baseline translation machine](images/original-slides/lecture-03-slide-02.png)

The source-side limitation is the single middle arrow. We already computed intermediate encoder states; now retain them and let each decoding step choose how to combine them.

![Additive attention occupies the scoring category in the attention map](images/original-slides/lecture-03-slide-03.png)

This lesson introduces one scoring method: **additive attention**, associated with Bahdanau, Cho and Bengio's 2014 translation work. We first understand its retrieval behavior, then the full decoding step, and only then open the scoring network. This remains a recurrent encoder–decoder, not yet a Transformer.

### Start with what changes: a weighted look back

![The decoder writing chat gives the largest weight to cat](images/original-slides/lecture-03-slide-07.png)

For “The cat sat today”, imagine a step that will produce **chat**. The illustrative retrieval weights are $[0.05,0.75,0.15,0.05]$. The strongest connection is to **cat**, while all four positions contribute. These numbers are **weights**, not raw scores: they are nonnegative and sum to one.

The model does not need to know the answer “chat” before computing those weights. It already has a decoder state summarizing its previous output. That state asks the retrieval question; the retrieved context then helps predict the next token. Describing a picture as “the step writing chat” names the result of the step, not an extra input that gives away its target.

![The encoder computes a state for every source position](images/original-slides/lecture-03-slide-08.png)

Follow the same encoder chain as before, but keep every $h_j$. Each is a contextual state, not necessarily the original token embedding. In a one-directional encoder, $h_j$ summarizes a prefix ending at position $j$. Bahdanau's full system uses a bidirectional encoder, combining forward and backward annotations so each source position can carry surrounding context from both directions.

### Multiply each whole vector, then add

![Four source states contribute to a weighted sum, with the largest contribution assigned to cat](images/original-slides/lecture-03-slide-09.png)

One scalar weight multiplies every coordinate of its source state. For the displayed four-position example,

$$c_2=0.05h_1+0.75h_2+0.15h_3+0.05h_4.$$

The answer has the same width as an $h$ vector. Saying the mixture is “75% cat” means the coefficient of the state at the cat position is $0.75$. It does not mean that 75% of the coordinates are reserved for cat, or that the probability of generating chat is $0.75$. Encoder states can already contain information from several words.

![One fixed context is replaced by separate contexts c1, c2 and c3 for different output steps](images/original-slides/lecture-03-slide-10.png)

Compare the two sides. The old context was reused unchanged. The new contexts can differ because their weights differ, even though they retrieve from the same retained encoder states. Attention has preserved the useful idea of a context vector while making its construction depend on the current request.

For output step $t$, the general recipe is

$$e_{tj}=a(s_{t-1},h_j),\qquad
\alpha_{tj}=\frac{\exp(e_{tj})}{\sum_{r=1}^{N}\exp(e_{tr})},\qquad
c_t=\sum_{j=1}^{N}\alpha_{tj}h_j.$$

Here $N$ counts source positions, $s_{t-1}$ is the already available decoder state, $e_{tj}$ is a raw scalar compatibility score, and $\alpha_{tj}$ is a normalized source-position weight. The three operations are **score, normalize, combine**. They define the softmax retrieval we develop in this foundations route; later attention alternatives can change that recipe.

### Put retrieval into one complete decoding step

![Encoder states supply a changing context to a recurrent decoder that already has its previous state](images/original-slides/lecture-03-slide-13.png)

Suppose “Le” has been produced. The decoder state is $s_1$. The next context is $c_2$ because it supports the second output step. Source states and $s_1$ determine that context, and the decoder uses its available state, previous output token and retrieved information to produce a new state and vocabulary prediction.

![Every arrow in one step: source states and decoder query produce scores, weights, a context and vocabulary probabilities](images/original-slides/lecture-03-slide-14.png)

Open the middle path before following the output path:

1. Apply the same scorer $f$ to $(s_1,h_1)$, $(s_1,h_2)$, $(s_1,h_3)$ and $(s_1,h_4)$. These are repeated calls with shared parameters.
2. Normalize the four scores together. Scores $[0.3,3.0,1.4,0.3]$ give approximately $[0.05,0.75,0.15,0.05]$ after softmax.
3. Multiply the four source states by those weights and add them to obtain $c_2$.
4. Use the context and decoder features to compute vocabulary logits, normalize those logits over the French vocabulary, and choose a token according to the decoding rule.

There are **two different softmax operations**. The first distributes retrieval weight over source positions. The second distributes output probability over vocabulary entries. The diagram's $0.81$ for chat belongs to the second distribution and is an illustrative prediction, not another attention weight.

For a dimension check, a context of width 768 and a decoder feature vector of width 768 concatenate to width **1,536**. A row-vector output matrix of shape $1536\times50000$ would yield 50,000 vocabulary logits. The context does not itself need 50,000 coordinates.

The diagram gives a useful high-level output head. Exact decoder update order varies. In the original Bahdanau formulation, $s_{t-1}$ determines $c_t$, then the decoder update uses the previous state, previous target token and $c_t$; the output function consumes the required decoder/context features. Specify one order in code and avoid a circular dependency on an uncomputed state.

### The next state changes the next request

![Each recurrent decoder update incorporates a previous state, a previous output token and that step's context](images/original-slides/lecture-03-slide-15.png)

The $s$ chain carries output history forward. A state after “Le chat” differs from a state after “Le”. It therefore sends a different request to the same source memory. The source sentence does not have to be encoded again for every target token.

![The decoder loop repeats attention, context construction, prediction and state update](images/original-slides/lecture-03-slide-16.png)

Follow the loop back to the scorer. It receives the new decoder state; it does not receive a newly translated source sentence or a new set of scoring parameters. During inference, the parameters stay fixed while activations, contexts and predictions change.

![The chat step focuses on cat; the noir step focuses on black](images/original-slides/lecture-03-slide-17.png)

Now use the source **The black cat**. For the chat step, the weights are $[0.10,0.15,0.75]$, so

$$c_2=0.10h_{\rm The}+0.15h_{\rm black}+0.75h_{\rm cat}.$$

For the noir step, they change to $[0.10,0.80,0.10]$:

$$c_3=0.10h_{\rm The}+0.80h_{\rm black}+0.10h_{\rm cat}.$$

The large coefficients move from the third source position to the second. That is dynamic retrieval. These example weights illustrate desirable alignment; they are not outputs from a trained model in this project.

### Output length and order are free to differ

![Three English words can produce two French words before an end token](images/original-slides/lecture-03-slide-18.png)

“I am hungry” and “J'ai faim” have different displayed word counts. At each output step, attention can still consult all three source positions. The decoder stops on its end token; it does not stop because it has visited each source position once. A maximum output length prevents an unending decode if the model fails to stop.

![The black cat becomes Le chat noir, with crossing retrieval connections](images/original-slides/lecture-03-slide-19.png)

Read the French words in generation order. Le emphasizes The, chat emphasizes cat, and noir emphasizes black. The lines cross because the language changes noun/adjective order. A rule forcing output position $t$ to retrieve source position $t$ would prevent this behavior.

![An alignment matrix stacks one row per output step and one column per source position](images/original-slides/lecture-03-slide-20.png)

The rows are **Le, chat, noir**; the columns are **The, black, cat**. Each row sums to one. The bright cell in the second row is in the cat column, and the bright cell in the third row is in the black column. Nothing requires the columns to sum to one or the bright cells to follow the main diagonal.

The matrix is square here because these examples happen to have equal source and target lengths. Two decoding steps over three source positions would instead produce a $2\times3$ matrix. Multiplying that by a $3\times d_h$ memory returns two contexts of width $d_h$.

![The European Economic Area alignment illustrates reversed adjective order in French](images/original-slides/lecture-03-slide-21.png)

Follow **Area → zone**, **Economic → économique**, and **European → européenne**. The strong cells run away from the main diagonal because French places the modifiers differently. An alignment picture makes a learned retrieval pattern inspectable. This redrawn teaching example is not an independently reproduced training result. Attention weights also need not be perfect word alignments or a complete explanation of a model's prediction.

### Where the weights come from

![Raw scores 2, 4, 1 and 3 become normalized attention weights through softmax](images/original-slides/lecture-03-slide-23.png)

A score can be negative, positive, or larger than one; it need not sum to anything meaningful. Softmax exponentiates and normalizes the whole row. For $[2,4,1,3]$, the weights are approximately $[0.0871,0.6439,0.0321,0.2369]$. The printed lower row is schematic and does not equal the softmax of those four raw scores; use the recalculated values here for arithmetic. Their order follows the scores, all are positive, and their sum is one. A large weight can arise from a moderately sized score if the other scores are smaller: the comparisons are relative.

### Open the additive scorer

![Separate learned projections of a decoder state and source state are added, passed through tanh and reduced to one score](images/original-slides/lecture-03-slide-24.png)

For this formula, write vectors as columns:

$$a(s,h)=u^\top\tanh(W_ss+W_hh+b).$$

If $s$ has width $d_s$, $h$ has width $d_h$, and the scoring network has hidden width $d_a$, then $W_s$ has shape $d_a\times d_s$, $W_h$ has shape $d_a\times d_h$, and $u,b$ have width $d_a$. The original vectors can have different widths; their projections must have the same width so they can be added.

Follow the network in order. Each projection puts its input in the shared scoring space. **Addition** combines them, giving additive attention its name. Elementwise $\tanh$ introduces a nonlinear interaction, with outputs between minus one and one. The dot product with learned $u$ reduces the hidden vector to one scalar score. This final scoring vector is a model parameter; it is not the stored value vector that is averaged to build the context.

Why keep the nonlinearity? Without it,

$$a(s,h)=u^\top W_ss+u^\top W_hh+u^\top b.$$

For one query, the query term is the same offset for every source position. Softmax cancels common offsets, so changing only $s$ would not change the relative weights in this separable scorer. The nonlinear interaction makes the request affect relative comparisons.

### A complete small calculation

We will reuse the hand-set memory

$$H=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

The rows will also serve as illustrative token features later. For now they are three available source states. Take scalar query $s=0$, query projection $[s,-s]$, identity source projection, scoring vector $u=[1,-1]$, and zero bias. The three hidden rows and scores are

| Source state | Hidden row before tanh | Score |
| --- | --- | --- |
| `[1,0]` | `[1,0]` | $\tanh(1)-\tanh(0)=0.761594$ |
| `[0,1]` | `[0,1]` | $\tanh(0)-\tanh(1)=-0.761594$ |
| `[1,1]` | `[1,1]` | $\tanh(1)-\tanh(1)=0$ |

Softmax gives approximately `[0.593494,0.129391,0.277115]`. The returned row is

$$c=0.593494[1,0]+0.129391[0,1]+0.277115[1,1]
\approx[0.870609,0.406506].$$

The first coordinate receives contributions from rows 1 and 3; the second receives contributions from rows 2 and 3. A negative score did not create a negative weight. Exponentiation makes all unmasked weights positive.

### Compute softmax without overflowing

For scores `[1000,1000]`, directly evaluating exponentials can overflow even though the answer is simply `[0.5,0.5]`. Subtract the maximum score $m$ before exponentiating:

$$\frac{e^{e_j-m}}{\sum_r e^{e_r-m}}=\frac{e^{-m}e^{e_j}}{e^{-m}\sum_r e^{e_r}}=\alpha_j.$$

The cancellation is exact in real arithmetic and makes numerical evaluation safer. This shared reference module supplies the row operations used from here onward. `dot` multiplies matching coordinates and sums them; `matmul` applies that dot product to every row/column pair. The shared input rows are the same hand-set memory used above; there are no trainable projection parameters in this helper module.

<!-- example:row_math.py:start -->

Complete file: [row_math.py](examples/row_math.py).

```python
"""Row-vector operations and the shared synthetic fixture; standard library only."""
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Dot product widths must agree')
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError('Expected a nonempty rectangular matrix')
    return [list(column) for column in zip(*matrix)]


def matmul(a, b):
    columns = transpose(b)
    return [[dot(row, column) for column in columns] for row in a]


def softmax(scores, allowed=None):
    if not scores:
        raise ValueError('Need at least one score')
    allowed = [True] * len(scores) if allowed is None else list(allowed)
    if len(allowed) != len(scores) or not any(allowed):
        raise ValueError('Every query needs at least one allowed key')
    if any(not math.isfinite(score) for score, use in zip(scores, allowed) if use):
        raise ValueError('Allowed scores must be finite')
    maximum = max(score for score, use in zip(scores, allowed) if use)
    numerators = [math.exp(score - maximum) if use else 0.0
                  for score, use in zip(scores, allowed)]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]




def rounded(value):
    return [rounded(x) for x in value] if isinstance(value, list) else round(value, 6)


X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
```

<!-- example:row_math.py:end -->

The additive example then implements the particular scorer above. All imports refer to the displayed module in the same `examples/` folder.

<!-- example:03_additive.py:start -->

Complete file: [03_additive.py](examples/03_additive.py).

```python
import json
import math
from row_math import X, matmul, rounded, softmax


def additive(query, states):
    # Project a scalar query to [query, -query]; project each key identically.
    # The output scoring vector is [1, -1]. No biases in this fixture.
    scores = [math.tanh(query + h[0]) - math.tanh(-query + h[1]) for h in states]
    weights = softmax(scores)
    context = matmul([weights], states)[0]
    return scores, weights, context


if __name__ == '__main__':
    scores, weights, context = additive(0.0, X)
    other = additive(1.0, X)
    assert abs(sum(weights) - 1.0) < 1e-14
    assert other[1] != weights
    assert softmax([1000.0, 1000.0]) == [0.5, 0.5]
    print(json.dumps({'scores': rounded(scores), 'weights': rounded(weights),
                      'context': rounded(context), 'new_query_weights': rounded(other[1])}))
```

<!-- example:03_additive.py:end -->

Run `python3 examples/03_additive.py` from the package directory. The result includes the scores, weights and context above. It also changes the query to $s=1$, keeping the source states and parameters fixed, and obtains weights `[0.541045,0.206330,0.252626]`. The changed context comes from a new retrieval, not a new memory or a training step.

### Place the result back on the map

![The scoring timeline moves from additive attention to dot-product and scaled dot-product attention](images/original-slides/lecture-03-slide-28.png)

We have filled the additive-scoring entry: a learned projection of each input, an addition, a nonlinearity and a scalar projection. The complete surrounding operation now has a purpose: a decoder can retrieve a different source mixture at each output step.

[[practice:additive-check]]

Next we keep the retrieval pipeline and change the compatibility calculation. How cheaply can two compatible vectors produce a useful scalar score, and what happens when their feature width becomes large?

### References

[Vizuara — The Attention Family, Lecture 3](https://www.youtube.com/watch?v=6yIe4ekER14) · [Presentation and diagrams](https://lecture3-attention-mechanisms.vercel.app/).

[Bahdanau, Cho and Bengio — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/html/1409.0473v7), decoder/context equations and bidirectional annotations.
