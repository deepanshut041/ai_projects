## 8. Multi-head attention

The Attention Foundations route ended with one causal attention head: projected queries and keys select a weighted mixture of projected values, with future positions excluded. Here we change the arrangement of those projections. Several heads build separate mixtures, then an output projection combines their results.

This folder owns head and key/value layouts. This page develops multi-head attention. The following separate lessons develop MQA, GQA and MLA in this same Attention Heads category.

![The previous causal pipeline supplies the component that each head will repeat](images/original-slides/lecture-08-slide-02.png)

![Heads and key/value organization form the third category in the attention map](images/original-slides/lecture-08-slide-03.png)

![The staircase reaches multiple heads after projections and causal masking](images/original-slides/lecture-08-slide-05.png)

This lecture has three passes: why several mixtures help, how a loop implements them directly, and how a packed tensor implementation preserves their operation. All three passes stay together in this MHA tutorial. MQA, GQA and MLA each have their own complete tutorial in Attention Heads.

### What a second head changes

![The last it can be related to its noun, acting verb and clause through different information patterns](images/original-slides/lecture-08-slide-08.png)

In “The dog chased the ball, but it could not catch it”, the final it refers to the ball and participates in the catch relation. These are possible information needs, not named responsibilities assigned to head numbers. One mixture may combine useful evidence about several things; several learned mixtures offer additional separately parameterized channels.

![Independent projected attention heads produce separate outputs before concatenation](images/original-slides/lecture-08-slide-09.png)


Follow the input into both projection branches, then keep the two resulting attention patterns distinct until their outputs are joined. Each general projection can use every input feature; this is not a requirement to divide the raw embedding into disjoint slices. Separate parameters allow different retrievals but do not guarantee linguistic specialization.

For one head, every value-feature column is mixed using the same matrix $A$:

$$Z=AV.$$

A single query row can distribute weight over several source positions. It is therefore too strong to say that one head can attend to only one token or represent only one relationship. Its actual restriction is that one distribution is shared across the value-feature channels of that head.

For heads $r=1,\ldots,H$, use separate parameter matrices:

$$Q_r=XW_{Q,r},\quad K_r=XW_{K,r},\quad V_r=XW_{V,r},$$
$$Z_r=\operatorname{softmax}\left(Q_rK_r^\top/\sqrt{d_k}+M\right)V_r.$$

The different $A_r$ matrices can route information differently. Separate parameters permit this; they do not guarantee interpretable roles or prevent redundant heads. Labels such as “syntax head” require analysis of an actually trained model, not an assumption based on head number.

![Independent attention patterns are concatenated before output projection](images/01-heads.svg)

![Three illustrative causal patterns emphasize the previous word, first-word anchor and selected content words](images/original-slides/lecture-08-lab-patterns.png)

Read one row across all three matrices. They retrieve different mixtures from the same sentence, while sharing the same causal restriction. These are designed examples of possible patterns, not measurements from a trained model in this project.

![Random head patterns need not have clean linguistic interpretations](images/original-slides/lecture-08-lab-random.png)

Separate initial parameters can produce different patterns, but meaningful specialization requires evidence about training and behavior. Heads can overlap or become redundant too. Training does not guarantee that every head becomes an easily named grammar detector.

### Count tokens and feature widths separately

For the six-token, width-six example, each of two heads projects from all six input features into three comparison features and three value features. Each per-head projection matrix has shape $6\times3$; each projected Q/K/V and head output has shape $6\times3$. Concatenate the **features of matching token rows** to obtain $6\times6$. Six tokens remain six tokens.

The output projection then uses a $6\times6$ matrix to combine channels inside each row. With twelve tokens and the same widths, the input and output would become $12\times6$, while the output projection would remain $6\times6$: its size depends on feature widths, not sequence length.

For six tokens, width twelve and four equal-width heads, each head returns $6\times3$; concatenation returns $6\times12$. Familiar model configurations use the same arithmetic: GPT-2 small has width $768=12\times64$, and the largest GPT-3 configuration has width $12288=96\times128$. These are architecture examples, not required sizes for this project.

### A small example with two genuinely different patterns

Keep the three input rows from the foundations route:

$$X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

Each of two heads now has comparison width one and value width one. Head 1 selects the first input coordinate for queries and keys, and returns the sum of the two coordinates as its scalar value. Head 2 selects the second coordinate for queries and keys, and returns the first coordinate minus the second as its value.

| Head | $W_Q$ | $W_K$ | $W_V$ |
| --- | --- | --- | --- |
| 1 | `[1,0]` as a column | `[1,0]` as a column | `[1,1]` as a column |
| 2 | `[0,1]` as a column | `[0,1]` as a column | `[1,-1]` as a column |

For the final position, Head 1 has query 1 and keys `[1,0,1]`, so its weight row is approximately `[0.422319,0.155362,0.422319]`. Head 2 has query 1 and keys `[0,1,1]`, yielding `[0.155362,0.422319,0.422319]`. Scaling divides by $\sqrt1$, so it makes no numerical change here.

The value lists are `[1,1,2]` and `[1,-1,0]`. Their final-position context scalars are about `1.422319` and `-0.266956`. Concatenation puts those two outputs next to each other:

$$z_{\mathrm{joined},3}=[1.422319,-0.266956].$$

At the first position only the first key is allowed. Its two head outputs are `[1]` and `[1]`, so concatenation gives `[1,1]`. The head arrangement changes neither the causal permission rule nor the meaning of the sequence axis.

### Concatenation preserves the separate channels; projection mixes them

![A learned output projection combines concatenated head features within each position](images/original-slides/lecture-08-slide-10.png)


Concatenation preserves the separate head channels; the following matrix can combine those channels into the layer’s output width. The output projection acts within a token row. It does not create a new attention distribution across tokens, and it is not required to mix every feature if the learned parameters preserve some channels.

Use the row-vector output matrix

$$W_O=\begin{bmatrix}1&0.5\\-0.5&1\end{bmatrix},\qquad
Y=\operatorname{Concat}(Z_1,Z_2)W_O.$$

The first position becomes $[1,1]W_O=[0.5,1.5]$. At the final position,

$$Y_3=[1.422319+0.133478,\ 0.711159-0.266956]
\approx[1.555797,0.444203].$$

The output projection acts on feature channels **within each token**. It does not create a new token-to-token attention distribution; the cross-token mixing already happened inside the heads. It can combine, rotate, suppress or preserve head features according to learned parameters.

Concatenation order is a consistent bookkeeping choice, not an automatic priority ranking. If we swap the two joined channels and swap the matching rows of $W_O$, the final output remains the same. The output projection supplies learned combinations of head features; it need not scramble every coordinate, and an identity projection would preserve their separation.

<!-- example:multihead.py:start -->

Complete file: [multihead.py](examples/multihead.py).

```python
"""Two independent scalar heads, with all helpers defined here."""
import json
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Mismatched widths')
    return sum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[dot(row, column) for column in zip(*b)] for row in a]


def head(x, wq, wk, wv):
    q, k, v = matmul(x, wq), matmul(x, wk), matmul(x, wv)
    outputs, patterns = [], []
    for i, query in enumerate(q):
        scores = [dot(query, key) / math.sqrt(len(key)) for key in k[:i + 1]]
        maximum = max(scores)
        numerators = [math.exp(s - maximum) for s in scores]
        weights = [s / sum(numerators) for s in numerators]
        outputs.append(matmul([weights], v[:i + 1])[0])
        patterns.append(weights + [0.0] * (len(x) - i - 1))
    return outputs, patterns


if __name__ == '__main__':
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    h1, a1 = head(x, [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [1.0]])
    h2, a2 = head(x, [[0.0], [1.0]], [[0.0], [1.0]], [[1.0], [-1.0]])
    joined = [first + second for first, second in zip(h1, h2)]
    output = matmul(joined, [[1.0, 0.5], [-0.5, 1.0]])
    def rounded(matrix):
        return [[round(v, 6) for v in row] for row in matrix]
    assert a1[-1] != a2[-1]
    assert output[0] == [0.5, 1.5]
    print(json.dumps({'head1_weights': rounded(a1), 'head2_weights': rounded(a2),
                      'concatenated': rounded(joined), 'output': rounded(output)}))
```

<!-- example:multihead.py:end -->

From `ai_projects/docs/tutorials/projects/attention-labs/attention-heads/v1/`, run:

```bash
python3 examples/multihead.py
```

The script is self-contained and prints both full head-weight matrices, the concatenated outputs and the final outputs. It checks that the last-row distributions differ. The weights are hand-set snapshots, so no claim about learned linguistic specialization follows from these results.

### Dimensions express a design choice

In general, $X$ has width $D$, each head returns width $d_v$, concatenation has width $Hd_v$, and $W_O$ can have shape $Hd_v\times D_{\mathrm{out}}$. A common Transformer choice sets $D_{\mathrm{out}}=D$ and $d_k=d_v=D/H$. That produces the familiar relation $D=H d_k$.

This relation is a convention enforced by our packed implementation, not a universal mathematical requirement for attention. Other implementations can use different projection widths. At fixed $D$ with the conventional split, increasing $H$ makes each head narrower rather than multiplying the total projection width by $H$.

Each general head projection can use **all $D$ input features**. Its $D\times d_k$ query matrix produces a smaller set of comparison features from that entire input row. Our sparse hand-set matrices make the arithmetic small; they are not a rule to cut the raw input into disjoint slices. The packed implementation splits the outputs of learned projections into heads.

### First implement the picture literally: one module per head

![A list of independent causal heads receives the input, then the outputs are concatenated](images/original-slides/lecture-08-slide-13.png)

`nn.ModuleList` registers the head modules so their parameters are included in optimizer discovery, device moves and checkpoint state. It does not automatically run them in parallel. The list comprehension in `forward` explicitly calls each head, then `torch.cat(..., dim=-1)` joins their output features. Each head has its own Q/K/V projections; the inputs can still be the same tensor.

Complete file: [head_loop.py](examples/head_loop.py). It contains the head definition before the wrapper that calls it, and runs a `[2,6,6]` example. Paste the whole file into Colab or upload it and run `!python head_loop.py`.

```python
"""An explicit ModuleList of independent causal heads, followed by output mixing."""
import math
import torch
from torch import nn


class CausalHead(nn.Module):
    def __init__(self, width, head_dim, context_len, dropout=0.0):
        super().__init__()
        self.query = nn.Linear(width, head_dim, bias=False, dtype=torch.float64)
        self.key = nn.Linear(width, head_dim, bias=False, dtype=torch.float64)
        self.value = nn.Linear(width, head_dim, bias=False, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('future', torch.ones(context_len, context_len,
                             dtype=torch.bool).triu(1))

    def forward(self, x):
        length = x.shape[-2]
        if not 0 < length <= self.future.shape[0]:
            raise ValueError('Input is outside the prepared context length')
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        weights = scores.masked_fill(self.future[:length, :length], -torch.inf).softmax(-1)
        return self.dropout(weights) @ v


class LoopMHA(nn.Module):
    def __init__(self, width=6, heads=2, context_len=6, dropout=0.0):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('This layout requires a positive width divisible by heads')
        self.heads = nn.ModuleList([
            CausalHead(width, width // heads, context_len, dropout) for _ in range(heads)])
        self.output = nn.Linear(width, width, bias=False, dtype=torch.float64)

    def forward(self, x):
        joined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.output(joined)


if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.randn(2, 6, 6, dtype=torch.float64)
    layer = LoopMHA().eval()
    print('input shape', list(x.shape))
    print('each head shape', [list(head(x).shape) for head in layer.heads])
    print('output shape', list(layer(x).shape))
    print('registered parameter count', sum(p.numel() for p in layer.parameters()))
    assert layer(x).shape == x.shape
    assert sum(p.numel() for p in layer.parameters()) == 144
```

The output reports two head tensors of shape `[2,6,3]` and a final tensor of shape `[2,6,6]`. The batch dimension indexes sequences, not heads. Distinct inputs in the two batch items also help expose an accidental mixing of the batch axis. With no biases, each head has $3(6\times3)=54$ projection parameters; two heads plus the $6\times6$ output matrix give $108+36=144$ registered parameters.

### Why replace the head loop?

![A timing illustration compares loops with two, four hundred and four thousand heads](images/original-slides/lecture-08-slide-14.png)

Read the head-count labels and runtime units together. The illustration shows roughly a millisecond for two heads, 0.08 seconds for four hundred, and 0.9 seconds for four thousand in its demonstrated environment. These values are not a runtime promise for another machine.

The large-count demonstration keeps each head's width fixed at three while adding more heads. It therefore increases total projection width, parameter count and arithmetic as well as loop dispatch. It is not a controlled comparison at fixed model width. The 4,000-head case is an exaggerated workload used to expose the issue, not our architecture choice.

![A browser race shows actual measured times for separate and combined projections](images/original-slides/lecture-08-lab-race.png)

This small browser run even makes the loop faster than the combined version. That is useful: fewer calls do not prove a speedup for every backend and shape. Python dispatch, kernel launches, array layout, device utilization and synchronization all matter. GPU work can be queued asynchronously; a host loop does not mean every GPU operation is forced to finish before the next launch. The next pass proves that packing preserves the calculation; a performance claim would additionally require a matched, synchronized benchmark.

[[practice:heads-check]]

The optional [transfer review](practice/transfer-review.md) tests feature mixing, equal-size axis mistakes and the difference between packing and sharing.

### Pack independent heads without changing the operation

The explicit loop above computed each head separately. A tensor implementation can place the head axis beside the batch axis, allowing a numerical backend to evaluate the heads through batched operations. Packing changes representation and execution; it need not change the parameters, outputs or derivatives.

### Combine the projections first

![One wide projection contains the output slices of multiple attention heads](images/original-slides/lecture-08-slide-16.png)


The packed matrix places the independent per-head projection columns together. Splitting its computed output recovers those head features without replacing the parameters with shared weights. This is why the equality below can be checked using one parameter set rather than comparing separately initialized layers.

For two heads with query matrices $W_{Q,1}$ and $W_{Q,2}$, concatenate their columns:

$$W_Q=[W_{Q,1}\ W_{Q,2}],\qquad
XW_Q=[XW_{Q,1}\ XW_{Q,2}].$$

This equality follows directly from matrix multiplication. The same construction applies separately to keys and values. Some implementations combine Q, K and V into one even wider projection; our reference keeps three linear layers so the three roles stay visible.

The packed layer does not share parameters between heads. Each head still selects its own contiguous output slice of the larger matrix. A grouped-query implementation would intentionally share some key/value information; that would be a mathematical change, not just this packing transformation.

### Track the axes through a real example

![The full multi-head tensor route labels batch, token, head and feature dimensions](images/original-slides/lecture-08-slide-18.png)


Read the axis names at each transition before trusting the shape alone. We first follow the six-token, two-head route completely, then use the smaller checked fixture to make the equivalence arithmetic inspectable.

### Follow the full six-token shape journey

Let $B=2$, $T=6$, $D=6$, $H=2$, and $d_h=3$. The first two leading quantities count different things: two **sequences** and six **tokens per sequence**. The last six counts features. Repeating the same sequence twice is one possible batch, but the implementation must also work for two different sequences.

![Stage 1: the input has batch, token and feature axes](images/original-slides/lecture-08-lab-stage-1.png)

**1. Input `[2,6,6]`.** A row is a token representation. No head axis exists yet. The final block output will have this shape too, because this conventional configuration returns the model width.

![Stage 2: a wide projection produces all head features together](images/original-slides/lecture-08-lab-stage-2.png)

**2. Projection `[2,6,6]`.** One $6\times6$ query matrix produces all query features. Its first three output columns belong to head 1 and its last three to head 2. Make the same kind of projection separately for keys and values. This is a feature partition **after projection**, not a restriction on which original input features a head can use.

![Stage 3: view splits the last width into head count and feature width](images/original-slides/lecture-08-lab-stage-3.png)

**3. Split `[2,6,2,3]`.** `view(B,T,H,d_h)` exposes the head grouping. The total element count stays $2\times6\times6=72$. No new numbers are learned or calculated by this view.

![Stage 4: move heads before tokens so each head receives a token matrix](images/original-slides/lecture-08-lab-stage-4.png)

**4. Transpose `[2,2,6,3]`.** `transpose(1,2)` swaps token and head axes. Every batch/head pair now has a $6\times3$ query matrix. There are **four** such pairs, not only two: two sequences times two heads.

![Stage 5: each batch/head pair produces a six-by-six score matrix](images/original-slides/lecture-08-lab-stage-5.png)

**5. Scores and weights `[2,2,6,6]`.** Multiply Q by `K.transpose(-2,-1)`. The contracted feature width is three; the remaining last axes are query token and source token. Scale by $\sqrt3$, apply the causal mask, then softmax across the final source-token axis. Masking, scaling and normalization preserve shape. Training-time dropout, if used, also preserves shape.

![Stage 6: source-position weights mix each head's value vectors](images/original-slides/lecture-08-lab-stage-6.png)

**6. Contexts `[2,2,6,3]`.** Multiply each $6\times6$ weight matrix by its $6\times3$ values. There are still four batch/head matrices, each with six token rows and three value features.

![Stage 7: transpose back to put each token's head outputs beside each other](images/original-slides/lecture-08-lab-stage-7.png)

**7. Restore `[2,6,2,3]`.** Transpose token and head axes back. Now each token owns two adjacent head groups. Repeating an in-place reassignment such as `context = context.transpose(1,2)` in a notebook cell swaps them again, which can undo the intended layout. Inspect the shape at each stage or recompute from named intermediate tensors.

![Stage 8: merge head features, then apply the output projection](images/original-slides/lecture-08-lab-stage-8.png)

**8. Merge and mix `[2,6,6]`.** Combine head and within-head feature axes, preserving their order. The $6\times6$ output projection then mixes feature channels within each token. These are two operations even though their input and output shapes agree.

The scale is the per-head width $d_h=3$. Using the combined width six would alter the softmax, and using the head count two would use a different quantity again.

### Relate the full route to the compact numerical check

Use batch size $B=2$, sequence length $T=3$, model width $D=2$, head count $H=2$, and head width $d_h=1$. The first sequence reuses the earlier three-row input. A second, different sequence verifies that the batch axis remains independent.

![Projection, head split, attention, and head merge](images/02-packed-shapes.svg)

| Stage | Shape | Meaning of the last two axes |
| --- | --- | --- |
| Input and full-width Q/K/V projections | `[2,3,2]` | token, combined head features |
| Reshape after projection | `[2,3,2,1]` | head, feature within head |
| Transpose token and head axes | `[2,2,3,1]` | token, feature within head |
| Q times transposed K | `[2,2,3,3]` | query token, source token |
| Weighted values | `[2,2,3,1]` | token, value feature within head |
| Transpose back | `[2,3,2,1]` | head, value feature within head |
| Merge the last two axes | `[2,3,2]` | token, concatenated features |
| Output projection | `[2,3,2]` | token, output features |

A reshape reinterprets a projected width as two axes. The transpose then determines which axes participate in matrix multiplication. Batched matrix multiplication contracts the last dimension of Q with the feature dimension of K and treats the earlier axes as batch-like axes.

After the head split but **before** that transpose, the last two axes are heads and features. Multiplication there produces a head-by-head matrix for each token, which is the wrong relationship. We need token-by-token comparisons separately for each head, so queries and keys must have shape `[B,H,T,d_h]` first.

### Transpose the right axes, then merge deliberately

![Moving the head axis before the token axis makes attention operate within each head](images/original-slides/lecture-08-slide-17.png)


The transpose positions batch and head as the leading batch-like axes, leaving token and feature in the last two axes. Matrix multiplication can then compare tokens within one head. Reversing that transpose before merging is equally necessary: having the right number of elements does not establish the right token/head ordering.

Use `k.transpose(-2,-1)` to change `[B,H,T,d_h]` into `[B,H,d_h,T]`. A plain `.T` on a tensor with more than two axes does not express this intended operation. The causal mask has shape `[T,T]` and broadcasts over batch and heads; future weights are zero in every head.

After value mixing, transpose `[B,H,T,d_h]` to `[B,T,H,d_h]` before merging heads. A tensor transpose often produces a noncontiguous view. Our reference explicitly calls `.contiguous()` and then reshapes, so the flattened feature order matches the concatenation in Lesson 1. A shape that merely has the expected number of elements does not prove that its token/head ordering is correct.

### The complete packed layer and independent loop

The following reference needs PyTorch. The scalar reference above remains dependency-free. This program uses CPU float64, explicit parameter assignment and no data downloads. It contains a packed layer and a separate implementation that loops over slices of the **same** parameters. Comparing independently initialized modules would not test packing equivalence.

PyTorch `nn.Linear` stores weights as `[out_features,in_features]` and applies their transpose to row-vector inputs. For example, the row-vector $W_O$ from Lesson 1 is stored transposed in `model.output.weight`. The assignment below accounts for that convention.

<!-- example:multihead_torch.py:start -->

Complete file: [multihead_torch.py](examples/multihead_torch.py).

```python
"""CPU float64 correctness probe, not a performance benchmark or trained GPT."""
import json
import math
import torch
from torch import nn


class PackedMHA(nn.Module):
    def __init__(self, width=2, heads=2):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('Positive width must be divisible by positive head count')
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x):
        batch, tokens, width = x.shape
        def split(layer):
            projected = layer(x)
            return projected.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.query), split(self.key), split(self.value)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        forbidden = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(forbidden, -torch.inf).softmax(dim=-1)
        contexts = weights @ v
        joined = contexts.transpose(1, 2).contiguous().reshape(batch, tokens, width)
        return self.output(joined), weights


def loop_reference(x, model):
    batch, tokens, width = x.shape
    mask = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).triu(1)
    outputs = []
    for index in range(model.heads):
        start, end = index * model.head_dim, (index + 1) * model.head_dim
        q = x @ model.query.weight[start:end].T
        k = x @ model.key.weight[start:end].T
        v = x @ model.value.weight[start:end].T
        scores = q @ k.transpose(-2, -1) / math.sqrt(model.head_dim)
        weights = scores.masked_fill(mask, -torch.inf).softmax(-1)
        outputs.append(weights @ v)
    return model.output(torch.cat(outputs, dim=-1))


if __name__ == '__main__':
    torch.manual_seed(17)
    model = PackedMHA().double().eval()
    with torch.no_grad():
        model.query.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.key.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.value.weight.copy_(torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64))
        model.output.weight.copy_(torch.tensor([[1., -0.5], [0.5, 1.]], dtype=torch.float64))
    x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]],
                      [[2., 0.], [0., 2.], [1., -1.]]], dtype=torch.float64, requires_grad=True)
    result, weights = model(x)
    looped = loop_reference(x, model)
    torch.testing.assert_close(result, looped, atol=1e-12, rtol=1e-12)
    loss = result.square().sum()
    parameter = model.query.weight
    (packed_gradient,) = torch.autograd.grad(loss, parameter, retain_graph=True)
    (looped_gradient,) = torch.autograd.grad(looped.square().sum(), parameter)
    torch.testing.assert_close(packed_gradient, looped_gradient, atol=1e-12, rtol=1e-12)
    changed = x.detach().clone()
    changed[:, -1] = 50.0
    torch.testing.assert_close(model(changed)[0][:, :2], result.detach()[:, :2])
    print(json.dumps({'shape': list(result.shape), 'weight_shape': list(weights.shape),
                      'first_sequence_output': [[round(v, 6) for v in row] for row in result[0].tolist()],
                      'max_output_difference': round((result - looped).abs().max().item(), 12),
                      'gradients_equal': True, 'future_invariant': True,
                      'torch_version': torch.__version__}))
```

<!-- example:multihead_torch.py:end -->

If PyTorch is already installed, run `python examples/multihead_torch.py` from this package. A separate disposable environment can be made without modifying a learner project:

```bash
uv venv --python 3.12 /tmp/attention-heads-env
uv pip install --python /tmp/attention-heads-env/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
/tmp/attention-heads-env/bin/python examples/multihead_torch.py
```

The installer command downloads a CPU dependency; it does not train or fetch a model. Author verification used the already-installed `2.14.1+cpu` environment rather than executing this installation. See the [actual run record](project/reference-runs.md). The checked program prints output shape `[2,3,2]`, weight shape `[2,2,3,3]`, zero maximum output difference for this fixture, equal query-projection gradients within tolerance, and a successful future-input intervention.

### Reproduce the six-token route with the very same head parameters

The previous program keeps the compact numerical fixture. This additional file uses `[2,6,6]` and the explicit ModuleList shown above, copies every Q/K/V and output coefficient into its corresponding packed slice, and compares all parameter gradients as well as input gradients.

Complete file: [lecture8_equivalence.py](examples/lecture8_equivalence.py). Keep it beside the complete `head_loop.py` and `multihead_torch.py` files. In Colab, upload all three and run `!python lecture8_equivalence.py`.

```python
"""The 2x6x6 shape route and independent ModuleList/packed forward/backward checks."""
import torch
from head_loop import LoopMHA
from multihead_torch import PackedMHA


if __name__ == '__main__':
    torch.manual_seed(123)
    loop = LoopMHA(width=6, heads=2, context_len=6).eval()
    packed = PackedMHA(width=6, heads=2).double().eval()
    with torch.no_grad():
        for name in ['query', 'key', 'value']:
            source = torch.cat([getattr(head, name).weight for head in loop.heads], dim=0)
            getattr(packed, name).weight.copy_(source)
        packed.output.weight.copy_(loop.output.weight)
    x_loop = torch.randn(2, 6, 6, dtype=torch.float64, requires_grad=True)
    x_packed = x_loop.detach().clone().requires_grad_()
    y_loop = loop(x_loop)
    y_packed, weights = packed(x_packed)
    torch.testing.assert_close(y_loop, y_packed, atol=1e-12, rtol=1e-12)
    y_loop.square().sum().backward()
    y_packed.square().sum().backward()
    torch.testing.assert_close(x_loop.grad, x_packed.grad, atol=1e-12, rtol=1e-12)
    for name in ['query', 'key', 'value']:
        expected = torch.cat([getattr(head, name).weight.grad for head in loop.heads], dim=0)
        torch.testing.assert_close(expected, getattr(packed, name).weight.grad,
                                   atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(loop.output.weight.grad, packed.output.weight.grad,
                               atol=1e-12, rtol=1e-12)
    q_full = packed.query(x_packed.detach())
    q_view = q_full.view(2, 6, 2, 3)
    wrong = q_view @ q_view.transpose(-2, -1)
    q_heads = q_view.transpose(1, 2)
    right = q_heads @ q_heads.transpose(-2, -1)
    for label, tensor in [('projected', q_full), ('split', q_view), ('wrong scores', wrong),
                          ('transposed', q_heads), ('right scores', right), ('weights', weights)]:
        print(label, list(tensor.shape))
    changed = x_packed.detach().clone()
    changed[:, 4:] += 100
    torch.testing.assert_close(packed(changed)[0][:, :4], y_packed.detach()[:, :4],
                               atol=1e-12, rtol=1e-12)
    print('output', list(y_packed.shape), 'maximum difference', (y_loop-y_packed).abs().max().item())
    print('Input, Q/K/V and output-projection gradients agree; future intervention passes.')
```

It prints the incorrect score shape `[2,6,2,2]` produced before moving heads, then the correct `[2,2,6,6]`. The loop and packed outputs differ by about $2.22\times10^{-16}$ in this CPU float64 run; the comparisons use tolerances. It also checks gradients for the input, all Q/K/V slices and the output projection, then changes future positions to test causality.

![Five operations summarize the completed packed route: project, split, attend, merge and mix](images/original-slides/lecture-08-slide-19.png)

The implementation is a packaging of the same independent learned heads. It does not share K/V across query heads. Packing fewer key/value heads would be a different head arrangement and is developed in the separate GQA lesson.

### Why compare derivatives too?

Forward equality on one small input can miss a detached tensor or an accidental copy that breaks gradient flow. The program differentiates the same squared-output objective through both implementations and compares derivatives with respect to the shared query weights. It then changes the final input token and checks that earlier outputs remain fixed.

Those checks establish the declared small computation on CPU. They do not establish a performance advantage on every device, equality across all possible inputs, or numerical identity between every optimized kernel. The loop's kernel-launch and interpreter overhead can matter in practice, but timings depend on hardware, shapes, synchronization, dtype and implementation. No timing claim is made from the original lecture's demonstration.

[[practice:layout-check]]

### Keep head arrangements separate from other attention choices

We now have a concrete multi-head implementation and a reference loop that computes the same result. This folder can grow into other arrangements of query heads and key/value heads without scattering the foundational single-head material across new project directories.

### Describe the current implementation exactly

The head/KV area highlighted in the opening map is the concern this folder develops. Scoring, visibility, positional information and execution strategy remain separate questions about the same component. The map motivates the separate GQA/MQA lessons. This MHA reference itself has independent KV heads and no retained cache.

Our MHA reference has two independent query projections, two independent key projections and two independent value projections. It uses scaled dot-product scores, causal visibility, no bias terms and an output projection. Attention dropout is disabled in the matched calculations. The packed reference has no padding interface, positional method or KV cache; the explicit-head class exposes an optional dropout setting for training illustrations. It implements an attention component rather than a complete GPT: residual paths, normalization, feed-forward blocks, tokenization, vocabulary prediction and training are outside this reference.

Being precise about that boundary makes later changes easier to understand. Adding a key/value cache would change retained state and the inference interface. Adding dropout would change training behavior. Adding positional information would change the inputs to the computation. None of those is implied merely by naming the component MHA.

### A vocabulary for comparing head layouts

Let $H_q$ be the number of query heads and $H_{kv}$ the number of key/value heads. The following is a planning vocabulary, not a report of additional implemented models:

The terminology follows [Ainslie and colleagues' GQA paper, version 3](https://arxiv.org/abs/2305.13245v3). We use its head-sharing definitions here; its model quality and speed results are not reproduced by this tutorial.

| Arrangement | Query heads | Key/value heads | Meaning of the relationship |
| --- | --- | --- | --- |
| MHA | $H_q$ | $H_{kv}=H_q$ | Each query head has its own key/value head |
| GQA | $H_q$ | Fewer than $H_q$, typically dividing it | Groups of query heads share key/value heads |
| MQA | $H_q$ | One | All query heads share one key/value head |

For a conventional cache containing one key row and one value row per stored token, per key/value head, a simple storage count is

$$\text{bytes}=B\,T\,L\,H_{kv}(d_k+d_v)\,s,$$

where $B$ is batch size, $T$ stored token count, $L$ layers, and $s$ bytes per stored number. This expression describes the arrays under its stated layout. It excludes allocator overhead, quantization scales, temporary buffers, duplicated cache blocks, and architectures with different retained-state representations.

At fixed $B,T,L,d_k,d_v,s$, reducing $H_{kv}$ reduces those array bytes. It does not imply that the attention score matrix disappears or that quality and speed are unchanged. The separate GQA lesson defines and checks the query-to-KV mapping, including masks and cached query offsets. MQA and MLA have their own references. This MHA program remains the independent-head baseline; no trained-model benchmark is inferred.

### Do not infer token order from the head axis

![All dogs are animals and All animals are dogs use the same tokens in different orders](images/original-slides/lecture-08-slide-22.png)

The first statement and second statement have different meanings. If the operation receives only token identity embeddings and allows every position to inspect every other position, rearranging those rows simply rearranges the corresponding output rows. There is no separate indication of which arrangement was supplied. Multiple independently projected heads and output mixing do not remove that limitation by themselves.

Several heads still need a specification of visibility and order. Unmasked, position-free attention is permutation equivariant even with independent heads and a shared output projection: shuffling input rows shuffles output rows. The causal mask selects an ordered prefix and changes that condition; it is inaccurate to apply the same permutation test while silently keeping a fixed triangular mask.

That is a useful boundary check for this implementation. It does not open the omitted positional-encoding lecture or add its slides to the course. Lecture 9 remains outside the requested material.

### What you can now review

Given a proposed packed implementation, identify the query-token and source-token axes at the score product, check that per-head parameters are preserved, compare output ordering with explicit concatenation, and test the same loss gradient. Given a proposed grouped-query implementation, ask which query heads share which key/value head and which retained state that sharing changes.

[[practice:boundary-check]]

The [readiness record](project/readiness.md) lists the checks and explanation prompts for this route. Technical verification records the behavior of these examples; learner understanding and full-model results remain separate.

The [learning points](project/learning-points.md), [implementation notes](project/implementation.md), [reference runs](project/reference-runs.md), and [lecture reference list](project/source-map.md) support returning to the completed route.

### References

[Vizuara — The Attention Family, Lecture 8](https://www.youtube.com/watch?v=IOmg660SvmM) · [Presentation and diagrams](https://lecture8-attention-mechanisms.vercel.app/) · [Interactive heads, shape and timing lab](https://lecture8-attention-lab.vercel.app/) · [Lecture Colab notebook](https://colab.research.google.com/drive/1WVYWQh4YcjQq-kPDi1Hug-KYDud-bDkK?usp=sharing).

[Ainslie et al. — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v3) supplies the terminology used by the separate head-sharing lessons.
