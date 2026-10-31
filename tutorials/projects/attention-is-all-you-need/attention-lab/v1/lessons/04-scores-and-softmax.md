## 2. Derive one attention output from the input rows

Chapter 1 showed how a position can gather information once contribution weights are supplied. We now open the part that was missing: how to calculate those weights from the current input. Stay with one receiving position first; chapter 3 will repeat the same calculation for the whole sequence.

Our source-side layer receives X, the input rows established in chapter 1. We want an output at position i that gathers useful information from other source positions. To calculate it, we need two things: a rule for assigning a weight to each source position, and the information that position will contribute after it receives a weight.

These are separate jobs. The features useful for deciding **whether to read a row** need not be the features useful for **what to pass onward**. We therefore give each input row three transformed versions.

### Where queries, keys and values come from

For input row x_i, calculate `q_i = x_i W_Q`, `k_i = x_i W_K` and `v_i = x_i W_V`. The W matrices are model parameters, shared across all positions. They will be adjusted during training. For one receiving position i, use:

| Vector | Comes from | Job in this output |
| --- | --- | --- |
| Query q_i | The receiving row x_i through W_Q | Supplies one side of every comparison for receiver i |
| Key k_j | Available row x_j through W_K | Supplies the other side of the comparison with position j |
| Value v_j | That same available row x_j through W_V | Supplies the features multiplied by position j's eventual weight |

A score against key j controls the contribution from value j. Keep that pairing: a high score against key 2 does not select value 0. “Query” and “key” name their roles in this comparison; no explicit question sentence or database is needed.

For the retained fixture, set `W_Q = I`, `W_K = I`, `W_V = 2I`, where I is the identity matrix. This gives exactly the X and V from chapter 1. We are choosing parameter settings so we can inspect one computation, rather than claiming that these settings were learned.

![One receiver compared with every key, then gathering the paired values](images/22-one-query.svg)

### Trace position 0 all the way through

Its query is `q_0 = [1,0]`. The keys are `[1,0]`, `[0,1]`, `[1,1]`, `[0,0]`. A dot product combines matching feature products into one scalar:

$$q_0\cdot k_0=1\times1+0\times0=1.$$

The four comparisons are `[1,0,1,0]`. Positions 1 and 3 have lower scores; positions 0 and 2 tie. A score is a learned compatibility calculation. It is not a dictionary-similarity label or a probability yet.

Scaled dot-product attention divides each score by the square root of the query/key feature width. Our width is two, so the scaled row is `[0.707107,0,0.707107,0]`. To see why this factor exists, suppose the coordinates of q and k are independent, zero-mean and unit-variance at initialization. Summing d products gives dot-product variance d. Dividing by `sqrt(d)` keeps that variance at one under those assumptions. Otherwise, wider comparisons can start with very large score differences and nearly saturated softmax weights. This is an initialization rationale, not a guarantee about every trained vector.

### Convert relative scores into contribution weights

We need a separate set of nonnegative weights for this receiver, with total weight one. Softmax takes each exponentiated score and divides by the total for this row:

$$a_{0j}=\frac{\exp(s_{0j})}{\exp(s_{00})+\exp(s_{01})+\exp(s_{02})+\exp(s_{03})}.$$

The exponentials are about `[2.028115,1,2.028115,1]`. Their total is `6.056230`. The resulting weights are `[0.334881,0.165119,0.334881,0.165119]`.

Now complete the weighted sum from chapter 1, using these calculated weights:

| Available position | Weight | Weighted value contribution |
| --- | --- | --- |
| 0 | 0.334881 | approximately `[0.669762,0]` |
| 1 | 0.165119 | approximately `[0,0.330238]` |
| 2 | 0.334881 | approximately `[0.669762,0.669762]` |
| 3 | 0.165119 | `[0,0]` |

Adding the unrounded contributions gives `c_0 = [1.339523,1.000000]`. This output has acquired information from other positions, but its relative contributions depend on the receiving row through q_0. That is the input-dependent communication we wanted.

### Implement the same chain

The program writes this one-query calculation directly. `query @ keys.T` performs all four comparisons. Subtracting the maximum score before exponentiating changes neither the final weights nor their ratios: the common factor cancels in the numerator and denominator. It avoids needlessly large exponentials.

```python
import math
import torch

torch.set_printoptions(precision=6, sci_mode=False)
query = torch.tensor([1., 0.], dtype=torch.float64)
keys = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
values = torch.tensor([[2., 0.], [0., 2.], [2., 2.], [0., 0.]], dtype=torch.float64)

raw_scores = query @ keys.T
scores = raw_scores / math.sqrt(query.shape[-1])
shifted_scores = scores - scores.max()
positive_weights = torch.exp(shifted_scores)
weights = positive_weights / positive_weights.sum()
context = weights @ values

print(raw_scores)
print(weights)
print(context)
```

### What changes when the receiving state changes?

Hold K and V fixed and use query `[0,1]`. Scores become `[0,1,1,0]`, and the gathered result becomes `[1.000000,1.339523]`. The memory is identical; the receiver's comparison changed. Query `[0,0]` gives four equal scores, weights `[0.25,0.25,0.25,0.25]` and the ordinary average `[1,1]`. This is also what the fourth input row produces as a receiver in this identity-projection fixture.

As a source, its zero key has score zero, so its exponential is 1 and its weight is positive. Its zero value contributes zeros, while its presence still changes the denominator. A zero input is therefore not automatically ignored; chapter 6 will show how a mask explicitly excludes padding.

This also explains why Q/K/V are useful roles rather than three interchangeable names for an embedding. Changing Q can redistribute weights. Changing V with Q and K fixed changes the content without redistributing weights. Chapter 4 examines the parameter maps that control those choices.

We used only one receiving row here. The encoder must retain an output for every source position, so next we arrange the same calculation into matrices. The formula and scaling follow [the original paper, section 3.2.1](https://arxiv.org/html/1706.03762v7).

**Runnable copy:** [one query through scores, weights and contributions](examples/walkthrough/02_softmax.py).

[[practice:softmax-check]]
