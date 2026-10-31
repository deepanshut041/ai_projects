## ALiBi: put a distance penalty directly in the scores

RoPE supplied relative-position structure by rotating Q and K. Now ask whether the model can express a simple preference for nearby context without rotating those vectors or looking up a position embedding. We keep the content comparison and add a number determined by query–key distance before normalization. That number is a bias in the attention logits.

The plan is to inspect the bias matrix, calculate one complete row and its value mixture, give heads different slopes, then verify that cached chunks use the same distances as full attention. A longer-context quality claim will remain a separate trained-model question.

### Read content scores and distance biases as separate arrays

![The original ALiBi method diagram adds a head-specific linear bias to content comparisons](images/alibi-method.png)

On the left, query–key comparisons measure content compatibility. On the right, the relative-position term is added before softmax. The two terms have different jobs. Content can make a distant key useful; the bias asks the model to overcome a larger penalty for a larger distance.

For a causal query at absolute position i and a permitted key at j, write

$$\ell_{h,i,j}=q_{h,i}\cdot k_{h,j}/\sqrt{d_k}-m_h(i-j),\qquad j\leq i.$$

The slope $m_h$ is positive and fixed in the original method. Future positions receive minus infinity separately. Apply the dot-product scaling to the content term **before** adding the bias. Dividing the bias by $\sqrt{d_k}$ too would change its effective slope.

At i=3, the permitted distances to keys 0,1,2,3 are `[3,2,1,0]`. With slope 0.5, the corresponding biases are `[-1.5,-1,-0.5,0]`. The current position has no distance penalty. We describe a causal decoder here; using the signed difference unchanged for unrestricted bidirectional attention would favor one direction and needs a separately chosen bidirectional policy.

### Work an entire probability row

Set all four content logits to zero to isolate position. Exponentiate the biases:

$$[e^{-1.5},e^{-1},e^{-0.5},1]\approx[.223130,.367879,.606531,1].$$

Their sum is 2.197540. Normalizing gives approximately `[0.101536,0.167405,0.276004,0.455054]`. With scalar values `[1,2,3,4]`, the output is

$$.101536(1)+.167405(2)+.276004(3)+.455054(4)\approx3.084576.$$



![A zero-content attention row turns four distance penalties into probability bars and a weighted answer](images/worked-alibi-row.svg)

Follow one key horizontally through all three stages. Key 0 starts three positions away, receives logit −1.5, contributes about 0.101536 of the probability mass and supplies value 1. Key 3 starts at distance zero and supplies value 4 with weight about 0.455054. Add the four products in the last panel to obtain the output. The bars share a 0–1 probability scale; exponentials are divided by the same sum.

The last position receives more probability here because the content scores were equal. If the oldest key's content logit increases by 3, its combined logit becomes 1.5 while the most recent remains zero, so that older key outranks the recent key. ALiBi is an inductive preference, not a rule that older information must always lose.

A useful check is the probability ratio. For equal content logits, newest/oldest is $\exp(1.5)\approx4.481689$. Softmax's common denominator cancels in the ratio. Negating the intended bias sign would reverse this preference.

[[practice:alibi-sign]]

### Give heads different distance preferences

The original work fixes a geometric set of slopes instead of learning a separate embedding table. For a power-of-two head count H, our implementation uses

$$m_h=2^{-8(h+1)/H},\qquad h=0,\ldots,H-1.$$

With H=4, the values are `[0.25,0.0625,0.015625,0.00390625]`. A larger slope produces a stronger recency preference when content is held fixed. The slope 0.5 above was an explicitly selected arithmetic example. The program rejects non-power-of-two head counts rather than quietly claim it implements the upstream extension schedule; that extension is available in the authors' code.



![Four ALiBi query-head slopes produce different linear penalties on a common distance and bias scale](images/worked-alibi-slopes.svg)

The horizontal axis measures how far back a key is, and the vertical axis is the additive logit bias. Every line starts at zero because a current-position key has zero distance. Blue falls fastest: at 32 positions back its penalty is −8, while purple is only −0.125. These are the four scheduled slopes in the program; the preceding 0.5 example was chosen separately to make one probability row easy to calculate.

Heads still have their own content projections. This positional choice can coexist with an MHA, MQA or GQA memory arrangement. For GQA, the bias belongs to each **query** head even though several query heads share a KV head. Sharing K/V does not require sharing the positional slope.

### Keep the distance contract during cached inference

If the cache contains positions 0–2 and the next chunk contains positions 3 and 4, form distances between `[3,4]` and `[0,1,2,3,4]`. This gives two rows of length five; apply the causal rule independently to each row. Starting the new query positions at zero would change both the distance bias and permitted-key set.

Shifting every query and key position by the same constant leaves differences unchanged. Shifting queries alone does not. This makes a good reference check: cached chunks must agree with full attention when their absolute positions are consistent.

ALiBi does not require rotating cached keys. It also does not remove the need to store K/V for full-context attention. The retained-cache payload formula is unchanged. If a sliding window evicts history, the runtime must retain enough position information to preserve the chosen distance semantics.

### Implement and test the mechanism

Save [alibi.py](examples/alibi.py) and run `python alibi.py` with PyTorch installed. In Google Colab, inspect `torch.__version__`, upload the file, then run `!python alibi.py`. No weights or dataset are downloaded.

The `slopes` helper defines the supported schedule. `attention` constructs query/key distances, broadcasts one slope per head, combines scaled content scores with the bias, masks future entries and computes the value mixture. Tests cover both the arithmetic row and a larger random multi-head fixture:

```python
"""Causal ALiBi with explicit positions and a stated power-of-two slope schedule."""
import json
import math
import torch


def slopes(heads):
    if heads < 1 or heads & (heads-1):
        raise ValueError('This reference supports power-of-two head counts')
    return 2.0 ** (-8*torch.arange(1,heads+1,dtype=torch.float64)/heads)


def attention(q,k,v,query_positions,key_positions,m):
    distance = query_positions[:,None]-key_positions[None,:]
    bias = -m.to(q)[:,None,None]*distance.to(q)[None,:,:]
    score = q @ k.transpose(-2,-1)/math.sqrt(q.shape[-1]) + bias
    score = score.masked_fill(distance[None,None,:,:]<0,-torch.inf)
    weights = score.softmax(-1)
    return weights @ v,weights


def run_checks():
    torch.manual_seed(61)
    q = torch.randn(2,4,5,2,dtype=torch.float64)
    k = torch.randn_like(q);v = torch.randn(2,4,5,3,dtype=torch.float64)
    pos=torch.arange(5);m=slopes(4)
    y,w=attention(q,k,v,pos,pos,m)
    a,_=attention(q[:,:,:2],k[:,:,:2],v[:,:,:2],pos[:2],pos[:2],m)
    b,_=attention(q[:,:,2:],k,v,pos[2:],pos,m)
    torch.testing.assert_close(torch.cat([a,b],2),y)
    shifted,_=attention(q,k,v,pos+100,pos+100,m)
    torch.testing.assert_close(shifted,y)
    altered=v.clone();altered[:,:,3:]+=100
    torch.testing.assert_close(attention(q,k,altered,pos,pos,m)[0][:,:,:3],y[:,:,:3])
    z=torch.zeros(1,1,4,1,dtype=torch.float64)
    values=torch.arange(1,5,dtype=torch.float64).reshape(1,1,4,1)
    out,weights=attention(z,z,values,torch.arange(4),torch.arange(4),torch.tensor([.5]))
    row=weights[0,0,-1]
    assert torch.all(row[1:]>row[:-1])
    assert math.isclose((row[-1]/row[0]).item(),math.exp(1.5),rel_tol=1e-12)
    return {'slopes':m.tolist(),'last_weights':[round(x,6) for x in row.tolist()],
            'last_output':round(out[0,0,-1,0].item(),6),
            'chunks_and_shared_position_shift_equal':True,'future_intervention':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
```

The scalar row reproduces output 3.084576. The four-head fixture processed as two chunks agrees with the full causal computation; shifting all positions by 100 preserves outputs. Changing values at future positions cannot alter earlier outputs. These are mathematical/component checks, not a trained model's long-context evaluation.

[[practice:alibi-cache]]

### Interpret the original evaluation plots carefully

![Original experiments compare perplexity beyond the training subsequence length](images/alibi-extrapolation.png)

Read the axes and training-length labels before comparing curves. The horizontal axis changes validation input length; the vertical axis measures perplexity. These plots test particular trained models and evaluation policies. The existence of a bias formula for any integer distance is a mathematical fact; maintaining quality at those distances is an empirical result that can depend on the model and data.

![Original cost measurements compare positional methods in a stated training and inference setup](images/alibi-cost.png)

The source measurements belong to their hardware, batch/layout and implementation. A dense materialized per-head bias can consume memory proportional to $H_qT^2$ during full-sequence evaluation; an optimized kernel may generate its elements on demand. “No learned positional parameters” is not permission to ignore allocated masks, bias buffers or memory traffic.

To test longer-context behavior yourself, freeze a model, a dataset, evaluation token accounting, stride/window policy and context lengths. Changing the number of early tokens scored can change perplexity even without a better mechanism. We have not run that training/evaluation study here.

### Reconnect position to the next topic

RoPE modifies Q/K coordinates; ALiBi modifies the score by a fixed distance term. Neither determines how many KV heads exist. MLA will combine a compressed content representation with a separate rotary positional path, so we need both the algebra of linear projections and a precise understanding of where rotation occurs.

The [implementation notes](project/implementation.md), [readiness checks](project/readiness.md) and [transfer review](practice/review.md) support review.

### References

[Press, Smith and Lewis, ALiBi, version 2, Section 3 and Figures 1–3](https://arxiv.org/html/2108.12409v2) and [the authors' implementation and explanation](https://github.com/ofirpress/attention_with_linear_biases). These are the original authors' figures. A separate public Vizuara ALiBi lesson with matching diagrams was not found; the worked progression follows the same problem → matrix → arithmetic → code → limits structure as the preceding tutorials.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
