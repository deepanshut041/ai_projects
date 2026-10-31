## Linear attention: summarize keys and values before a query arrives

A sliding window deliberately forgets old individual entries. Now let every old token contribute to a fixed-size state. The question is what summary lets a new query retrieve a meaningful mixture without revisiting every past key. We will first define a similarity function, then move the parentheses, preserve normalization, and turn the result into a recurrence.

### The parentheses can move only after the similarity is specified

![Kernel features permit a different multiplication order and a smaller intermediate](images/linear-1.png)

Dense softmax attention uses similarities $e^{q_i k_j^\top/\sqrt{d_k}}$, normalized across source j. You cannot simply replace $\operatorname{softmax}(QK^\top)V$ by $Q(K^\top V)$: softmax is nonlinear. The left complexity annotation should be $O(N^2d)$ for full prefill, not $O(Nd)$; one-query decode is a separate case.

Choose instead a positive kernel with a finite feature map $\phi:\mathbb R^{d_k}\to\mathbb R^F$:

$$\kappa(q,k)=\phi(q)\phi(k)^\top,\qquad y_i=\frac{\sum_j\kappa(q_i,k_j)v_j}{\sum_j\kappa(q_i,k_j)}.$$

The numerator can be reassociated:

$$\sum_j\phi(q_i)\phi(k_j)^\top v_j=\phi(q_i)\left(\sum_j\phi(k_j)^\top v_j\right).$$

The object in parentheses has shape F by dv, independent of the number of source tokens. Keep a second F-vector $z=\sum_j\phi(k_j)$ for the denominator. The diagram's formula displays only the numerator; omitting z would change the output scale as history grows.

The original linear-transformer paper uses $\phi(x)=\operatorname{ELU}(x)+1$, applied coordinatewise. For nonnegative x, this is x+1; for negative x, it is $e^x$. It gives positive features in exact arithmetic. This kernel is a different function from the exponential dot-product kernel. Performer, next, instead approximates that exponential kernel with random features.

### Write one association, then a second

Use already mapped key features $[1,2]$ and $[2,1]$, with scalar values 2 and 4. These correspond to raw nonnegative keys `[0,1]` and `[1,0]` under ELU+1. Start S as two zeros and z as `[0,0]`.

The first outer product adds `[2,4]` to S and `[1,2]` to z. Every normalized query with positive similarity returns the only visible value, 2. The second outer product adds `[8,4]`. Now

$$S=\begin{bmatrix}10\\8\end{bmatrix},\qquad z=[3,3].$$

For query feature `[1,3]`, the numerator is $1(10)+3(8)=34$ and the denominator is $1(3)+3(3)=12$. The result is $17/6=2.833333$. Check the explicit retrieval view: the two similarities are 7 and 5, so weights are 7/12 and 5/12 and the mixture is $(7(2)+5(4))/12$. The stored state and the explicit pairwise computation agree.



![Two positive-kernel writes update numerator and normalizer states before the query retrieves 34 divided by 12](images/worked-linear-state.svg)

Blue cells hold value-weighted sums, while purple cells count key-feature mass. The second token changes both objects. Follow the two coordinates separately: S becomes `[2+8,4+4]`, and z becomes `[1+2,2+1]`. The right panel contracts the same query with each state. Dividing their results is essential; returning 34 alone would grow with unnormalized accumulated mass.

This also corrects a common intuition: additive state updates do **not** make all tokens equally weighted for every query. The weights depend on the query/key feature match. What the simple additive rule lacks is a separate learned erase or correction mechanism for an association already stored.

[[practice:linear-normalize]]

### Causality becomes a prefix state

![Each token contributes an outer product and a normalizer update to fixed-size state](images/linear-2.png)

For self-inclusive causal attention, update before reading:

$$S_t=S_{t-1}+\phi(k_t)^\top v_t,\quad z_t=z_{t-1}+\phi(k_t),\quad y_t=\frac{\phi(q_t)S_t}{\phi(q_t)z_t}.$$

Reading before the update excludes the current token and gives a different causal convention. The matrix-state and vector-normalizer paths have different jobs; the diagram's final normalizer label accidentally includes a value term. z sums key features only. State indices should progress once per token.

A full training sequence can compute prefix sums, while generation carries only the current S,z between steps. Our simple autograd loop saves its computational graph when gradients are requested; it does not implement the paper's memory-efficient custom backward. Constant retained **inference** state does not mean the naive training graph has constant memory.

### Count state dimensions instead of comparing incompatible bars

![Storage bars contrast growing token caches with fixed-size state representations](images/linear-3.png)

For B batches, H heads, F kernel features, value width dv and element size s, the retained state is $BH(Fd_v+F)s$ bytes per layer. At B=H=1,F=dv=128,float32, that is 66,048 bytes, or 64.5 KiB. A corresponding ordinary KV cache at T=32768,dk=dv=128,float32 is 32 MiB per head per layer. Count every head and layer when comparing whole models.

The picture mixes per-token, per-layer and per-sequence labels, so its KB numbers and universal ratios are not a consistent allocation ledger. The precise formulas above define our comparison. A state-space model's state shape depends on its channels and internal dimensions; its label cannot be substituted for F-by-dv without defining that architecture.

### A bounded summary can merge distinct histories

Suppose every key feature is the scalar1. Histories with scalar values `[1,3]` and `[2,2]` both yield S=4,z=2. Every later query sees the same state and normalized answer2, although the histories differ. The state has no separately addressable row that tells us the first history's most recent value was3.



![Different two-token histories merge into the same numerator and normalizer state and therefore give identical future reads](images/worked-linear-collision.svg)

Each arrow into the central box represents accumulating a complete history. The values arrive in different sequences, but the state records only their sum and total key-feature mass in this scalar fixture. Every positive query cancels between numerator and denominator and returns 2. A larger fixed matrix can express more associations, but this example shows why a summary is not an individually addressable list of old tokens.

Positive features do not make values nonnegative: signed value contributions can cancel. Conversely, a blanket statement that positive kernel keys point in opposite directions is inappropriate for this feature map. The concrete collision above demonstrates the information limit without relying on that explanation.

![Illustrative retrieval bars motivate measuring the cost of a compressed history](images/linear-4.png)

These percentages are explicitly a stylized comparison in the workshop, not a common controlled benchmark we reproduced. They motivate testing exact fact retrieval, recency, repeated conflicting facts and long gaps. They do not establish that every linear model has 40 percent accuracy or every hybrid has 92 percent. Quality depends on training, architecture and the task. Delta-rule updates later address one specific problem: revising what the state predicts for a repeated key.

### Run both the pairwise and recurrent computations

`parallel` forms a kernel similarity matrix, removes future entries, then normalizes. `recurrent` instead accumulates S,z, reads the query and returns the final state for a later chunk. It never needs the full kernel matrix. The two paths test the **same kernel**, not equality to softmax attention.

Download [linear.py](examples/linear.py), run `python linear.py` with PyTorch, or upload it to Colab and run `!python linear.py`. CPU float64 is sufficient. The complete program includes all functions and declared inputs.

```python
"""Normalized positive-kernel attention: parallel and recurrent references."""
import json
import torch
import torch.nn.functional as F


def phi(x):return F.elu(x)+1


def parallel(q,k,v):
    weights=phi(q)@phi(k).T
    weights=weights.tril()
    return (weights@v)/weights.sum(-1,keepdim=True)


def recurrent(q,k,v,state=None):
    if state is None:
        s=q.new_zeros(q.shape[-1],v.shape[-1]);z=q.new_zeros(q.shape[-1])
    else:s,z=state
    outputs=[]
    for query,key,value in zip(phi(q),phi(k),v):
        s=s+torch.outer(key,value);z=z+key
        denominator=query@z
        if not denominator>0:raise ValueError('Nonpositive denominator')
        outputs.append((query@s)/denominator)
    return torch.stack(outputs),(s,z)


def run_checks():
    q=torch.tensor([[0.,0.],[0.,2.]],dtype=torch.float64)
    k=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
    v=torch.tensor([[2.],[4.]],dtype=torch.float64)
    hand,state=recurrent(q,k,v)
    torch.testing.assert_close(hand,torch.tensor([[2.],[17/6]],dtype=torch.float64))
    torch.manual_seed(97)
    q,k,v=[torch.randn(7,3,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    a,state=recurrent(q,k,v);b=parallel(q,k,v)
    torch.testing.assert_close(a,b)
    ga=torch.autograd.grad(a.square().sum(),(q,k,v),retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),(q,k,v))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s=recurrent(q[:3],k[:3],v[:3]);later,_=recurrent(q[3:],k[3:],v[3:],s)
    torch.testing.assert_close(torch.cat([first,later]),a)
    modified=v.detach().clone();modified[-1]+=100
    torch.testing.assert_close(recurrent(q,k,modified)[0][:-1],a[:-1])
    fresh,_=recurrent(q[3:],k[3:],v[3:])
    assert not torch.allclose(fresh,later)
    # Even a one-dimensional summary can collapse distinct histories.
    keys=torch.ones(2,1,dtype=torch.float64)
    left=keys.T@torch.tensor([[1.],[3.]],dtype=torch.float64)
    right=keys.T@torch.tensor([[2.],[2.]],dtype=torch.float64)
    torch.testing.assert_close(left,right)
    return {'worked_outputs':hand.flatten().tolist(),'state_elements_per_head':12,
            'parallel_recurrent_outputs_gradients_chunks_equal':True,
            'future_and_reset_checks':True,'distinct_histories_can_collide':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
```

Expected outputs include `[2,2.8333333333333335]`,12 retained scalar elements for the three-feature/three-value random case, equal outputs/gradients across both paths and equal chunked continuation. Changing the final value leaves earlier results unchanged. Starting a new state for the later chunk differs from continuation, illustrating why request ownership and deliberate resets matter.

A positive denominator is required; the reference rejects a nonpositive one rather than silently altering the formula. Extreme floating-point feature values can still underflow/overflow. Production kernels need a declared stabilization and accumulation policy. Actual GPU speed, custom-backward memory, model quality and hosted execution remain separate checks.

[[practice:linear-collision]]

[Implementation and commands](project/implementation.md) · [Readiness and limits](project/readiness.md) · [Transfer review](practice/review.md)

### References

- [Vizuara workshop book](https://vizuaraai.github.io/convolution-musical-journey/book.html), Sections9.3 and9.6; original kernel, state and comparison figures.
- [Transformers are RNNs](https://arxiv.org/html/2006.16236v3), Sections3.2–3.4, normalized kernel formulation and recurrence.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
