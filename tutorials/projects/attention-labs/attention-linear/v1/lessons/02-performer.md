## Performer: approximate the kernel, then reuse the state

The previous lesson chose ELU+1 similarity. That enabled a small recurrent state but changed the attention kernel. Performer instead asks whether the exponential dot-product similarities behind softmax can be estimated with positive finite features. Keep two questions separate throughout: how accurately those features estimate softmax, and whether two computations of the selected feature approximation agree.

### Read the multiplication order and its missing normalizer

![The original method diagram contrasts a full pairwise matrix with low-dimensional feature products](images/performer-method.png)

On the left, L-by-L similarities multiply L-by-d values. On the right, key features are combined with V first, producing an r-by-d state, then query features read it. The dashed blue box identifies the product to compute first. The diagram depicts the unnormalized numerator; normalized retrieval also needs the corresponding feature sum. This is the same S,z structure from linear attention, with a new feature map.

For raw query/key width d, let $x=q/d^{1/4}$ and $y=k/d^{1/4}$. Then $x\cdot y=q\cdot k/\sqrt d$. Scaling **both** inputs by the fourth root is necessary; dividing both by the square root would apply too much scaling.

### Derive a positive random feature

For a Gaussian vector $\omega\sim\mathcal N(0,I)$, define

$$f_\omega(x)=\exp(\omega\cdot x-\|x\|^2/2).$$

The Gaussian moment identity gives

$$\mathbb E[f_\omega(x)f_\omega(y)]
=e^{-(\|x\|^2+\|y\|^2)/2}\mathbb E[e^{\omega\cdot(x+y)}]
=e^{x\cdot y}.$$

With r samples, form $\phi(x)_a=f_{\omega_a}(x)/\sqrt r$. Their dot product is an average of r positive products, estimating the exponential kernel. Every finite sample gives positive similarities in exact arithmetic. That helps keep normalization meaningful, although floating-point exponentials can still overflow or underflow.

Check a hand example where d=1, x=1, y=0 and two selected directions are +1 and -1. The feature dot product is

$$\tfrac12(e^{1-.5}+e^{-1-.5})=.935925,$$

whereas the true kernel is $e^{1(0)}=1$. This deliberately chosen two-direction example is an arithmetic illustration, not a Gaussian-distributed estimator. It shows why a finite feature calculation is not exact softmax even when every operation is implemented correctly. At x=y=0, every component is $1/\sqrt r$, so the estimated kernel is exactly1 for any projection samples.



![Two positive feature products sum to approximately 0.935925 while the exact exponential kernel is one](images/worked-performer-features.svg)

Each row is one feature coordinate. The factor 1/√2 appears in both vectors, so their product carries 1/2. Sum the two positive products to obtain the estimate, then compare it with the exact kernel in the green panel. A positive result can still have approximation error. The rounded products may sum to 0.935926; the displayed total is rounded after summing the unrounded exponentials.

[[practice:performer-scale]]

### Why positive and orthogonal both appear in FAVOR+

![All panels compare positive-feature estimation error with trigonometric-feature estimation error](images/performer-variance.png)

Read the left surface as the error ratio across angle and length, the center as a fixed-length angular slice, and the inset as the two errors in a small-kernel region. The plotted ratio is estimator MSE, not downstream model accuracy or output latency. Signed sine/cosine approximations can produce negative estimated similarities or problematic denominator cancellation; positive features address that issue for this kernel construction.

Orthogonal directions reduce redundant sampling. At most d vectors can be mutually orthogonal in d dimensions, so for r>d our reference creates independent orthogonal blocks. A Gaussian vector has both a uniform direction and a random radius. QR alone returns unit-length directions; multiply each by an independent Gaussian-vector norm to preserve Gaussian radial marginals. Setting every radius to $\sqrt d$ instead corresponds to a related regularized construction, not the same Gaussian-marginal estimator.

`orthogonal_features` therefore draws a square Gaussian matrix, takes QR, corrects column signs, transposes to row directions and attaches independent radii. Rows within each block are orthogonal; rows across different blocks need not be. The code checks this property before using the features.

### Normalize the estimated similarities before interpreting their output

With a fixed sampled projection matrix, update $S_t=S_{t-1}+\phi(k_t)^\top v_t$ and $z_t=z_{t-1}+\phi(k_t)$, then return $\phi(q_t)S_t/(\phi(q_t)z_t)$. This is exactly the recurrent evaluation of that sample's finite kernel. It approximates normalized softmax retrieval because the kernel itself is estimated.

Even an unbiased estimate of each unnormalized similarity does not make a ratio of random numerator and denominator an unbiased output estimator. Nor does it guarantee an error threshold for every finite seed and query. Increasing features improves statistical estimates under stated assumptions; one particular sample's error need not decrease monotonically.

The sampled matrix is part of the state contract. Redrawing it halfway through a cached sequence changes the coordinates in which old S,z were accumulated. Rebuild the state when changing the projection, or use an explicitly derived compatible transformation. Save the projection/seed and feature convention with any reusable state.

### Execute equality and approximation as different checks

Download [performer.py](examples/performer.py), run `python performer.py` with PyTorch, or upload it to Colab and run `!python performer.py`. CPU float64 is sufficient. The complete reference includes orthogonal sampling, feature construction, pairwise finite-kernel normalization, recurrent evaluation and the exact softmax reference.

```python
"""Positive orthogonal random features; small float64 reference, not a fused kernel."""
import json
import math
import torch


def orthogonal_features(count,width,seed):
    generator=torch.Generator().manual_seed(seed);blocks=[]
    for _ in range(math.ceil(count/width)):
        a=torch.randn(width,width,generator=generator,dtype=torch.float64)
        q,r=torch.linalg.qr(a)
        signs=torch.where(r.diag()<0,-1.,1.)
        directions=(q*signs).T
        radii=torch.randn(width,width,generator=generator,dtype=torch.float64).norm(dim=1)
        blocks.append(directions*radii[:,None])
    return torch.cat(blocks)[:count]


def features(x,omega):
    x=x/(x.shape[-1]**.25)
    exponent=x@omega.T-x.square().sum(-1,keepdim=True)/2
    return exponent.exp()/math.sqrt(len(omega))


def pairwise(qf,kf,v):
    a=(qf@kf.T).tril()
    return (a@v)/a.sum(-1,keepdim=True)


def recurrent(qf,kf,v,state=None):
    s,z=(qf.new_zeros(qf.shape[-1],v.shape[-1]),qf.new_zeros(qf.shape[-1])) if state is None else state
    out=[]
    for q,k,value in zip(qf,kf,v):
        s=s+torch.outer(k,value);z=z+k
        out.append((q@s)/(q@z))
    return torch.stack(out),(s,z)


def softmax_reference(q,k,v):
    scores=q@k.T/math.sqrt(q.shape[-1])
    return scores.masked_fill(torch.ones_like(scores,dtype=torch.bool).triu(1),-torch.inf).softmax(-1)@v


def run_checks():
    torch.manual_seed(101)
    q=(torch.randn(6,4,dtype=torch.float64)*.3).requires_grad_()
    k=(torch.randn(6,4,dtype=torch.float64)*.3).requires_grad_()
    v=torch.randn(6,2,dtype=torch.float64,requires_grad=True)
    omega=orthogonal_features(64,4,7)
    units=omega[:4]/omega[:4].norm(dim=1,keepdim=True)
    torch.testing.assert_close(units@units.T,torch.eye(4,dtype=torch.float64))
    qf,kf=features(q,omega),features(k,omega)
    assert (qf>0).all() and (kf>0).all()
    a,state=recurrent(qf,kf,v);b=pairwise(qf,kf,v)
    torch.testing.assert_close(a,b)
    ga=torch.autograd.grad(a.square().sum(),(q,k,v),retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),(q,k,v),retain_graph=True)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s=recurrent(qf[:2],kf[:2],v[:2]);later,_=recurrent(qf[2:],kf[2:],v[2:],s)
    torch.testing.assert_close(torch.cat([first,later]),a)
    target=softmax_reference(q,k,v)
    errors={}
    for count in [16,64,256]:
        errors[count]=[]
        for seed in range(4):
            w=orthogonal_features(count,4,seed)
            result,_=recurrent(features(q,w),features(k,w),v)
            errors[count].append(float((result-target).abs().mean().detach()))
    return {'output_mae_by_features_and_seeds':errors,'same_feature_outputs_gradients_chunks_equal':True,
            'orthogonal_within_blocks':True,'state_elements_at_64_features':192}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
```

For the fixed six-token fixture, pairwise and recurrent feature outputs, Q/K/V gradients and chunk continuation agree. That is the equality check. Against exact causal softmax, mean absolute output errors across seeds0–3 are:

| Features r | Seed0 | Seed1 | Seed2 | Seed3 |
| --- | --- | --- | --- | --- |
|16|.018012|.120389|.037320|.013626|
|64|.008472|.039542|.009077|.013935|
|256|.004816|.010569|.005387|.002583|

These are tiny-component results, not task accuracy. Seed3 slightly worsens from16 to64, demonstrating why we do not assert monotonicity for a single draw. At r=64,dv=2, the state contains $64(2+1)=192$ scalar elements per head. Raising r buys sampling capacity while increasing state size and work.

This reference uses direct exponentials and bounded inputs to expose the derivation. A performant FAVOR+ implementation also needs numerical stabilization, efficient scans/backward and a projection-redraw policy during training. A per-query common feature scale cancels in its normalized output; arbitrary per-key scales change relative source weights unless compensated consistently. Do not stabilize a recurrence by rescaling new keys while leaving old state in incompatible units.



![Twelve output error bars compare three feature counts across all four fixed random seeds on a common scale](images/worked-performer-errors.svg)

The horizontal groups change feature count; color identifies the same seed across groups. Height is output MAE, not model accuracy or runtime. Seed 1 is the tallest at every count in this fixture, while seed 3’s small increase from 16 to 64 features shows why one finite draw need not improve monotonically. The runnable reference prints every value used here. Its exact pairwise/recurrent equality check holds within each fixed feature sample; the bars test that sample against a different target, exact softmax.

[[practice:performer-state]]

The next lesson changes the write rule itself. Instead of always adding another association, DeltaNet reads what memory already predicts and writes the residual needed to correct it.

[Implementation and commands](project/implementation.md) · [Readiness and limits](project/readiness.md) · [Transfer review](practice/review.md)

### References

- [Rethinking Attention with Performers](https://arxiv.org/html/2009.14794v4), Sections2.2–2.4, Lemma1 and Figures1–2. Both original primary-author figures are included with every panel. This topic uses the primary method source because a matching public Vizuara lesson was not available.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
