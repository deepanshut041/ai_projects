## Gated DeltaNet: forget first, then correct what remains

DeltaNet corrects an association when a related key arrives. Some old associations become irrelevant without ever receiving a contradictory update. Add a separate decay control so the state can fade before the next correction. The two controls answer different questions: alpha chooses how much old state survives; beta chooses how strongly the incoming key/value corrects it.

### Read the scalar dial before the channel-wise extension

![A scalar decay fades the whole memory uniformly, while a channel-wise extension uses different feature decay rates](images/delta-4.png)

The left panel is our subject: scalar $\alpha_t$ multiplies the entire head's state. The right panel previews a different extension with a diagonal matrix of per-channel decays. We retain both original panels so the distinction is visible, but the executable lesson implements **scalar** Gated DeltaNet. A scalar per head can differ between heads and tokens without becoming a per-channel vector.

We keep the previous lesson's row-vector convention S shaped dk-by-dv. Its update is

$$\bar S_t=\alpha_t S_{t-1},\quad \hat v_t=k_t\bar S_t,\quad S_t=\bar S_t+\beta_t k_t^\top(v_t-\hat v_t),\quad y_t=q_tS_t.$$

The positional order matters: predict from the **decayed** state. Expanding yields

$$S_t=\alpha_t(I-\beta_t k_t^\top k_t)S_{t-1}+\beta_t k_t^\top v_t.$$

Transpose the equation to match the value-by-key orientation printed in the diagram. Alpha is a scalar in this lesson; it commutes with the key transition. A diagonal channel gate generally does not commute, so substituting a vector and moving it around is not justified.

### Work one two-association memory

Let

$$S_{old}=\begin{bmatrix}2&0\\0&4\end{bmatrix},\quad k=[1,0],\quad v=[5,0],\quad\alpha=.5,\quad\beta=.25.$$

Before the write, `[1,0]` retrieves `[2,0]` and `[0,1]` retrieves `[0,4]`. Decay first gives $\bar S=\operatorname{diag}(1,2)$. The incoming key now predicts `[1,0]`, so residual is `[4,0]`. The correction outer product adds1 to the first diagonal entry. The final state is $\operatorname{diag}(2,2)$.

Two things happened at once. The unrelated orthogonal association fell from4 to2 because alpha affects the whole state. The targeted prediction moved from its decayed1 toward5 by one quarter, reaching2. If we incorrectly computed the residual from the old prediction2, the first entry would become $1+.25(5-2)=1.75$. The program tests that this wrong order differs.



![Two by two memory matrices show global decay before a residual write changes only the key-selected row](images/worked-gated-order.svg)

Each matrix uses key-feature rows and value-feature columns. Compare the diagonals from left to right: 2 and 4 become 1 and 2 under decay. The key `[1,0]` then selects the first row, predicts `[1,0]`, and writes one quarter of residual `[4,0]` back into that row. The second diagonal stays 2 during this correction. Computing the residual before decay would produce the red failure case instead.

With alpha1, the same state/key/value/beta gives $\operatorname{diag}(2.75,4)$: ordinary DeltaNet. With beta0 and alpha.5, no correction is written but the entire state still halves. “No write” therefore does not mean “state unchanged” once forgetting exists.

[[practice:gated-order]]

### Derive the same-key behavior and memory lifetime

For a unit key,

$$k_tS_t=(1-\beta_t)\alpha_t(k_tS_{t-1})+\beta_t v_t.$$

Beta1 replaces that key's prediction with the new value even after decay. Alpha0 removes the old-state contribution but still allows a fresh write; alpha0,beta0 clears state completely. For an arbitrary key norm, the correction again includes $\|k\|^2$, so normalize or specify an appropriate step-size rule.

With beta0 over n steps and constant alpha, $S_{t+n}=\alpha^nS_t$. For alpha.9, the half-life is $\log(.5)/\log(.9)\approx6.58$ updates. Token-dependent alpha produces a product of the intervening decays. Forgetting can filter stale information, but it can also erase a useful old fact. No gate setting guarantees perfect long-term retrieval from finite state.



![Gate boundary cases distinguish forgetting from writing and an exponential curve shows the lifetime of old state with no writes](images/worked-gated-controls.svg)

Read the left table as four separate interventions, each starting from the same old state. Zero beta disables correction while alpha can still forget; zero alpha clears the old contribution while beta can still write a new association. The right curve follows only the surviving old contribution over repeated no-write updates. Its vertical scale is a fraction of the starting state, and the dashed marker reaches one half at about 6.58 updates. Learned token-dependent gates need not follow one fixed curve.

The original work compares retention and filtering on different recall tasks and reports both strengths and remaining failures. Its model includes learned projections, short convolutions, query/key L2 normalization, output normalization/gating and an MLP stack. Our reference isolates the recurrence. Supplying sigmoid gate values in a toy program does not reproduce its trained gate parameterization or full architecture.

### The training algorithm must preserve the transition product

The expanded recurrence has the affine form $S_t=A_tS_{t-1}+B_t$, where $A_t=\alpha_t(I-\beta_t k_t^\top k_t)$ and $B_t=\beta_tk_t^\top v_t$. Over two tokens,

$$S_2=A_2A_1S_0+A_2B_1+B_2.$$

The first write is transformed by later transitions. Summing all $B_t$ independently drops that interaction. Chunkwise algorithms exploit structured transition products to train efficiently while retaining the same sequential result. A sequential Python implementation verifies semantics and supports chunk continuation; it does not demonstrate parallel GPU throughput or constant training-memory use.

### Run the residual and affine forms

Download [gated_delta.py](examples/gated_delta.py), run `python gated_delta.py` with PyTorch, or upload it to Colab and run `!python gated_delta.py`. CPU float64 is sufficient. `retrieve` makes the decay/predict/correct order explicit before the fixture exercises it.

```python
"""Scalar forgetting followed by a delta correction, in row-vector convention."""
import json
import torch
import torch.nn.functional as F


def retrieve(q,k,v,alpha,beta,state=None,matrix=False):
    s=q.new_zeros(k.shape[-1],v.shape[-1]) if state is None else state
    outputs=[]
    for query,key,value,a,b in zip(q,k,v,alpha,beta):
        if matrix:
            transition=a*(torch.eye(len(key),dtype=key.dtype)-b*torch.outer(key,key))
            s=transition@s+b*torch.outer(key,value)
        else:
            decayed=a*s
            prediction=key@decayed
            s=decayed+b*torch.outer(key,value-prediction)
        outputs.append(query@s)
    return torch.stack(outputs),s


def run_checks():
    initial=torch.diag(torch.tensor([2.,4.],dtype=torch.float64))
    key=torch.tensor([[1.,0.]],dtype=torch.float64);value=torch.tensor([[5.,0.]],dtype=torch.float64)
    _,hand=retrieve(key,key,value,torch.tensor([.5]),torch.tensor([.25]),initial)
    torch.testing.assert_close(hand,torch.diag(torch.tensor([2.,2.],dtype=torch.float64)))
    _,ungated=retrieve(key,key,value,torch.ones(1),torch.tensor([.25]),initial)
    torch.testing.assert_close(ungated,torch.diag(torch.tensor([2.75,4.],dtype=torch.float64)))
    wrong=.5*initial+.25*torch.outer(key[0],value[0]-key[0]@initial)
    assert not torch.allclose(hand,wrong)
    torch.manual_seed(113)
    qr,kr=[torch.randn(6,3,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    v=torch.randn(6,2,dtype=torch.float64,requires_grad=True)
    ar,br=[torch.randn(6,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    q,k=F.normalize(qr,dim=-1),F.normalize(kr,dim=-1);alpha,beta=ar.sigmoid(),br.sigmoid()
    out,s=retrieve(q,k,v,alpha,beta);ref,r=retrieve(q,k,v,alpha,beta,matrix=True)
    torch.testing.assert_close(out,ref);torch.testing.assert_close(s,r)
    args=(qr,kr,v,ar,br)
    ga=torch.autograd.grad(out.square().sum(),args,retain_graph=True)
    gb=torch.autograd.grad(ref.square().sum(),args,retain_graph=True)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s1=retrieve(q[:2],k[:2],v[:2],alpha[:2],beta[:2])
    later,s2=retrieve(q[2:],k[2:],v[2:],alpha[2:],beta[2:],s1)
    torch.testing.assert_close(torch.cat([first,later]),out);torch.testing.assert_close(s2,s)
    old=torch.randn(3,2,dtype=torch.float64)
    _,decayed=retrieve(q[:1],k[:1],v[:1],torch.tensor([.5]),torch.zeros(1),old)
    torch.testing.assert_close(decayed,.5*old)
    _,cleared=retrieve(q[:1],k[:1],v[:1],torch.zeros(1),torch.zeros(1),old)
    torch.testing.assert_close(cleared,torch.zeros_like(old))
    _,written=retrieve(q[:1],k[:1],v[:1],torch.zeros(1),torch.ones(1),old)
    torch.testing.assert_close(k[0]@written,v[0])
    return {'worked_state':hand.tolist(),'outputs_gradients_chunks_equal':True,
            'decay_before_prediction_required':True,'zero_write_still_forgets':True,'zero_decay_reset_and_write_checks':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
```

The worked state prints `[[2,0],[0,2]]`. The program verifies alpha1's ungated boundary, full output/final-state and query/key/value/alpha/beta-logit gradient equality between residual and affine forms, and chunk continuation. Beta0 decay, complete clearing and a fresh unit-key write after alpha0 are checked separately. Wrong residual timing is detected.

State storage remains dk-by-dv per batch/head/layer. A full network also needs any convolution state and per-layer bookkeeping. Hybrid full/sliding-attention layers retain their own caches; their storage does not disappear because other layers use a recurrence. Model quality, efficient kernels and actual hosted Colab execution remain separate work.

[[practice:gated-forget]]

[Implementation and commands](project/implementation.md) · [Readiness and limits](project/readiness.md) · [Transfer review](practice/review.md)

### References

- [Vizuara, Write only the surprise](https://vizuaraai.github.io/vizuara-ai-daily/kimi-delta-attention-explained/), Section4 and its complete original scalar/channel-gate figure.
- [Gated Delta Networks](https://arxiv.org/html/2412.06464v3), Sections3.1–3.4, Equation10, recurrence, retention/filtering cases and full-block boundary.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
