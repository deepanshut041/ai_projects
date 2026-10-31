## DeltaNet: read the old association before writing the correction

A fixed matrix can summarize a long history, but an additive write does not know whether the same association is already present. Suppose the current value for a key changes from2 to5. Adding5 on top of2 produces7. DeltaNet first retrieves the old prediction, compares it with5, and writes only the correction. Follow those three operations before naming the matrix recurrence.

### Start from the cost and the retained object

![The dense pairwise matrix and quadratic prefill work grow with sequence length](images/delta-1.png)

The square-matrix picture concerns a full sequence computation. A single cached decode query has a one-by-T score row and linear historical KV traffic. FlashAttention avoids materializing the square intermediate without removing dense comparisons; DeltaNet instead changes the history representation and read/write function. These solve different problems.

![A growing collection of individual token entries is replaced by a fixed matrix state](images/delta-2.png)

Think of S as a small learned associative memory. We use row-vector convention with S shaped dk-by-dv: a key k predicts `k @ S`, and a query q reads `q @ S`. The drawing uses the transposed value-by-key convention, so its $vk^\top$ update becomes $k^\top v$ in ours. Both are valid when used consistently. A query reads state; key and value define a write. The query arrow should not be interpreted as replacing the key in the outer product.

Unlike normalized positive-kernel retrieval, this DeltaNet reference does not carry a z denominator. It follows the delta-rule fast-memory formulation with normalized queries/keys and direct matrix readout. Do not silently attach the preceding lesson's normalizer and claim the same architecture.

### Predict, compare, correct

![The complete delta write sequence predicts a value, forms its residual and writes a scaled correction](images/delta-3.png)

For current k,v and write strength $\beta\in[0,1]$:

$$\hat v=kS_{old},\qquad e=v-\hat v,\qquad S_{new}=S_{old}+\beta k^\top e.$$

The learned model produces beta from its input; our tiny reference supplies beta explicitly. A beta of zero leaves the state unchanged. A beta of one applies the full residual outer product. This is a state update inside a forward pass, not a change to the model's learned projection parameters at inference.

Use dk=2,dv=1 and key `[1,0]`. Start S=`[0,0]` as a column. First value2 with beta1 gives S=`[2,0]`. A later value5 at the same key predicts2, residual3, and adds `[3,0]`, giving S=`[5,0]`. Read with query `[1,0]` and obtain5. With beta one quarter on the second write, add only .75 and obtain2.75. With zero beta, it stays2.

The additive rule would instead reach7. A normalized additive kernel at this repeated key would average to3.5. These are three different update/read functions; the desired behavior determines which comparison is meaningful.



![A residual of three moves the old same-key prediction from two toward five according to the write strength](images/worked-delta-correction.svg)

The top row follows the three operations in order. In the lower bars the dashed line marks the old read, 2, on the common 0–5 scale. Beta zero keeps it, beta one quarter adds 0.75, and beta one adds the complete residual. The target value is not added wholesale to the old memory. This distinction explains why additive accumulation would instead give 7.

### Derive when “overwrite” is exact

Multiply the updated state by the same key:

$$kS_{new}=\hat v+\beta\|k\|^2(v-\hat v).$$

For a **unit-norm** key, this is $(1-\beta)\hat v+\beta v$. The picture's full-overwrite statement at beta1 relies on this normalization. If k=`[2,0]`, its squared norm is4 and the effective correction is four times as large. The raw beta bound alone does not ensure interpolation.

Writing at one key can also affect another query u:

$$u(S_{new}-S_{old})=\beta(u\cdot k)(v-\hat v).$$

An orthogonal query is unaffected; a similar query sees some change. For k=`[1,0]`, query `[0,1]` sees none, while query $[1,1]/\sqrt2$ sees the correction divided by $\sqrt2$. Error-driven writes improve control; they do not create unlimited independent storage or erase every form of interference.



![Unit queries parallel, diagonal and orthogonal to an incoming key receive different changes from the same residual write](images/worked-delta-interference.svg)

The blue horizontal arrow is the incoming key direction. A query along it receives the full change of 3; the green unit query receives 3/√2; the orange orthogonal query receives none. The right-hand numbers are changes to whatever those queries read before the write. They show how a correction can affect related associations without promising that every association has its own independent slot.

[[practice:delta-overwrite]]

### Interpret the update as one online regression step

Hold the incoming key/value fixed and define

$$L(S)=\tfrac12\|kS-v\|^2,\qquad \nabla_S L=k^\top(kS-v).$$

Then $S_{new}=S_{old}-\beta\nabla L$ is exactly our residual update. S is the transient fast memory being fitted online. The surrounding model parameters are learned across training examples by the model's outer loss. Distinguish these two learning timescales.

Expand the residual and collect terms:

$$S_{new}=(I-\beta k^\top k)S_{old}+\beta k^\top v.$$

This affine matrix form makes the key-dependent transition explicit. Products of such transitions permit chunkwise parallel training strategies. Plain cumulative outer-product sums are insufficient because the new write depends on the previous prediction. Order matters: writing2 and then5 at the same unit key differs from writing5 and then2.

The primary parallel-DeltaNet work develops efficient compact representations and GPU algorithms. Our program checks the residual and matrix forms; a Python loop is not that optimized implementation.

### Execute both update forms, including their derivatives

Download [delta.py](examples/delta.py), run `python delta.py` with PyTorch, or upload it to Colab and run `!python delta.py`. CPU float64 is sufficient. `retrieve` shows both forms before `run_checks` calls either. Normalization is explicit in the random fixture; the hand example already uses a unit key.

```python
"""Delta-rule fast memory with equivalent residual and matrix updates."""
import json
import torch
import torch.nn.functional as F


def retrieve(q,k,v,beta,state=None,matrix=False):
    s=q.new_zeros(k.shape[-1],v.shape[-1]) if state is None else state
    out=[]
    for query,key,value,b in zip(q,k,v,beta):
        if matrix:
            transition=torch.eye(len(key),dtype=key.dtype)-b*torch.outer(key,key)
            s=transition@s+b*torch.outer(key,value)
        else:
            prediction=key@s
            s=s+b*torch.outer(key,value-prediction)
        out.append(query@s)
    return torch.stack(out),s


def run_checks():
    q=k=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    v=torch.tensor([[2.],[5.]],dtype=torch.float64)
    overwrite,_=retrieve(q,k,v,torch.ones(2))
    partial,_=retrieve(q,k,v,torch.tensor([1.,.25]))
    torch.testing.assert_close(overwrite.flatten(),torch.tensor([2.,5.],dtype=torch.float64))
    torch.testing.assert_close(partial.flatten(),torch.tensor([2.,2.75],dtype=torch.float64))
    torch.manual_seed(107)
    qraw,kraw=[torch.randn(7,3,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    values=torch.randn(7,2,dtype=torch.float64,requires_grad=True)
    gate=torch.randn(7,dtype=torch.float64,requires_grad=True)
    q,k=F.normalize(qraw,dim=-1),F.normalize(kraw,dim=-1);beta=gate.sigmoid()
    a,sa=retrieve(q,k,values,beta);b,sb=retrieve(q,k,values,beta,matrix=True)
    torch.testing.assert_close(a,b);torch.testing.assert_close(sa,sb)
    inputs=(qraw,kraw,values,gate)
    ga=torch.autograd.grad(a.square().sum(),inputs,retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),inputs,retain_graph=True)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s=retrieve(q[:3],k[:3],values[:3],beta[:3]);later,end=retrieve(q[3:],k[3:],values[3:],beta[3:],s)
    torch.testing.assert_close(torch.cat([first,later]),a);torch.testing.assert_close(end,sa)
    memory=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    loss=.5*(k[0]@memory-values[0]).square().sum()
    gradient=torch.autograd.grad(loss,memory)[0]
    expected=memory-beta[0]*gradient
    actual=memory+beta[0]*torch.outer(k[0],values[0]-k[0]@memory)
    torch.testing.assert_close(actual,expected)
    untouched=memory.detach().clone();_,result=retrieve(q[:1],k[:1],values[:1],torch.zeros(1),untouched)
    torch.testing.assert_close(result,untouched)
    return {'overwrite':overwrite.flatten().tolist(),'partial_write':partial.flatten().tolist(),
            'residual_matrix_outputs_gradients_chunks_equal':True,'online_sgd_identity':True,'zero_gate_preserves_state':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
```

The repeated-key outputs are `[2,5]`; the partial-write route prints `[2,2.75]`. Both matrix/residual computations agree on all outputs, final state and gradients with respect to raw queries, raw keys, values and write-gate logits. A carried state reproduces a full seven-token execution. An independent autograd calculation verifies the one-step regression identity, and beta0 preserves the previous state.

These checks establish local algebra and continuation. They do not reproduce the paper's language model, recall benchmark, custom GPU training kernel or actual hosted Colab run. A real request must carry the correct per-layer/per-head state and reset it deliberately. The next lesson adds a separate control for fading old state even when no conflicting key arrives.

[[practice:delta-interference]]

[Implementation and commands](project/implementation.md) · [Readiness and limits](project/readiness.md) · [Transfer review](practice/review.md)

### References

- [Vizuara, Write only the surprise](https://vizuaraai.github.io/vizuara-ai-daily/kimi-delta-attention-explained/), Sections1–3 and the three original cost/state/correction figures. This lesson uses its mechanism explanation; full-model product and performance claims are outside the component check.
- [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/html/2406.06484v6), Section2.2 and the online-regression/affine recurrence derivation.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
