## Grouped-query attention: choose who shares memory

MQA showed that query heads can retrieve different mixtures from one shared K/V representation. It also introduced a restriction: every query head must use the same learned key and value projection. Now keep several KV projections and let only a selected group of query heads share each one. The decision is **which query head uses which KV head**.

We will draw that mapping before reshaping tensors. Then calculate one group, count storage, compare grouped and expanded implementations including derivatives, and inspect what a checkpoint conversion would change. This lesson implements GQA retrieval; it does not infer quality from a small example.

### Read the group boundary before counting heads

![Eight query lanes form groups that each retrieve from their own shared key/value representation](images/gqa-8-4.png)

Let $H_q$ be the number of query heads and $G=H_{kv}$ the number of KV heads. For equal contiguous groups, each group contains $R=H_q/G$ query heads. Require G to divide $H_q$. Using zero-based query-head index h, choose

$$g(h)=\left\lfloor h/R\right\rfloor.$$

For eight query heads and two KV groups, the mapping is `[0,0,0,0,1,1,1,1]`. Read one arrow all the way to its KV group. Heads 0–3 compare against $K_0$ and retrieve from $V_0$; heads 4–7 use $K_1,V_1$. Queries within a group still have independent projections. The shared objects contain a row per token, not a single vector for the entire sequence.

For each h,

$$z_h=\operatorname{softmax}\!\left(Q_hK_{g(h)}^\top/\sqrt{d_k}+M\right)V_{g(h)}.$$

After that, concatenate head outputs in query-head order and apply the existing output projection. Grouping changes neither the causal rule nor the meaning of the query/source-token axes.



![Eight adjacent query heads map in two contiguous groups to two retained KV heads](images/worked-gqa-map.svg)

Trace each arrow from a query to the memory it reads. Blue and green identify KV ownership, not token positions. A KV box contains every retained token row for that group. `repeat_interleave` reproduces the pictured mapping; alternating copies connect half the queries to the wrong memory even though every tensor dimension still fits.

### Work one group before generalizing

Use four query heads in two groups so the numbers fit on paper. The mapping is `[0,0,1,1]`. At the final position, heads 0 and 1 have scalar queries 1 and 2; group 0 has keys `[1,0,2]` and values `[2,-1,3]`. Their outputs from the MQA calculation are `2.395149` and `2.819185`.

Let heads 2 and 3 both have query zero, but let group 1 values be `[4,0,2]`. Zero queries give equal scores and therefore uniform weights within this fully visible prefix. Their output is $(4+0+2)/3=2$. Equal queries in different groups need not yield equal outputs because the retrieved values can differ. Here the joined feature row is `[2.395149,2.819185,2,2]` before the output projection.

This exposes both freedoms: queries can differ inside a group, and memory projections can differ between groups. A head cannot silently pick another group's KV just because the resulting tensor has a legal shape.

[[practice:gqa-map]]

### Cache size depends on KV groups

![Bars compare retained bytes for independent, grouped and shared key/value representations](images/gqa-8-5.png)

At $H_q=32,d_k=d_v=128$, FP16, one token and one layer, MHA retains $32(128+128)2=16384$ bytes. GQA with G=8 retains 4096 bytes. MQA retains 512 bytes. The GQA/MHA ratio is $G/H_q=1/4$. This is a per-layer, per-token payload count; multiply by batch, retained tokens and layers for a complete cache.



![MHA, GQA and MQA retained KV payloads are compared on one common byte scale](images/worked-gqa-bytes.svg)

The horizontal lengths now follow exactly the dimensions above. GQA is one quarter of MHA because 8 of 32 KV heads remain; MQA is one thirty-second. This comparison includes both keys and values, and excludes weights, temporary tensors and allocation metadata. Those exclusions are the same for all three bars.

The latent bar requires its own representation specification. If a positional component is cached alongside the latent, count it too; a latent rank alone does not define a complete MLA cache. We will resolve that in the MLA lesson.

For a second explicit example, $H_q=64,G=8,d_h=128,L=80,T=32768,B=1,s=2$ gives GQA 10 GiB and the corresponding MHA 80 GiB. MQA would be 1.25 GiB. The storage ratio for this 64-head example is 8, not 4. Binary GiB and decimal GB are different units.

### Choose a group count through evaluation

![Illustrative quality and memory curves show the motivation for an intermediate number of KV heads](images/gqa-8-6.png)

The memory curve follows directly from the group-count formula. A quality curve requires trained models, a dataset and an evaluator. Its plateau cannot be established from architecture alone. Treat this drawing as the selection idea, not a measured proof that a chosen group count has negligible quality loss in every model.

The GQA paper evaluated uptrained T5 models under stated tasks and hardware. That is evidence about those experiments. Our component equality checks below establish a different fact: two ways to compute the **same GQA** agree when their inputs and coefficients match.

### Group the query axis explicitly

Our reference accepts already projected Q/K/V. Q has shape `[B,Hq,S,dk]`, K `[B,G,T,dk]` and V `[B,G,T,dv]`. S counts current query rows, T retained key/value rows. Reshape Q as `[B,G,R,S,dk]`; insert a size-one axis in K and V to obtain `[B,G,1,T,dk/dv]`. The backend broadcasts memory within each group. Scores have shape `[B,G,R,S,T]`.

That extra group axis makes the mapping visible. An equivalent expanded reference repeats each KV head R times along the head axis. It is `repeat_interleave`, not a tiling that alternates groups: `[0,0,1,1]` and `[0,1,0,1]` represent different wiring. Both can fit the same dimensions.

The retained cache should stay `[B,G,T,D]`; saving an expanded Hq-head copy forfeits the retained-state saving. A backend may still materialize expanded intermediates, which affects peak memory and performance separately.

### Verify endpoints, derivatives and appended chunks

Save [gqa.py](examples/gqa.py), inspect `torch.__version__`, then run `python gqa.py` in your PyTorch environment or upload it and run `!python gqa.py` in Google Colab. CPU float64 is sufficient. `attention` defines visibility and score normalization first. `grouped_attention` then validates the layout and exposes grouping; `run_checks` exercises it independently through expansion.

```python
"""Projected GQA: explicit grouping, expanded reference and cached chunk checks."""
import json
import math
import torch


def attention(q, k, v, offset=0):
    query = offset + torch.arange(q.shape[-2],device=q.device)
    key = torch.arange(k.shape[-2],device=k.device)
    scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
    return scores.masked_fill(key[None,:] > query[:,None],-torch.inf).softmax(-1) @ v


def grouped_attention(q, k, v, offset=0):
    if any(t.ndim != 4 for t in [q,k,v]):
        raise ValueError('Expected four named axes')
    b,h,t,d = q.shape
    g = k.shape[1]
    if g < 1 or h % g or k.shape[:3] != v.shape[:3]:
        raise ValueError('KV heads must divide query heads and K/V leading axes must agree')
    if k.shape[0] != b or k.shape[-1] != d:
        raise ValueError('Batch or comparison width mismatch')
    queries = q.reshape(b,g,h//g,t,d)
    out = attention(queries,k.unsqueeze(2),v.unsqueeze(2),offset)
    return out.reshape(b,h,t,v.shape[-1])


def run_checks():
    torch.manual_seed(43)
    q = torch.randn(2,4,5,2,dtype=torch.float64,requires_grad=True)
    results = {}
    for g in [1,2,4]:
        k = torch.randn(2,g,5,2,dtype=torch.float64,requires_grad=True)
        v = torch.randn(2,g,5,3,dtype=torch.float64,requires_grad=True)
        actual = grouped_attention(q,k,v)
        repeat = 4//g
        expanded = attention(q,k.repeat_interleave(repeat,1),v.repeat_interleave(repeat,1))
        torch.testing.assert_close(actual,expanded,atol=1e-12,rtol=1e-12)
        left = torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
        right = torch.autograd.grad(expanded.square().sum(),(q,k,v))
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        first = grouped_attention(q[:,:,:2],k[:,:,:2],v[:,:,:2])
        later = grouped_attention(q[:,:,2:],k,v,offset=2)
        torch.testing.assert_close(torch.cat([first,later],2),actual)
        results[str(g)] = {'head_map':[h//repeat for h in range(4)],
                          'kv_payload_bytes':sum(x.numel()*x.element_size() for x in [k,v])}
    try:
        grouped_attention(q,torch.zeros(2,3,5,2),torch.zeros(2,3,5,3))
    except ValueError:
        pass
    else:
        raise AssertionError('Non-divisible grouping accepted')
    return {'groups':results,'output_and_gradients_equal':True,'cached_chunks_equal':True,
            'invalid_grouping_rejected':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
```

For G=1, the printed mapping is `[0,0,0,0]`: MQA. For G=4, it is `[0,1,2,3]`: MHA. G=2 gives `[0,0,1,1]`, our intermediate case. At B=2, T=5, dk=2, dv=3, float64, their payloads are 400, 800 and 1600 bytes respectively. Every case agrees with its expanded reference on outputs and Q/K/V gradients. Processing the first two query positions and then the remaining three with offset two reproduces the full causal result. G=3 is rejected for four query heads.

This is a retrieval kernel, not a complete trained decoder. Q/K/V projection, joining and output mixing use the contracts already established in MHA/MQA. The cache lesson supplies request/reset/absolute-position ownership. Real grouped kernels can avoid the expanded intermediate; this reference makes no GPU timing claim.

### Converting pretrained coefficients changes the model

Ainslie and colleagues initialize each group's K/V projection by averaging the corresponding MHA heads, then continue training. For two scalar key weights 1 and 3, the group initializer is 2. In general,

$$\operatorname{softmax}\!\left(q(K_1+K_2)^\top/2\right)
\ne\tfrac12\left(\operatorname{softmax}(qK_1^\top)+\operatorname{softmax}(qK_2^\top)\right).$$

Softmax is nonlinear; averaging parameters is not an output-preserving rewrite. Averaging value coefficients also removes independent degrees of freedom. The original paper's uptraining recipe and measured results should not be confused with the grouping/expansion identity we tested.

![A schematic frontier compares quality with retained memory](images/gqa-8-9.png)

A point on this plot must correspond to a specified trained configuration and evaluation before we can rank it. There is no theorem here that GQA always dominates MQA or MLA always dominates GQA. State quality, bytes and runtime separately, and report the conditions behind comparisons.

![Parameter-count bars distinguish conventional head sharing from latent projections](images/gqa-8-10.png)

With model width D and equal head width d, the input-projection parameter count is $DH_qd+2DGd$. Include $H_qdD$ more for the output projection if it maps joined heads back to D. For D=8, Hq=4, d=2 and G=2, this gives 128 input-projection parameters and 64 output parameters, total 192 without biases. The corresponding MHA total is 256; MQA is 160. MLA requires counting its own compression and expansion matrices rather than inserting an arbitrary group count into this formula.

[[practice:gqa-conversion]]

### What we have and what comes next

MHA, MQA and GQA now share one consistent vocabulary: the query-head count, KV-head count, query-to-KV mapping and actual retained tensor shape. GQA changes how many memory representations remain. MLA will change how those representations are stored, but first we need to understand how positional information interacts with query/key transformations.

Use the [implementation notes](project/implementation.md), [readiness record](project/readiness.md) and [transfer review](practice/transfer-review.md) for checks and review.

### References

[Vizuara: Understand Grouped Query Attention](https://www.youtube.com/watch?v=kx3rETIxo4Q). [Vizuara Inference Engineering, Chapter 8](https://vizuaraai.github.io/convolution-musical-journey/book.html): original figures 8.4–8.6 and 8.9–8.10. [Ainslie et al., GQA, version 3, Sections 2–3](https://arxiv.org/html/2305.13245v3): grouping, mean-pooling initialization, uptraining and scoped experimental results. The workshop diagrams are not asserted to be frames/slides from the older video.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
