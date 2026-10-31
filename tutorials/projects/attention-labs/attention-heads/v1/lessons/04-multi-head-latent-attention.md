## Multi-head latent attention: retain the ingredients

GQA reduced how many distinct key/value heads we keep. MLA asks a different question: can each token retain a smaller common representation from which the heads' keys and values are computed? Start with the stored object, then follow its two uses: comparing a query against history and retrieving a weighted mixture. Finally add position, because a content-only compression picture is incomplete for RoPE.

### Replace many retained vectors with one latent row

![Independent, grouped, shared and latent key/value cache layouts](images/mla-original-comparison.png)

The shaded objects are retained between generation steps. On the left, MHA stores every head's K and V. GQA/MQA share some of those objects. On the right, MLA stores a common compressed row, while head-specific linear maps still define different content representations. A head can retain a distinct retrieval pattern without storing an expanded historical key for every head.

For one input row $x_t\in\mathbb R^D$, use row-vector notation:

$$c_t=x_tW_D\in\mathbb R^r,\quad k^C_{t,h}=c_tU^K_h,\quad v_{t,h}=c_tU^V_h.$$

$W_D$ has shape D by r. Each $U^K_h$ is r by $d_c$, and each $U^V_h$ is r by $d_v$. Across T retained tokens, C has shape T by r. This is a learned bottleneck, not a lossless encoding of arbitrary independently chosen MHA keys and values. A model must be trained or adapted for these factorized coefficients.

### Follow the multiplication before storing expanded keys

![A token is compressed into a latent row and the key expansion can be absorbed into the query path](images/mla-8-7.png)

For one head, the content score is

$$q^C_h(CU^K_h)^\top=(q^C_h(U^K_h)^\top)C^\top.$$

The left computes expanded historical keys first. The right transforms the current query into latent coordinates first. Both produce the same T scores. If $q^C_h=xW^Q_h$, the product $W^Q_h(U^K_h)^\top$ can be precomputed when fixed inference weights are loaded. We call this absorption: it changes the multiplication order, preserving the same factorized model.

Work a two-dimensional example. Let the retained latent rows be `[1,0]`, `[0,1]`, `[1,1]`, the query `[1,1]`, and

$$U^K=\begin{bmatrix}1&2\\0&1\end{bmatrix}.$$

Expanded keys are `[1,2]`, `[0,1]`, `[1,3]`. Their raw dot products with q are `[3,1,4]`. Instead form $q(U^K)^\top=[3,1]$ and compare that against the three latent rows. The result is again `[3,1,4]`. Scale by $\sqrt2$ in this **content-only** example and apply softmax once across those three scores. Do not softmax the coordinates of the transformed query.

The value path has the same associativity. For attention weights a,

$$a(CU^V_h)=(aC)U^V_h.$$

If $U^V=\operatorname{diag}(2,3)$, the expanded values are `[2,0]`, `[0,3]`, `[2,3]`. Mixing the latent rows first gives `[a_0+a_2,a_1+a_2]`; multiplying by $U^V$ gives `[2(a_0+a_2),3(a_1+a_2)]`, exactly the expanded weighted sum. A fixed output projection can also be combined with the value up-projection. Neither rewrite permits moving a linear map through softmax arbitrarily; we preserved the logits and the weights themselves.



![Expanded historical keys and an absorbed current query produce the same three raw scores and the same value mixture](images/worked-mla-absorption.svg)

Read the two columns as alternative computations, using the same fixed coefficients. Each ends at the score row `[3,1,4]`; scaling and normalization therefore remain identical. The lower strip follows the value mixture through the same change of parentheses. We moved linear operations around matrix multiplication, while leaving softmax between the score calculation and value retrieval.

[[practice:mla-absorb]]

### Position introduces a second retained object

RoPE rotates a query at its query position and a key at its source position. Applying those rotations to the content keys inserts a position-dependent matrix between the factors. In column notation this contains $R_i^\top R_j$; in general it does not commute with the learned up-projection. A single fixed absorbed matrix cannot stand in for every relative position.

Decouple the paths. Give each query head a content vector $q^C_{i,h}$ and a rotated positional vector $q^R_{i,h}$. Retain the latent content $c_j$ and a shared rotated positional key $k^R_j$. Then

$$\ell_{h,i,j}=\frac{q^C_{i,h}(U^K_h)^\top c_j^\top+q^R_{i,h}(k^R_j)^\top}{\sqrt{d_c+d_r}}+M_{i,j}.$$

This is the dot product of the conceptual concatenations $[q^C;q^R]$ and $[k^C;k^R]$. The scale counts their combined comparison width. Softmax normalizes the combined content-plus-position score. Values use the content latent; they are not rotated in this construction.

The drawing above isolates the content factorization. Its phrase “only latent cached” omits this positional path. A complete decoupled-RoPE cache keeps **both** $c_j$ and $k^R_j$. Queries, including any query compression used during training, are current computations and need not be retained as historical cache entries.



![Current content and positional queries meet separately retained latent rows and rotated positional keys before one shared softmax](images/worked-mla-position-cache.svg)

The left-hand boxes separate current computations from retained arrays. Blue produces a content score for each source token; purple produces its positional score. Follow both arrows into the green box: add, scale and mask before one softmax. Its weights mix C for the value path. The two retained boxes sum to the reference’s 400-byte payload; retaining C alone would omit the positional keys. The diagram shows one retrieval per head, while the labeled cache arrays include both batch items.

### Count a complete cache before reading the bar chart

![A schematic comparison illustrates the motivation for smaller retained latent state](images/mla-8-8.png)

The general payload count for batch B, tokens T, layers L and s bytes per element is

$$\text{MLA cache bytes}=BTL(r+d_r)s.$$

For the executable example below, B=2, T=5, L=1, r=3, $d_r=2$ and float64 give $2\cdot5\cdot(3+2)\cdot8=400$ bytes. Keeping two heads of fully expanded content keys of width two, positional keys of width two and values of width two would use 960 bytes. If the positional key remains shared even in the expanded reference, that alternative uses 800 bytes. State which representation is being compared.

The large bars are a teaching illustration, not a derivation of a particular DeepSeek checkpoint's allocation. Their 80/10 GB labels do not follow from unspecified rank, positional width, layer count or precision. Nor does equal architecture-level “capacity” establish equal benchmark quality. The original DeepSeek-V2 paper reports its own trained-model experiments; our checks establish factorization identities for fixed coefficients. Runtime allocation also includes metadata, workspace and allocator overhead beyond the payload formula.

### Trace a complete reference through both paths

Our input X is `[B,T,4]`. `projections` first computes C `[B,T,3]`, two content queries `[B,2,T,2]`, two positional queries of width two and a shared positional key `[B,T,2]`. `rotate_pair` uses absolute token positions. `attend` either expands keys/values or performs the equivalent latent-domain contractions. Both paths add the same positional scores, divide by two, mask future positions, normalize, join two value heads and apply the same output matrix.

The code fixes dimensions so each axis is visible. It omits the full model's query bottleneck, normalization, feed-forward network, tokenizer and trained coefficients. The gradient comparison is useful precisely because a shape-correct contraction can still wire the wrong head or latent axis.

Save [mla.py](examples/mla.py) and run `python mla.py` in a PyTorch environment. In Colab upload the file, inspect `torch.__version__`, and run `!python mla.py`; CPU float64 is sufficient for this small check.

```python
"""Tiny MLA algebra: expanded and latent-domain retrieval with decoupled RoPE."""
import json
import math
import torch


def rotate_pair(x, positions):
    c,s=positions.to(x).cos(),positions.to(x).sin()
    a,b=x[...,0],x[...,1]
    return torch.stack([a*c-b*s,a*s+b*c],-1)


def projections(x,wd,wq,wqr,wkr):
    b,t,_=x.shape
    latent=x@wd
    q=(x@wq).reshape(b,t,2,2).transpose(1,2)
    qr=(x@wqr).reshape(b,t,2,2).transpose(1,2)
    kr=x@wkr
    positions=torch.arange(t,device=x.device)
    return q,rotate_pair(qr,positions),latent,rotate_pair(kr,positions)


def attend(q,qr,latent,kr,uk,uv,wo,offset=0,absorbed=True):
    t=latent.shape[1];s=q.shape[-2]
    if absorbed:
        query_latent=torch.einsum('bhsd,hrd->bhsr',q,uk)
        content_scores=torch.einsum('bhsr,btr->bhst',query_latent,latent)
    else:
        keys=torch.einsum('btr,hrd->bhtd',latent,uk)
        content_scores=q@keys.transpose(-2,-1)
    scores=(content_scores+torch.einsum('bhsd,btd->bhst',qr,kr))/math.sqrt(2+2)
    query_positions=offset+torch.arange(s,device=q.device)
    future=torch.arange(t,device=q.device)[None,:]>query_positions[:,None]
    weights=scores.masked_fill(future,-torch.inf).softmax(-1)
    if absorbed:
        mixed_latent=torch.einsum('bhst,btr->bhsr',weights,latent)
        context=torch.einsum('bhsr,hrd->bhsd',mixed_latent,uv)
    else:
        values=torch.einsum('btr,hrd->bhtd',latent,uv)
        context=weights@values
    joined=context.transpose(1,2).contiguous().reshape(q.shape[0],s,4)
    return joined@wo


def run_checks():
    torch.manual_seed(71)
    shapes=[(2,5,4),(4,3),(4,4),(4,4),(4,2),(2,3,2),(2,3,2),(4,4)]
    x,wd,wq,wqr,wkr,uk,uv,wo=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in shapes]
    q,qr,c,kr=projections(x,wd,wq,wqr,wkr)
    a=attend(q,qr,c,kr,uk,uv,wo,absorbed=True)
    b=attend(q,qr,c,kr,uk,uv,wo,absorbed=False)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    variables=(x,wd,wq,wqr,wkr,uk,uv,wo)
    ga=torch.autograd.grad(a.square().sum(),variables,retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),variables)
    for left,right in zip(ga,gb):
        torch.testing.assert_close(left,right,atol=1e-9,rtol=1e-9)
    first=attend(q[:,:,:2],qr[:,:,:2],c[:,:2],kr[:,:2],uk,uv,wo)
    later=attend(q[:,:,2:],qr[:,:,2:],c,kr,uk,uv,wo,offset=2)
    torch.testing.assert_close(torch.cat([first,later],1),a)
    assert sum(z.numel()*z.element_size() for z in [c,kr])==400
    wrong=attend(q[:,:,2:],rotate_pair((x@wqr).reshape(2,5,2,2).transpose(1,2)[:,:,2:],torch.arange(3)),c,kr,uk,uv,wo,offset=2)
    assert not torch.allclose(wrong,a[:,2:])
    return {'expanded_absorbed_outputs_and_gradients_equal':True,
            'cached_chunks_equal':True,'latent_plus_position_bytes':400,
            'wrong_rotary_offset_detected':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
```

The program prints that expanded/absorbed outputs and all input/matrix gradients agree, cached chunks agree, payload is 400 bytes and resetting the rotary offset is detected. The chunks use query positions 0–1 and then 2–4. The second chunk sees all five retained rows but its causal mask uses offset two. Resetting its rotations to 0–2 is a different positional computation even if tensor shapes still match.

During training, changing coefficients invalidates precomputed absorbed products; form them from the current coefficients or recompute them appropriately. In serving, cache ownership, absolute positions, truncation and reset remain the contracts from the KV-cache lesson. Smaller storage does not make stale state valid.

[[practice:mla-position]]

### What the identity does and does not buy

The same factorized MLA can be evaluated through expanded or absorbed paths. This does not show that arbitrary MHA coefficients can be compressed into rank r without changing outputs. It also does not guarantee a speedup: latent width, kernel support, batch shape, memory traffic and the cost of additional projections matter. The next lesson keeps the dense attention formula and changes execution order through tiled online softmax.

[Implementation and commands](project/implementation.md) · [Readiness and limits](project/readiness.md) · [Transfer review](practice/transfer-review.md)

### References

- [Vizuara, Inference Engineering workshop book](https://vizuaraai.github.io/convolution-musical-journey/book.html), Chapter 8.7–8.9; original compression and memory figures.
- [DeepSeek-V2](https://arxiv.org/html/2405.04434v5), Sections 2.1.2–2.1.4 and Figure 3; original architecture comparison and decoupled rotary derivation.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
