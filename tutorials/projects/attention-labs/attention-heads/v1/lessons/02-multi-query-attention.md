## Multi-query attention: many questions, one stored memory

MHA gives each query head its own key/value projection. Caching avoids recomputing those projections, but it retains every head's keys and values for every token. Our next problem is the size of that retained memory. Can several independent questions retrieve from one shared representation of the same history?

The change we will make is precise: keep the query heads, share the K/V projection. We will trace the wiring, calculate distinct answers from the shared memory, count parameters and stored bytes, then implement the layer and compare it with an explicit head loop. This is a change to the model's architecture, unlike the equivalent tensor packing in the MHA tutorial.

### Find the factor we can change

![MHA, MQA, GQA and latent attention arrange retained key/value information differently](images/mqa-8-1.png)

Read the query count separately from the number of stored key/value channels. A model can keep eight query heads while storing eight KV heads, two KV groups or one shared KV head. The latent case changes the representation again; we will open that only after understanding sharing.

From the cache lesson, payload is $BTLH_{kv}(d_k+d_v)s$. Holding sequence length, feature widths, layers, batch and dtype fixed, changing $H_{kv}$ changes this payload directly. It does not remove the query heads or the query-token/key-token comparison work.

### Start from the familiar MHA wiring

![MHA maintains an independent key/value lane for every query head](images/mqa-8-2.png)

Each head gets all input features through its own projection. Its output is a mixture over the same source-token positions, using its own learned comparisons and value coordinates. The output projection mixes the concatenated channels at each token. Multiple heads permit distinct patterns; specialization into grammatical roles is not guaranteed.

### Share the memory projection while keeping the queries

![Independent query lanes retrieve from one shared key matrix and one shared value matrix](images/mqa-8-3.png)

The vertical shared K/V blocks represent tensors over **all retained token positions**. They are not one key vector and one value vector for the whole conversation. There is one KV head, containing one key and one value row per retained token. Every query head compares against those same key rows but can assign them different weights.

For query head $h$ and token $i$,

$$a_{h,i}=\operatorname{softmax}\!\left(q_{h,i}K^\top/\sqrt{d_k}+M_i\right),\qquad z_{h,i}=a_{h,i}V.$$

$M_i$ applies the same visibility rule to each head. The shared K/V appear without a head-specific subscript. Query vectors retain the head subscript. Finally concatenate $z_{h,i}$ across h and apply $W_O$.

The picture's arrows describe this retrieval relationship. V is combined with the attention weights, not multiplied directly by K as an extra projection stage. The box's input-projection parameter formula excludes $W_O$; include the output projection when counting the complete layer.

### Calculate different answers from identical keys and values

Use a scalar comparison width, so the scale is one. All four query heads can see three keys $[1,0,2]$ and three values $[2,-1,3]$. At the final position, choose queries 1, 2, −1 and 0. These are declared numerical probes, not trained word meanings.

| Query | Scores across the three keys | Softmax weights | Output |
| --- | --- | --- | --- |
| 1 | `[1,0,2]` | `[0.244728,0.090031,0.665241]` | `2.395149` |
| 2 | `[2,0,4]` | `[0.117310,0.015876,0.866813]` | `2.819185` |
| −1 | `[-1,0,-2]` | `[0.244728,0.665241,0.090031]` | `0.094308` |
| 0 | `[0,0,0]` | `[1/3,1/3,1/3]` | `1.333333` |

For the first head, the value mixture is $2(0.244728)-0.090031+3(0.665241)\approx2.395149$. Sharing memory did not force identical attention patterns. Equal queries would yield equal patterns under these same keys and masks; distinct queries make different patterns possible, not inevitable.



![Four query heads share the same source tokens but assign different probability bars and return different values](images/worked-mqa-patterns.svg)

Each colored panel is one query head, and every panel uses the same source-token order. Compare token 1: its value is negative, so the large middle bar for q = −1 pulls that head’s answer toward −1. The zero query gives three equal bars. Follow the probabilities into the weighted answers before the output projection joins the four channels. This is what the table’s four numerical rows look like as retrieval patterns.

The output projection combines the four resulting channels. It still does not mix source-token rows directly: that mixing happened inside each attention distribution.

[[practice:mqa-patterns]]

### Follow the tensor dimensions into code

For the complete implementation below, model width is 4, query-head count is 4 and each head width is 1. The small dimensions make parameter counting inspectable. With input `[B,T,4]`:

| Step | Shape | Meaning |
| --- | --- | --- |
| Query projection | `[B,T,4]` | All four query-head features |
| Reshape and transpose Q | `[B,4,T,1]` | Four independent query heads |
| Key/value projection and head-axis insertion | `[B,1,T,1]` each | One shared KV head |
| Scores and probabilities | `[B,4,T,T]` | Four token-to-token patterns |
| Contexts | `[B,4,T,1]` | One channel per head and token |
| Merge, then output projection | `[B,T,4]` | Restored model feature width |

Broadcasting the size-one KV-head axis means the same logical K/V are used for all query heads. It does not mean the cache should physically store four copies. Some backends expand or repeat K/V as an intermediate; account for that separately from retained cache bytes.

For bias-free row-vector projections, $W_Q$ has shape $[4,4]$, $W_K,W_V$ each $[4,1]$, and $W_O$ has shape $[4,4]$. Total parameters are $16+4+4+16=40$. PyTorch Linear stores the transposed coefficient arrangement internally. An otherwise corresponding MHA layer has $16+16+16+16=64$ parameters. Output width stays four in both cases.

### Implement both the broadcast and explicit retrieval paths

Save [mqa.py](examples/mqa.py). The `MQA` class defines all projections before use. Its forward pass builds Q, inserts the singleton K/V-head axis, applies the causal rule, retrieves contexts, restores token/head ordering and projects the joined features. The alternative `loop=True` path calculates one query head at a time against the same K/V and parameters.

Run `python mqa.py` in a PyTorch environment. In Google Colab, first inspect `torch.__version__`, upload the file, then run `!python mqa.py`. The author run uses CPU float64 and no data/model downloads. Here is the complete file:

```python
"""Complete illustrative MQA layer; same parameters, broadcast and explicit loop."""
import json
import math
import torch
from torch import nn


class MQA(nn.Module):
    def __init__(self, width=4, heads=4, head_width=1):
        super().__init__()
        self.heads, self.head_width = heads, head_width
        self.wq = nn.Linear(width, heads*head_width, bias=False, dtype=torch.float64)
        self.wk = nn.Linear(width, head_width, bias=False, dtype=torch.float64)
        self.wv = nn.Linear(width, head_width, bias=False, dtype=torch.float64)
        self.wo = nn.Linear(heads*head_width, width, bias=False, dtype=torch.float64)

    def forward(self, x, loop=False):
        b, t, _ = x.shape
        q = self.wq(x).reshape(b,t,self.heads,self.head_width).transpose(1,2)
        k, v = self.wk(x).unsqueeze(1), self.wv(x).unsqueeze(1)
        future = torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
        if loop:
            weights = torch.stack([
                ((q[:,h] @ k[:,0].transpose(-2,-1))/math.sqrt(self.head_width))
                .masked_fill(future,-torch.inf).softmax(-1)
                for h in range(self.heads)], dim=1)
            context = torch.stack([weights[:,h] @ v[:,0] for h in range(self.heads)], dim=1)
        else:
            scores = q @ k.transpose(-2,-1) / math.sqrt(self.head_width)
            weights = scores.masked_fill(future,-torch.inf).softmax(-1)
            context = weights @ v
        joined = context.transpose(1,2).contiguous().reshape(b,t,-1)
        return self.wo(joined), weights, k, v


def run_checks():
    torch.manual_seed(31)
    layer = MQA()
    x = torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
    y, weights, k, v = layer(x)
    loop, _, _, _ = layer(x,loop=True)
    torch.testing.assert_close(y,loop,atol=1e-12,rtol=1e-12)
    parameters = (x,*layer.parameters())
    a = torch.autograd.grad(y.square().sum(),parameters,retain_graph=True)
    b = torch.autograd.grad(loop.square().sum(),parameters)
    for left,right in zip(a,b):
        torch.testing.assert_close(left,right,atol=1e-12,rtol=1e-12)
    changed = x.detach().clone();changed[:,2] += 100
    torch.testing.assert_close(layer(changed)[0][:,:2],y.detach()[:,:2])
    assert not torch.allclose(weights[:,0,-1],weights[:,1,-1])
    assert k.shape == v.shape == (2,1,3,1)
    assert sum(p.numel() for p in layer.parameters()) == 40
    return {'loop_and_gradients_equal':True,'causal_intervention':True,
            'shared_kv_shape':[2,1,3,1],'distinct_query_patterns':True,'parameters':40}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
```

The output confirms 40 parameters and actual shared K/V shape `[2,1,3,1]`. The broadcast and explicit-loop outputs agree. Differentiating the same squared-output objective gives matching input and all-projection gradients; this checks that shared K/V gradients collect contributions from all query heads. Changing the final input cannot alter the first two causal outputs. The two tested query patterns differ even though their K/V are shared.

This program checks the MQA component. It does not load or convert an MHA checkpoint, implement a generation loop or establish model quality. Combine its head relationship with the cache lesson's append/position contract when adding generation.

### Count the cache saving without claiming a speed result

For $B=1,T=3,L=1,d_k=d_v=1,s=2$, four-head MHA retains $1\cdot3\cdot1\cdot4\cdot2\cdot2=48$ bytes. MQA retains 12 bytes. At fixed head width, the MHA-to-MQA payload ratio is $H_q$; with 64 query heads it is 64, not universally 32. These values exclude temporary tensors and allocation overhead.

Each query head still creates scores over permitted keys, so sharing alone does not eliminate dense attention's pairwise dependence on sequence length. Cache bandwidth, kernel support and batching determine whether a measured workload gets faster. Architectural sharing can affect learned quality; averaging existing MHA K/V weights is not an algebraically exact conversion for arbitrary models. Model adaptation and evaluation are separate work.

[[practice:mqa-memory]]

### The next unresolved choice

One shared KV head is the strongest sharing among these conventional head arrangements. Could we retain several KV representations and share only within groups? GQA will give that mapping an explicit group index, then check MHA and MQA as its two endpoints.

The [implementation notes](project/implementation.md), [readiness record](project/readiness.md) and [transfer review](practice/transfer-review.md) remain available for returning to the route.

### References

[Vizuara: Multi-Query Attention Explained](https://www.youtube.com/watch?v=Z6B51Odtn-Y). [Vizuara Inference Engineering, Chapter 8](https://vizuaraai.github.io/convolution-musical-journey/book.html): original workshop figures and head-sharing progression. [Shazeer, Fast Transformer Decoding, version 1, Sections 2–3](https://arxiv.org/html/1911.02150v1): primary definition and incremental-memory analysis. Workshop figures are separate from the video page; this lesson does not assert a complete subtitle audit of that older video.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
