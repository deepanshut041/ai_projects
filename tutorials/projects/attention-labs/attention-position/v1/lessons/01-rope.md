## RoPE: make the query-key comparison depend on distance

Attention already knows what content is available and which positions it is allowed to inspect. A different question remains: how does a query-key comparison express where two tokens sit relative to each other? We will transform the query and key coordinates before taking their dot product. The transformation should preserve vector length and make a shared shift of the two positions cancel from that comparison.

We will begin with the ordering problem, rotate a two-number vector by hand, calculate one score at two different position pairs, then build a complete multi-pair implementation and check cached positions. The worked unit-vector example stays fixed throughout. This is a new topic sourced from Vizuara's public RoPE explanation; it does not restore the omitted Attention Family Lecture 9.

### Separate content, visibility and position

![The same content tokens swap slots while a position-free pair comparison stays fixed](images/rope-01.png)

“Dog bites man” and “man bites dog” contain the same words, but their roles differ. If Q/K depend only on token identity, the content comparison for the same word pair has no separate position input. Unmasked, position-free self-attention is permutation equivariant: reordering inputs reorders its outputs. A causal mask already introduces ordered visibility, so the same unrestricted permutation argument cannot be applied while silently keeping a fixed triangle. RoPE supplies an explicit positional transformation within that visibility rule.

### Change direction rather than add a position vector

![Adding a positional vector changes a sum, while rotation turns the content vector without changing its length](images/rope-02.png)

Learned absolute vectors, sinusoidal additions and relative score biases are other valid choices with different contracts. Learned tables need an extension policy beyond their allocated positions. Relative biases can be implemented efficiently too; they are not inherently incompatible with fast attention. RoPE's particular choice is to rotate paired query/key coordinates using their positions.

### Understand a single two-dimensional rotation

![A unit vector visits successive angles while its length stays one](images/rope-03.png)

With column-vector notation, rotating by angle a uses

$$R(a)=\begin{bmatrix}\cos a&-\sin a\\\sin a&\cos a\end{bmatrix}.$$

Thus $[x,y]$ becomes $[x\cos a-y\sin a,\;x\sin a+y\cos a]$ when written as the resulting coordinate list. Position m sets the angle $a=m\theta$. For the first pair, take $\theta=1$ radian per position. Position 0 leaves `[1,0]` fixed; position 1 gives approximately `[0.540302,0.841471]`; position 3 gives `[-0.989992,0.141120]`. Squared length remains $\cos^2a+\sin^2a=1$.

### Rotate the query and key separately

![The query rotates to position three and the key to position one](images/rope-04.png)

Keep the unrotated content vectors fixed at $q=k=[1,0]$. Put the query at m=3 and the key at n=1. Their rotated coordinates are $R(3)q$ and $R(1)k$. The relative angle is two radians even though neither rotated vector uses the number two directly.

![The two coordinate products add to the cosine of the positional gap](images/rope-05.png)

Calculate the dot product:

$$\cos3\cos1+\sin3\sin1=\cos(3-1)=\cos2\approx-0.416147.$$

The first product is approximately −0.534895 and the second +0.118748. Their sum is the raw dot product; an attention layer would additionally divide by $\sqrt{d_k}$ before softmax. A negative raw score is valid: exponentiation still gives a positive normalization contribution.

### Move both positions while retaining the same gap

![Moving both tokens together retains their separation and the same comparison](images/rope-06.png)

Move the query/key pair to (5,3). The coordinates are approximately `[0.283662,-0.958924]` and `[-0.989992,0.141120]`. Their dot product is still −0.416147. The pair (100,98) gives the same result up to numerical roundoff. This holds with the unrotated content vectors fixed; it is not a claim that shifting arbitrary text in an entire trained decoder leaves every hidden state unchanged.

![Subtracting a common angle leaves only the relative rotation between query and key](images/rope-07.png)

The identity behind that numerical observation is

$$\begin{aligned}(R(m\theta)q)^\top(R(n\theta)k)
&=q^\top R(m\theta)^\top R(n\theta)k\\
&=q^\top R((n-m)\theta)k.\end{aligned}$$

A rotation transpose is its inverse. Composing two rotations adds their angles, so the inverse contributes the minus sign. The two absolute position terms enter only through their difference. General content vectors still contribute their magnitudes and directions: a RoPE score is not merely a function of distance independent of content.



![Three common position shifts rotate both unit vectors while preserving the two-radian angle and raw dot product](images/worked-rope-common-shift.svg)

Use the same horizontal and vertical axes in all three panels. Both arrows move when the absolute positions change, while the orange arc between them stays two radians. Their dot product therefore stays the same. These are the fixed-content numerical probes used by the reference; the figure does not predict that an entire shifted sentence has identical hidden states. An attention score still applies its comparison-width scaling after this raw dot product.

[[practice:rope-gap]]

### Give different coordinate pairs different frequencies

![Several coordinate pairs rotate at different rates](images/rope-08.png)

A width-d head is split into d/2 pairs, requiring an even rotated width in this reference. Pair r uses

$$\theta_r=b^{-2r/d},\qquad r=0,1,\ldots,d/2-1.$$

For d=4 and base b=10000, the two frequencies are 1 and 0.01 radians per position. At position 3, their angles are 3 and 0.03. For d=8, four pairs use 1, 0.1, 0.01 and 0.001. These multiple rates let comparisons represent several distance scales. A clock phase is periodic, so one pair alone cannot identify arbitrarily many positions uniquely.

![Independent two-by-two blocks form the conceptual rotary transformation](images/rope-09.png)

The block-diagonal matrix is a mathematical description. Each pair interacts only with its partner; no off-diagonal pair mixing is needed. We implement the coordinate formulas directly, avoiding a large dense rotation matrix. The program separately builds explicit blocks as an independent reference and compares the derivatives.

### Apply the transformation at the right location

![Only query and key paths receive positional rotations before attention scores](images/rope-10.png)

Project the layer input into Q/K/V, rotate Q and K with their own positions, then calculate scaled scores and apply visibility. In this formulation V is unchanged. GQA's shared key head is rotated once per token and reused by its query group. Query-head count and KV-head count need not agree; the positional frequency contract for corresponding comparison coordinates must agree.

For cached generation, storing already-rotated keys is convenient. Prefill positions 0–2, then process new tokens at 3 and 4. Do not restart their position IDs at 0 and 1, and do not rotate stored keys a second time when appending. That changes their relative geometry even if every tensor shape remains legal. A different context-rescaling policy may invalidate an existing cache unless the runtime explicitly supports adapting it.

### Understand what the distance picture does and does not prove

![An illustrative aggregate curve shows distance-dependent oscillation across frequency pairs](images/rope-11.png)

RoPE changes how distance enters scores, but it does not guarantee that every individual score decreases monotonically as distance grows. Our one-pair example already refutes that: $\cos6\approx0.960170$ exceeds $\cos3\approx-0.989992$. Content-dependent sums and their bounds can show aggregate decay tendencies under stated assumptions; they do not turn every attention row into a monotone distance penalty.

### Run a complete implementation

Save [rope.py](examples/rope.py). The `rotate` function receives tensors ending in `[T,D]` and one absolute position per token. Leading batch/head axes broadcast over the same position/frequency grid. It reshapes adjacent pairs, applies the two coordinate formulas, and restores the original shape. Other models use a half-split pair convention; coefficients and permutation must match the checkpoint's convention.

Use `python rope.py` with PyTorch available. In Google Colab, inspect the installed Torch version, upload the file, then run `!python rope.py`. The reference needs no model download or GPU:

```python
"""Adjacent-pair RoPE: norms, relative positions, explicit blocks and cache offsets."""
import json
import math
import torch


def rotate(x, positions, base=10000.0):
    d = x.shape[-1]
    if d % 2 or len(positions) != x.shape[-2] or base <= 0:
        raise ValueError('Even rotated width, matching positions and positive base required')
    frequencies = base ** (-torch.arange(0,d,2,dtype=x.dtype,device=x.device)/d)
    angles = positions.to(dtype=x.dtype,device=x.device)[:,None] * frequencies[None,:]
    pairs = x.reshape(*x.shape[:-1],d//2,2)
    a,b = pairs[...,0],pairs[...,1]
    c,s = angles.cos(),angles.sin()
    return torch.stack([a*c-b*s,a*s+b*c],dim=-1).flatten(-2)


def run_checks():
    unit = torch.tensor([[[[1.0,0.0]]]],dtype=torch.float64)
    scores = []
    for m,n in [(3,1),(5,3),(100,98)]:
        q = rotate(unit,torch.tensor([m]))
        k = rotate(unit,torch.tensor([n]))
        score = (q*k).sum().item()
        assert math.isclose(score,math.cos(2),abs_tol=1e-12)
        scores.append(round(score,6))
    torch.manual_seed(53)
    x = torch.randn(2,3,6,4,dtype=torch.float64,requires_grad=True)
    pos = torch.arange(6)
    actual = rotate(x,pos)
    torch.testing.assert_close(actual.norm(dim=-1),x.norm(dim=-1))
    matrices = []
    for t in pos.tolist():
        blocks = []
        for f in [1.0,0.01]:
            c,s = math.cos(t*f),math.sin(t*f)
            blocks.append(torch.tensor([[c,-s],[s,c]],dtype=x.dtype))
        matrices.append(torch.block_diag(*blocks))
    reference = torch.einsum('tij,bhtj->bhti',torch.stack(matrices),x)
    torch.testing.assert_close(actual,reference)
    left = torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    right = torch.autograd.grad(reference.square().sum(),x)[0]
    torch.testing.assert_close(left,right)
    chunks = torch.cat([rotate(x[:,:,:2],pos[:2]),rotate(x[:,:,2:],pos[2:])],dim=2)
    torch.testing.assert_close(chunks,actual)
    wrong = rotate(x[:,:,2:],torch.arange(4))
    assert not torch.allclose(wrong,actual[:,:,2:])
    # One pair oscillates; position separation does not guarantee monotonic scores.
    assert math.cos(6) > math.cos(3)
    try:
        rotate(torch.ones(1,1,2,3),torch.arange(2))
    except ValueError:
        pass
    else:
        raise AssertionError('Odd rotated width accepted')
    return {'equal_gap_scores':scores,'norm_and_block_gradients_equal':True,
            'cached_positions_equal':True,'reset_position_bug_detected':True,
            'odd_width_rejected':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
```

All three equal-gap scores print −0.416147. The random multi-head fixture preserves norms, equals explicit block-matrix multiplication and has matching input gradients. Splitting the token sequence into two chunks while retaining absolute positions agrees with one full rotation. Resetting the second chunk's positions is deliberately detected; an odd rotated width is rejected. A partial-RoPE model could leave an unrotated suffix, but that is a separate interface choice from accepting an invalid paired width silently.

[[practice:rope-cache]]

### Keep quality and context extension as separate evidence

![An external experiment chart compares positional methods under its own training conditions](images/rope-12.png)

The plotted quality numbers describe source experiments, not this tiny geometry check. To reproduce a ranking, fix architecture, corpus, training budget, context, evaluation and positional method. Having no learned positional parameters does not imply no execution cost: sine/cosine handling and pairwise rotation still require arithmetic and memory access.

![Context-extension diagrams contrast unmodified positions with a rescaled positional schedule](images/rope-13.png)

RoPE can compute rotations for position IDs beyond the training length. That mathematical availability does not guarantee useful trained-model behavior there. The diagram's “scores explode” label is not a consequence of rotation alone: preserved norms give the bound $|q^\top R(a)k|\leq\lVert q\rVert\lVert k\rVert$. Unfamiliar phases can harm learned behavior without an unbounded raw dot product. Position interpolation and frequency-rescaling methods change the positional schedule and require their own model/evaluation contract. We have not reproduced their training or promised unlimited context.

ALiBi will open a contrasting design: keep Q/K coordinates as they are and add a distance-dependent term directly to the attention logits. Then MLA will use this understanding of positional transformations to explain why its content and positional key paths are separated.

The [implementation notes](project/implementation.md), [readiness checks](project/readiness.md) and [transfer review](practice/review.md) support returning to this topic.

### References

[Vizuara: What exactly is Rotary Position Embedding?](https://vizuara.medium.com/what-exactly-is-rotary-position-embedding-rope-59d616f02e0d): original diagrams and the two-dimensional teaching progression, with numerical/claim qualifications above. [Su et al., RoFormer, version 5, Section 3](https://arxiv.org/html/2104.09864v5): rotation construction, relative-position identity and scoped decay analysis. The complete source animations remain available at the original article.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
