## PagedAttention: separate token order from storage order

A server does not know how long every request will become. Reserving each request's maximum possible cache can leave large amounts of device memory unavailable to others. PagedAttention breaks the cache into fixed-size blocks and keeps a small mapping from logical token order to physical storage. The attention function stays the same; allocation, sharing and lifetime become explicit.

### Find the unused reservation before changing the attention math

![Four maximum-sized reservations leave capacity unavailable to new requests](images/paged-01.png)

Three costs should be distinguished: space reserved for possible future tokens, unused capacity inside an allocation, and free gaps between allocations that cannot satisfy a large contiguous request. These are related but not identical. The diagram's blanket external-fragmentation label conflates them.

Its numbers also need a consistent basis. At equal 4096-token reservations and identical bytes per token, actual lengths512,3800,1024,200 use $5536/16384\approx33.79$ percent of the reserved space, not22 percent. The allocation problem is real without relying on the inconsistent total printed in the picture.

### Borrow address translation, not an automatic operating-system implementation

![A logical-to-physical table connects token blocks to scattered KV storage](images/paged-02.png)

The left diagram shows the general virtual-memory idea; the right applies indirection to a model's cache. A request sees tokens0,1,2,… in sequence order. A block table maps each group of P tokens to a physical block identifier. The serving engine and kernel manage that table; this does not imply an OS page fault or MMU translation for every token lookup.

![Equal-sized physical blocks can be assigned to requests without a large contiguous request allocation](images/paged-03.png)

Any free block of the right shape can extend a request. Larger batches may fit because the engine avoids reserving every request's worst-case length. The illustrated utilization percentage is workload-dependent. Paging still wastes the unused tail of a last block and needs table/allocator metadata. A request should not list the same physical block twice for different logical content; repeated IDs in a schematic table are not an instruction to alias unrelated tokens.

### Translate one address completely

![Logical token groups map through a block table to physical blocks](images/paged-04.png)

For zero-based token t,

$$b=\lfloor t/P\rfloor,\qquad o=t\bmod P,\qquad p=\text{table}[b].$$

Read physical block p at offset o. In the figure, P=4 and the table is `[37,12,91]`, so token11 is logical block2, physical91, offset3. If P were16, token11 would instead be logical block0, offset11. Do not mix the four-token figure with the sixteen-token convention used elsewhere in the workshop. Physical block IDs are addresses, not positional coordinates for RoPE or causal masks.



![Token eleven splits into logical block two and offset three, then maps through the table to physical block ninety one](images/worked-paged-address.svg)

Follow the two-part address through the top row. Only the logical block number goes through the table; the offset remains 3. In the physical block, slot 3 contains the KV for absolute token 11. Neither the allocation ID nor the offset replaces its absolute position for masks or RoPE. The table and four cells fully specify this lookup.

### Allocate at a boundary and mask the incomplete tail

![A request obtains blocks as its logical sequence grows](images/paged-05.png)

With P=4, lengths0,4,9,13 require0,1,3,4 blocks. With P=8, they require0,1,2,2. The picture's0,1,2,3 progression does not match one fixed P for those lengths. The rule is always $\lceil T/P\rceil$, allocating the next block only when needed.

For T=9,P=4, capacity is12 and three positions are unused. They must never enter the softmax denominator or the value mixture, even if their storage happens to contain zeros. Zero data is not a mask. In general, last-block waste is $(P-(T\bmod P))\bmod P$, strictly less than P positions per unshared request tail.

[[practice:paged-address]]

### Fetch the logical prefix through the table

![A query retrieves key/value blocks in logical order despite scattered physical addresses](images/paged-06.png)

The same scaled dot products, permitted positions and value mixture apply after address translation. A dense teaching reference can gather all valid entries into a contiguous tensor first. A real paged kernel reads physical blocks directly without that full gather. Correct addressing is necessary for both; the gather's allocation and runtime are not representative of the production kernel.

![Block-local softmax contributions are merged across scattered blocks](images/paged-07.png)

The equations printed inside this drawing are not a valid recurrence. Use the common-maximum merge established in FlashAttention: for old state m,l,u and a new tile with local maximum $m_b$, set $m'=\max(m,m_b)$, then

$$l'=e^{m-m'}l+\sum_{j\in B}e^{s_j-m'},\qquad u'=e^{m-m'}u+\sum_{j\in B}e^{s_j-m'}v_j.$$

Finally return u/l. Include only valid logical tokens; exclude future positions when processing more than a final decode query. Scattered physical addresses do not change these formulas. PagedAttention and FlashAttention are related execution ideas, but the allocator, cache lifetime and prefix-sharing mechanisms are additional responsibilities.

### Share a prefix without sharing future mutations

![Multiple requests reference shared prefix blocks and privately retain diverging suffixes](images/paged-08.png)

Keep one physical copy of a valid common prefix and let each request's table reference it. Increment a reference count for every table that owns the block. When one request finishes, decrement counts and return a block to the free pool only when its count reaches zero.

The subtle case is a **partially filled shared last block**. Suppose P=4, request A has six tokens and table `[4,1]`. Fork B with the same six-token prefix; both blocks now have refcount2. B's seventh token belongs in offset2 of the second block. Clone that partial block into a new physical block3 first, copy its two valid entries, change B's table to `[4,3]`, decrement block1's count, then write B's new entry. A still sees six unchanged tokens through `[4,1]`.

A full shared block need not be copied merely because a request appends after it; allocate a new suffix block instead. Sharing also requires identical model parameters, tokenized prefix, relevant positions and runtime configuration. Equal text under different tokenization or adapters does not prove equal cached tensors. Our `fork` operates on an already identical prefix; it is not a general cross-request prefix matching service.



![A fork shares two prefix blocks, a partial-tail append creates a private copy, and subsequent allocation preserves request ownership](images/worked-paged-copy-on-write.svg)

The three panels are consecutive states of the executable pool example. Each row is a physical block, with its reference count above four token slots. Start with two shared blocks. When B appends token 6, it copies the two valid tail entries into block 3 before writing; A still points to unchanged block 1. At nine tokens B fills block 3 and allocates block 0. The final block’s three dots are spare capacity, not valid zero-valued tokens. The release line follows the reference counts so shared prefix block 4 stays alive until B also releases it.

### Run the ownership lifecycle before discussing capacity gains

Download [paged.py](examples/paged.py), run `python paged.py` with PyTorch, or upload it to Colab and run `!python paged.py`. CPU float64 suffices. `Request` owns logical length/table; `Pool` owns physical blocks, reference counts and the free list. All helpers are defined before the worked lifecycle.

```python
"""Logical cache pages, reference counts and copy-on-write; CPU teaching model."""
from dataclasses import dataclass,field
import json
import math
import torch


@dataclass
class Request:
    table:list=field(default_factory=list)
    length:int=0
    active:bool=True


class Pool:
    def __init__(self,block_size=4):
        self.block_size=block_size;self.blocks={};self.refs={}
        self.free=[4,1,3,0,2]

    def allocate(self):
        if not self.free:raise MemoryError('No free physical block')
        block=self.free.pop(0)
        self.blocks[block]=[None]*self.block_size;self.refs[block]=1
        return block

    def append(self,request,key,value):
        if not request.active:raise ValueError('Closed request')
        logical,offset=divmod(request.length,self.block_size)
        if offset==0:
            block=self.allocate();request.table.append(block)
        else:
            block=request.table[logical]
            if self.refs[block]>1:
                old=block;block=self.allocate()
                self.blocks[block][:offset]=list(self.blocks[old][:offset])
                self.refs[old]-=1;request.table[logical]=block
        self.blocks[block][offset]=(key.clone(),value.clone())
        request.length+=1

    def fork(self,request):
        if not request.active:raise ValueError('Closed request')
        for block in request.table:self.refs[block]+=1
        return Request(list(request.table),request.length)

    def materialize(self,request):
        if not request.active or not request.length:raise ValueError('No active tokens')
        rows=[]
        for token in range(request.length):
            logical,offset=divmod(token,self.block_size)
            rows.append(self.blocks[request.table[logical]][offset])
        return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows])

    def release(self,request):
        if not request.active:raise ValueError('Already closed')
        for block in request.table:
            self.refs[block]-=1
            if self.refs[block]==0:
                self.blocks[block]=[None]*self.block_size
                self.free.append(block)
        request.active=False;request.table.clear();request.length=0


def attention(query,k,v):
    return (query@k.T/math.sqrt(k.shape[-1])).softmax(-1)@v


def run_checks():
    torch.manual_seed(127)
    k=torch.randn(9,2,dtype=torch.float64);v=torch.randn(9,3,dtype=torch.float64)
    query=torch.randn(2,dtype=torch.float64)
    pool=Pool();a=Request()
    for i in range(6):pool.append(a,k[i],v[i])
    assert a.table==[4,1]
    b=pool.fork(a);pool.append(b,k[6],v[6])
    assert a.table==[4,1] and b.table==[4,3]
    ak,av=pool.materialize(a);torch.testing.assert_close(ak,k[:6]);torch.testing.assert_close(av,v[:6])
    for i in [7,8]:pool.append(b,k[i],v[i])
    bk,bv=pool.materialize(b)
    torch.testing.assert_close(attention(query,bk,bv),attention(query,k,v))
    assert b.table==[4,3,0] and 0<=len(b.table)*4-b.length<4
    pool.release(a);assert pool.refs[4]==1
    pool.release(b);assert len(pool.free)==5 and all(x==0 for x in pool.refs.values())
    fresh=Request()
    for _ in range(20):pool.append(fresh,k[0],v[0])
    fk,fv=pool.materialize(fresh)
    torch.testing.assert_close(fk,k[0].expand(20,2));torch.testing.assert_close(fv,v[0].expand(20,3))
    before=(list(fresh.table),fresh.length,dict(pool.refs))
    try:pool.append(fresh,k[1],v[1])
    except MemoryError:pass
    else:raise AssertionError('Expected pool exhaustion')
    assert before==(fresh.table,fresh.length,pool.refs)
    try:pool.materialize(a)
    except ValueError:pass
    else:raise AssertionError('Closed request accepted')
    return {'noncontiguous_table':[4,3,0],'paged_contiguous_output_equal':True,
            'partial_tail_copy_on_write':True,'reclaimed_blocks_reused_without_old_tokens':True,
            'exhaustion_preserves_request':True,'last_block_waste':3}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
```

The program creates A's `[4,1]` mapping, forks B, copies B's partial tail and grows B to `[4,3,0]` at nine tokens. Gathered paged retrieval matches the contiguous reference. A's prefix remains unchanged. Releasing A preserves block4 while B still references it; releasing B returns all blocks. A fresh20-token request fills every block, proving reuse reads only new entries. A failed21st append leaves the request/table/counts unchanged, and a closed request cannot be read.

This single-process teaching pool is not thread-safe or an actual GPU allocator. It supports append and whole-prefix forks, not arbitrary edits, swapping, persistent prefix hashing or scheduler preemption. Those need additional contracts. The output comparison verifies the final decode query over its valid prefix; it does not benchmark a fused paged kernel.

### Read capacity and roofline bars as conditional consequences

![Allocation and packing can change how much capacity remains for concurrent sequences](images/paged-09.png)

The bars illustrate reservation versus use, but neither a fixed user count nor an eightfold throughput gain follows from paging alone. Context lengths, KV head count, dtype, weight allocation, workspace and scheduling determine capacity. The100-GB schematic is not a verified configuration of a particular accelerator. Retained KV bytes per logical token are unchanged unless another compression method is applied.

![Larger effective batches can amortize some model-weight traffic across more decoded tokens](images/paged-10.png)

If improved packing permits a scheduler to run a larger useful batch, model-weight reads can be shared across more token computations. That is the route from allocation to throughput; the address table itself does not multiply FLOPs per byte automatically. Continuous batching and paging can complement each other, but continuous batching is not mathematically dependent on this one allocator design.

Local pointer/count/output checks, measured server capacity, per-user latency, deployed vLLM behavior and hosted execution remain separate results. The final topic moves blocks between devices while preserving the same global normalization.

[[practice:paged-share]]

[Implementation and commands](project/implementation.md) · [Readiness and limits](project/readiness.md) · [Transfer review](practice/review.md)

### References

- [Vizuara workshop book](https://vizuaraai.github.io/convolution-musical-journey/book.html), Chapter11 and all ten original allocation/translation/kernel/sharing/utilization figures.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/html/2309.06180v1), Sections3.1 and4.1–4.4, block tables, reference counts and partial-block copy-on-write.
- Additional worked SVG figures are original to this tutorial; their inputs and calculations follow the complete executable reference above.
