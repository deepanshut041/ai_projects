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
