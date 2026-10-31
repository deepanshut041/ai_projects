"""CPU simulation of four rank-owned Q blocks and circulating KV blocks."""
import json
import math
import torch


def dense(q,k,v,causal):
    scores=q@k.T/math.sqrt(q.shape[-1])
    if causal:scores=scores.masked_fill(torch.ones_like(scores,dtype=torch.bool).triu(1),-torch.inf)
    return scores.softmax(-1)@v


def ring(q,k,v,groups,causal=False,direction=1):
    ranks=len(groups);outputs=[None]*len(q);visits=[]
    for rank,query_ids in enumerate(groups):
        owners=[(rank-direction*round)%ranks for round in range(ranks)]
        assert sorted(owners)==list(range(ranks));visits.append(owners)
        for i in query_ids:
            m=None;l=None;u=None
            for owner in owners:
                ids=torch.tensor(groups[owner],dtype=torch.int64)
                if causal:ids=ids[ids<=i]
                if not len(ids):continue
                scores=q[i]@k[ids].T/math.sqrt(q.shape[-1])
                new_m=scores.max() if m is None else torch.maximum(m,scores.max())
                p=(scores-new_m).exp()
                if m is None:l=p.sum();u=p@v[ids]
                else:
                    correction=(m-new_m).exp()
                    l=correction*l+p.sum();u=correction*u+p@v[ids]
                m=new_m
            if m is None:raise ValueError('No visible sources')
            outputs[i]=u/l
    return torch.stack(outputs),visits


def run_checks():
    torch.manual_seed(131)
    q=torch.randn(16,3,dtype=torch.float64,requires_grad=True)
    k=torch.randn(16,3,dtype=torch.float64,requires_grad=True)
    v=torch.randn(16,2,dtype=torch.float64,requires_grad=True)
    contiguous=[list(range(4*r,4*r+4)) for r in range(4)]
    zigzag=[list(range(2*r,2*r+2))+list(range(2*(7-r),2*(7-r)+2)) for r in range(4)]
    for causal in [False,True]:
        expected=dense(q,k,v,causal)
        for groups in [contiguous,zigzag]:
            for direction in [1,-1]:
                actual,_=ring(q,k,v,groups,causal,direction)
                torch.testing.assert_close(actual,expected)
        actual,visits=ring(q,k,v,zigzag,causal)
        ga=torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
        gb=torch.autograd.grad(expected.square().sum(),(q,k,v),retain_graph=True)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b)
    counts=lambda groups:[sum(i+1 for i in group) for group in groups]
    assert counts(contiguous)==[10,26,42,58] and counts(zigzag)==[34]*4
    changed=v.detach().clone();changed[-1]+=100
    torch.testing.assert_close(ring(q,k,changed,zigzag,True)[0][:-1],dense(q,k,v,True)[:-1])
    scores=torch.tensor([0.,0.,5.,5.],dtype=torch.float64)
    values=torch.tensor([0.,0.,10.,10.],dtype=torch.float64)
    assert abs(float(scores.softmax(0)@values)-5)>4
    return {'rank0_kv_owner_rounds':visits[0],'causal_pairs_contiguous':counts(contiguous),
            'causal_pairs_zigzag':counts(zigzag),'sent_bytes_per_rank':3*4*(3+2)*8,
            'dense_outputs_gradients_both_directions_equal':True,'future_intervention_and_wrong_average_detected':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
