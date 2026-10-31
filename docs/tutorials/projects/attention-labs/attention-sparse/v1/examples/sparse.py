"""Masks, selected-key execution, receptive fields and a bounded decode cache."""
import json
import math
import torch


def visibility(n,window,globals=(),causal=True,dilation=1):
    i=torch.arange(n)[:,None];j=torch.arange(n)[None,:]
    distance=i-j
    local=((distance>=0)&(distance<window)) if causal else (distance.abs()<window)
    if dilation>1:
        local=((distance>=0)&(distance<=dilation*(window-1))&(distance%dilation==0))
    g=torch.tensor([x in globals for x in range(n)])
    allowed=local|g[None,:]|g[:,None]
    return allowed&(j<=i) if causal else allowed


def selected(q,k,v,mask):
    out=[]
    for i in range(len(q)):
        ids=mask[i].nonzero().flatten()
        if not len(ids):raise ValueError('No visible keys')
        out.append((q[i]@k[ids].T/math.sqrt(q.shape[-1])).softmax(-1)@v[ids])
    return torch.stack(out)


def run_checks():
    torch.manual_seed(83)
    q,k,v=[torch.randn(8,2,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    counts=[]
    for globals in [(),(0,)]:
        mask=visibility(8,4,globals)
        dense=(q@k.T/math.sqrt(2)).masked_fill(~mask,-torch.inf).softmax(-1)@v
        actual=selected(q,k,v,mask)
        torch.testing.assert_close(actual,dense)
        ga=torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
        gb=torch.autograd.grad(dense.square().sum(),(q,k,v),retain_graph=True)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b)
        counts.append(int(mask.sum()))
    ring={};decoded=[];peak=0
    for i in range(8):
        ring[i%4]=(i,k[i],v[i])
        retained={pos:(key,value) for pos,key,value in ring.values()}
        retained[0]=(k[0],v[0])
        ids=sorted(retained);peak=max(peak,len(ids))
        keys=torch.stack([retained[j][0] for j in ids]);values=torch.stack([retained[j][1] for j in ids])
        decoded.append((q[i]@keys.T/math.sqrt(2)).softmax(-1)@values)
    torch.testing.assert_close(torch.stack(decoded),selected(q,k,v,visibility(8,4,(0,))))
    assert peak==5
    reach=visibility(16,4)
    total=torch.eye(16,dtype=torch.bool)
    for _ in range(3):total=(total.to(torch.int64)@reach.to(torch.int64))>0
    assert total[12].nonzero().flatten().tolist()==list(range(3,13))
    assert visibility(8,4,(0,),False)[0,7] and not visibility(8,4,(0,),True)[0,7]
    changed=v.detach().clone();changed[7]+=100
    mask=visibility(8,4,(0,))
    torch.testing.assert_close(selected(q,k,changed,mask)[:7],selected(q,k,v,mask)[:7])
    return {'window_and_global_pairs':counts,'dense_causal_pairs':36,'peak_retained_tokens':peak,
            'three_layer_receptive_positions':list(range(3,13)),
            'outputs_gradients_cache_and_future_checks':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
