"""Online tiled softmax mathematics; this is not a CUDA kernel."""
import json
import math
import torch


def dense(q,k,v,causal=False,offset=0):
    scores=q@k.T/math.sqrt(q.shape[-1])
    if causal:
        mask=torch.arange(k.shape[0])[None,:]>offset+torch.arange(q.shape[0])[:,None]
        scores=scores.masked_fill(mask,-torch.inf)
    return scores.softmax(-1)@v


def online(q,k,v,block=4,causal=False,offset=0,reverse=False):
    outputs=[]
    starts=list(range(0,k.shape[0],block))
    if reverse:starts.reverse()
    for i,query in enumerate(q):
        m=None;total=None;numerator=None
        for start in starts:
            stop=min(start+block,k.shape[0])
            ids=torch.arange(start,stop)
            allowed=(ids<=i+offset) if causal else torch.ones_like(ids,dtype=torch.bool)
            if not allowed.any():continue
            keys=k[start:stop][allowed];values=v[start:stop][allowed]
            scores=query@keys.T/math.sqrt(q.shape[-1])
            tile_m=scores.max()
            new_m=tile_m if m is None else torch.maximum(m,tile_m)
            weights=(scores-new_m).exp()
            if m is None:
                total=weights.sum();numerator=weights@values
            else:
                correction=(m-new_m).exp()
                total=correction*total+weights.sum()
                numerator=correction*numerator+weights@values
            m=new_m
        if m is None:raise ValueError('Query has no visible keys')
        outputs.append(numerator/total)
    return torch.stack(outputs)


def run_checks():
    q=torch.tensor([[1,.5,0,.2],[.3,1,.4,.1],[.7,.2,.8,.5],[.1,.6,.3,.9],
                    [.5,.3,.7,.4],[.8,.1,.2,.6],[.2,.7,.5,.3],[.4,.5,.1,.8]],dtype=torch.float64,requires_grad=True)
    k=torch.tensor([[.8,.3,.1,.4],[.2,.9,.5,.3],[.6,.1,.7,.8],[.4,.7,.2,.6],
                    [.9,.4,.3,.1],[.1,.8,.6,.5],[.5,.2,.4,.7],[.3,.6,.8,.2]],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([[1,0,.5,.2],[.3,.8,.1,.6],[.7,.3,.9,.1],[.2,.5,.4,.8],
                    [.6,.2,.7,.3],[.4,.7,.2,.9],[.8,.1,.6,.4],[.5,.4,.3,.7]],dtype=torch.float64,requires_grad=True)
    for causal in [False,True]:
        expected=dense(q,k,v,causal)
        for size in [1,3,4,8]:
            for reverse in [False,True]:
                actual=online(q,k,v,size,causal,reverse=reverse)
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        ga=torch.autograd.grad(online(q,k,v,3,causal).square().sum(),(q,k,v),retain_graph=True)
        gb=torch.autograd.grad(expected.square().sum(),(q,k,v),retain_graph=True)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(online(q[3:],k,v,3,True,offset=3),dense(q,k,v,True)[3:])
    torch.testing.assert_close(online(q*1000,k,v,3),dense(q*1000,k,v))
    scores=(q[0]@k.T/2).detach();w=(scores-scores.max()).exp()
    first=(scores[:4]-scores[:4].max()).exp()
    return {'scores_row0':scores.tolist(),'first_m':float(scores[:4].max()),
            'first_l':float(first.sum()),'first_u':(first@v[:4].detach()).tolist(),
            'correction':float((scores[:4].max()-scores.max()).exp()),
            'final_l':float(w.sum()),'output_row0':dense(q,k,v).detach()[0].tolist(),
            'outputs_gradients_masks_and_orders_equal':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
