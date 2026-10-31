"""Normalized positive-kernel attention: parallel and recurrent references."""
import json
import torch
import torch.nn.functional as F


def phi(x):return F.elu(x)+1


def parallel(q,k,v):
    weights=phi(q)@phi(k).T
    weights=weights.tril()
    return (weights@v)/weights.sum(-1,keepdim=True)


def recurrent(q,k,v,state=None):
    if state is None:
        s=q.new_zeros(q.shape[-1],v.shape[-1]);z=q.new_zeros(q.shape[-1])
    else:s,z=state
    outputs=[]
    for query,key,value in zip(phi(q),phi(k),v):
        s=s+torch.outer(key,value);z=z+key
        denominator=query@z
        if not denominator>0:raise ValueError('Nonpositive denominator')
        outputs.append((query@s)/denominator)
    return torch.stack(outputs),(s,z)


def run_checks():
    q=torch.tensor([[0.,0.],[0.,2.]],dtype=torch.float64)
    k=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
    v=torch.tensor([[2.],[4.]],dtype=torch.float64)
    hand,state=recurrent(q,k,v)
    torch.testing.assert_close(hand,torch.tensor([[2.],[17/6]],dtype=torch.float64))
    torch.manual_seed(97)
    q,k,v=[torch.randn(7,3,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    a,state=recurrent(q,k,v);b=parallel(q,k,v)
    torch.testing.assert_close(a,b)
    ga=torch.autograd.grad(a.square().sum(),(q,k,v),retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),(q,k,v))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s=recurrent(q[:3],k[:3],v[:3]);later,_=recurrent(q[3:],k[3:],v[3:],s)
    torch.testing.assert_close(torch.cat([first,later]),a)
    modified=v.detach().clone();modified[-1]+=100
    torch.testing.assert_close(recurrent(q,k,modified)[0][:-1],a[:-1])
    fresh,_=recurrent(q[3:],k[3:],v[3:])
    assert not torch.allclose(fresh,later)
    # Even a one-dimensional summary can collapse distinct histories.
    keys=torch.ones(2,1,dtype=torch.float64)
    left=keys.T@torch.tensor([[1.],[3.]],dtype=torch.float64)
    right=keys.T@torch.tensor([[2.],[2.]],dtype=torch.float64)
    torch.testing.assert_close(left,right)
    return {'worked_outputs':hand.flatten().tolist(),'state_elements_per_head':12,
            'parallel_recurrent_outputs_gradients_chunks_equal':True,
            'future_and_reset_checks':True,'distinct_histories_can_collide':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
