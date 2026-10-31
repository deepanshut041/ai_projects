"""Scalar forgetting followed by a delta correction, in row-vector convention."""
import json
import torch
import torch.nn.functional as F


def retrieve(q,k,v,alpha,beta,state=None,matrix=False):
    s=q.new_zeros(k.shape[-1],v.shape[-1]) if state is None else state
    outputs=[]
    for query,key,value,a,b in zip(q,k,v,alpha,beta):
        if matrix:
            transition=a*(torch.eye(len(key),dtype=key.dtype)-b*torch.outer(key,key))
            s=transition@s+b*torch.outer(key,value)
        else:
            decayed=a*s
            prediction=key@decayed
            s=decayed+b*torch.outer(key,value-prediction)
        outputs.append(query@s)
    return torch.stack(outputs),s


def run_checks():
    initial=torch.diag(torch.tensor([2.,4.],dtype=torch.float64))
    key=torch.tensor([[1.,0.]],dtype=torch.float64);value=torch.tensor([[5.,0.]],dtype=torch.float64)
    _,hand=retrieve(key,key,value,torch.tensor([.5]),torch.tensor([.25]),initial)
    torch.testing.assert_close(hand,torch.diag(torch.tensor([2.,2.],dtype=torch.float64)))
    _,ungated=retrieve(key,key,value,torch.ones(1),torch.tensor([.25]),initial)
    torch.testing.assert_close(ungated,torch.diag(torch.tensor([2.75,4.],dtype=torch.float64)))
    wrong=.5*initial+.25*torch.outer(key[0],value[0]-key[0]@initial)
    assert not torch.allclose(hand,wrong)
    torch.manual_seed(113)
    qr,kr=[torch.randn(6,3,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    v=torch.randn(6,2,dtype=torch.float64,requires_grad=True)
    ar,br=[torch.randn(6,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    q,k=F.normalize(qr,dim=-1),F.normalize(kr,dim=-1);alpha,beta=ar.sigmoid(),br.sigmoid()
    out,s=retrieve(q,k,v,alpha,beta);ref,r=retrieve(q,k,v,alpha,beta,matrix=True)
    torch.testing.assert_close(out,ref);torch.testing.assert_close(s,r)
    args=(qr,kr,v,ar,br)
    ga=torch.autograd.grad(out.square().sum(),args,retain_graph=True)
    gb=torch.autograd.grad(ref.square().sum(),args,retain_graph=True)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s1=retrieve(q[:2],k[:2],v[:2],alpha[:2],beta[:2])
    later,s2=retrieve(q[2:],k[2:],v[2:],alpha[2:],beta[2:],s1)
    torch.testing.assert_close(torch.cat([first,later]),out);torch.testing.assert_close(s2,s)
    old=torch.randn(3,2,dtype=torch.float64)
    _,decayed=retrieve(q[:1],k[:1],v[:1],torch.tensor([.5]),torch.zeros(1),old)
    torch.testing.assert_close(decayed,.5*old)
    _,cleared=retrieve(q[:1],k[:1],v[:1],torch.zeros(1),torch.zeros(1),old)
    torch.testing.assert_close(cleared,torch.zeros_like(old))
    _,written=retrieve(q[:1],k[:1],v[:1],torch.zeros(1),torch.ones(1),old)
    torch.testing.assert_close(k[0]@written,v[0])
    return {'worked_state':hand.tolist(),'outputs_gradients_chunks_equal':True,
            'decay_before_prediction_required':True,'zero_write_still_forgets':True,'zero_decay_reset_and_write_checks':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
