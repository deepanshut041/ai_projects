"""Positive orthogonal random features; small float64 reference, not a fused kernel."""
import json
import math
import torch


def orthogonal_features(count,width,seed):
    generator=torch.Generator().manual_seed(seed);blocks=[]
    for _ in range(math.ceil(count/width)):
        a=torch.randn(width,width,generator=generator,dtype=torch.float64)
        q,r=torch.linalg.qr(a)
        signs=torch.where(r.diag()<0,-1.,1.)
        directions=(q*signs).T
        radii=torch.randn(width,width,generator=generator,dtype=torch.float64).norm(dim=1)
        blocks.append(directions*radii[:,None])
    return torch.cat(blocks)[:count]


def features(x,omega):
    x=x/(x.shape[-1]**.25)
    exponent=x@omega.T-x.square().sum(-1,keepdim=True)/2
    return exponent.exp()/math.sqrt(len(omega))


def pairwise(qf,kf,v):
    a=(qf@kf.T).tril()
    return (a@v)/a.sum(-1,keepdim=True)


def recurrent(qf,kf,v,state=None):
    s,z=(qf.new_zeros(qf.shape[-1],v.shape[-1]),qf.new_zeros(qf.shape[-1])) if state is None else state
    out=[]
    for q,k,value in zip(qf,kf,v):
        s=s+torch.outer(k,value);z=z+k
        out.append((q@s)/(q@z))
    return torch.stack(out),(s,z)


def softmax_reference(q,k,v):
    scores=q@k.T/math.sqrt(q.shape[-1])
    return scores.masked_fill(torch.ones_like(scores,dtype=torch.bool).triu(1),-torch.inf).softmax(-1)@v


def run_checks():
    torch.manual_seed(101)
    q=(torch.randn(6,4,dtype=torch.float64)*.3).requires_grad_()
    k=(torch.randn(6,4,dtype=torch.float64)*.3).requires_grad_()
    v=torch.randn(6,2,dtype=torch.float64,requires_grad=True)
    omega=orthogonal_features(64,4,7)
    units=omega[:4]/omega[:4].norm(dim=1,keepdim=True)
    torch.testing.assert_close(units@units.T,torch.eye(4,dtype=torch.float64))
    qf,kf=features(q,omega),features(k,omega)
    assert (qf>0).all() and (kf>0).all()
    a,state=recurrent(qf,kf,v);b=pairwise(qf,kf,v)
    torch.testing.assert_close(a,b)
    ga=torch.autograd.grad(a.square().sum(),(q,k,v),retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),(q,k,v),retain_graph=True)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s=recurrent(qf[:2],kf[:2],v[:2]);later,_=recurrent(qf[2:],kf[2:],v[2:],s)
    torch.testing.assert_close(torch.cat([first,later]),a)
    target=softmax_reference(q,k,v)
    errors={}
    for count in [16,64,256]:
        errors[count]=[]
        for seed in range(4):
            w=orthogonal_features(count,4,seed)
            result,_=recurrent(features(q,w),features(k,w),v)
            errors[count].append(float((result-target).abs().mean().detach()))
    return {'output_mae_by_features_and_seeds':errors,'same_feature_outputs_gradients_chunks_equal':True,
            'orthogonal_within_blocks':True,'state_elements_at_64_features':192}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
