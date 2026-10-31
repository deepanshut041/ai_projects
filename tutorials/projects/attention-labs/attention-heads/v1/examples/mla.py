"""Tiny MLA algebra: expanded and latent-domain retrieval with decoupled RoPE."""
import json
import math
import torch


def rotate_pair(x, positions):
    c,s=positions.to(x).cos(),positions.to(x).sin()
    a,b=x[...,0],x[...,1]
    return torch.stack([a*c-b*s,a*s+b*c],-1)


def projections(x,wd,wq,wqr,wkr):
    b,t,_=x.shape
    latent=x@wd
    q=(x@wq).reshape(b,t,2,2).transpose(1,2)
    qr=(x@wqr).reshape(b,t,2,2).transpose(1,2)
    kr=x@wkr
    positions=torch.arange(t,device=x.device)
    return q,rotate_pair(qr,positions),latent,rotate_pair(kr,positions)


def attend(q,qr,latent,kr,uk,uv,wo,offset=0,absorbed=True):
    t=latent.shape[1];s=q.shape[-2]
    if absorbed:
        query_latent=torch.einsum('bhsd,hrd->bhsr',q,uk)
        content_scores=torch.einsum('bhsr,btr->bhst',query_latent,latent)
    else:
        keys=torch.einsum('btr,hrd->bhtd',latent,uk)
        content_scores=q@keys.transpose(-2,-1)
    scores=(content_scores+torch.einsum('bhsd,btd->bhst',qr,kr))/math.sqrt(2+2)
    query_positions=offset+torch.arange(s,device=q.device)
    future=torch.arange(t,device=q.device)[None,:]>query_positions[:,None]
    weights=scores.masked_fill(future,-torch.inf).softmax(-1)
    if absorbed:
        mixed_latent=torch.einsum('bhst,btr->bhsr',weights,latent)
        context=torch.einsum('bhsr,hrd->bhsd',mixed_latent,uv)
    else:
        values=torch.einsum('btr,hrd->bhtd',latent,uv)
        context=weights@values
    joined=context.transpose(1,2).contiguous().reshape(q.shape[0],s,4)
    return joined@wo


def run_checks():
    torch.manual_seed(71)
    shapes=[(2,5,4),(4,3),(4,4),(4,4),(4,2),(2,3,2),(2,3,2),(4,4)]
    x,wd,wq,wqr,wkr,uk,uv,wo=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in shapes]
    q,qr,c,kr=projections(x,wd,wq,wqr,wkr)
    a=attend(q,qr,c,kr,uk,uv,wo,absorbed=True)
    b=attend(q,qr,c,kr,uk,uv,wo,absorbed=False)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    variables=(x,wd,wq,wqr,wkr,uk,uv,wo)
    ga=torch.autograd.grad(a.square().sum(),variables,retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),variables)
    for left,right in zip(ga,gb):
        torch.testing.assert_close(left,right,atol=1e-9,rtol=1e-9)
    first=attend(q[:,:,:2],qr[:,:,:2],c[:,:2],kr[:,:2],uk,uv,wo)
    later=attend(q[:,:,2:],qr[:,:,2:],c,kr,uk,uv,wo,offset=2)
    torch.testing.assert_close(torch.cat([first,later],1),a)
    assert sum(z.numel()*z.element_size() for z in [c,kr])==400
    wrong=attend(q[:,:,2:],rotate_pair((x@wqr).reshape(2,5,2,2).transpose(1,2)[:,:,2:],torch.arange(3)),c,kr,uk,uv,wo,offset=2)
    assert not torch.allclose(wrong,a[:,2:])
    return {'expanded_absorbed_outputs_and_gradients_equal':True,
            'cached_chunks_equal':True,'latent_plus_position_bytes':400,
            'wrong_rotary_offset_detected':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
