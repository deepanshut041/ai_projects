"""Delta-rule fast memory with equivalent residual and matrix updates."""
import json
import torch
import torch.nn.functional as F


def retrieve(q,k,v,beta,state=None,matrix=False):
    s=q.new_zeros(k.shape[-1],v.shape[-1]) if state is None else state
    out=[]
    for query,key,value,b in zip(q,k,v,beta):
        if matrix:
            transition=torch.eye(len(key),dtype=key.dtype)-b*torch.outer(key,key)
            s=transition@s+b*torch.outer(key,value)
        else:
            prediction=key@s
            s=s+b*torch.outer(key,value-prediction)
        out.append(query@s)
    return torch.stack(out),s


def run_checks():
    q=k=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    v=torch.tensor([[2.],[5.]],dtype=torch.float64)
    overwrite,_=retrieve(q,k,v,torch.ones(2))
    partial,_=retrieve(q,k,v,torch.tensor([1.,.25]))
    torch.testing.assert_close(overwrite.flatten(),torch.tensor([2.,5.],dtype=torch.float64))
    torch.testing.assert_close(partial.flatten(),torch.tensor([2.,2.75],dtype=torch.float64))
    torch.manual_seed(107)
    qraw,kraw=[torch.randn(7,3,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    values=torch.randn(7,2,dtype=torch.float64,requires_grad=True)
    gate=torch.randn(7,dtype=torch.float64,requires_grad=True)
    q,k=F.normalize(qraw,dim=-1),F.normalize(kraw,dim=-1);beta=gate.sigmoid()
    a,sa=retrieve(q,k,values,beta);b,sb=retrieve(q,k,values,beta,matrix=True)
    torch.testing.assert_close(a,b);torch.testing.assert_close(sa,sb)
    inputs=(qraw,kraw,values,gate)
    ga=torch.autograd.grad(a.square().sum(),inputs,retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),inputs,retain_graph=True)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    first,s=retrieve(q[:3],k[:3],values[:3],beta[:3]);later,end=retrieve(q[3:],k[3:],values[3:],beta[3:],s)
    torch.testing.assert_close(torch.cat([first,later]),a);torch.testing.assert_close(end,sa)
    memory=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    loss=.5*(k[0]@memory-values[0]).square().sum()
    gradient=torch.autograd.grad(loss,memory)[0]
    expected=memory-beta[0]*gradient
    actual=memory+beta[0]*torch.outer(k[0],values[0]-k[0]@memory)
    torch.testing.assert_close(actual,expected)
    untouched=memory.detach().clone();_,result=retrieve(q[:1],k[:1],values[:1],torch.zeros(1),untouched)
    torch.testing.assert_close(result,untouched)
    return {'overwrite':overwrite.flatten().tolist(),'partial_write':partial.flatten().tolist(),
            'residual_matrix_outputs_gradients_chunks_equal':True,'online_sgd_identity':True,'zero_gate_preserves_state':True}


if __name__=='__main__':print(json.dumps(run_checks(),sort_keys=True))
