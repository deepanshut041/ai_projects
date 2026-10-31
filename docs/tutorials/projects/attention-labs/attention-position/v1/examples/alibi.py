"""Causal ALiBi with explicit positions and a stated power-of-two slope schedule."""
import json
import math
import torch


def slopes(heads):
    if heads < 1 or heads & (heads-1):
        raise ValueError('This reference supports power-of-two head counts')
    return 2.0 ** (-8*torch.arange(1,heads+1,dtype=torch.float64)/heads)


def attention(q,k,v,query_positions,key_positions,m):
    distance = query_positions[:,None]-key_positions[None,:]
    bias = -m.to(q)[:,None,None]*distance.to(q)[None,:,:]
    score = q @ k.transpose(-2,-1)/math.sqrt(q.shape[-1]) + bias
    score = score.masked_fill(distance[None,None,:,:]<0,-torch.inf)
    weights = score.softmax(-1)
    return weights @ v,weights


def run_checks():
    torch.manual_seed(61)
    q = torch.randn(2,4,5,2,dtype=torch.float64)
    k = torch.randn_like(q);v = torch.randn(2,4,5,3,dtype=torch.float64)
    pos=torch.arange(5);m=slopes(4)
    y,w=attention(q,k,v,pos,pos,m)
    a,_=attention(q[:,:,:2],k[:,:,:2],v[:,:,:2],pos[:2],pos[:2],m)
    b,_=attention(q[:,:,2:],k,v,pos[2:],pos,m)
    torch.testing.assert_close(torch.cat([a,b],2),y)
    shifted,_=attention(q,k,v,pos+100,pos+100,m)
    torch.testing.assert_close(shifted,y)
    altered=v.clone();altered[:,:,3:]+=100
    torch.testing.assert_close(attention(q,k,altered,pos,pos,m)[0][:,:,:3],y[:,:,:3])
    z=torch.zeros(1,1,4,1,dtype=torch.float64)
    values=torch.arange(1,5,dtype=torch.float64).reshape(1,1,4,1)
    out,weights=attention(z,z,values,torch.arange(4),torch.arange(4),torch.tensor([.5]))
    row=weights[0,0,-1]
    assert torch.all(row[1:]>row[:-1])
    assert math.isclose((row[-1]/row[0]).item(),math.exp(1.5),rel_tol=1e-12)
    return {'slopes':m.tolist(),'last_weights':[round(x,6) for x in row.tolist()],
            'last_output':round(out[0,0,-1,0].item(),6),
            'chunks_and_shared_position_shift_equal':True,'future_intervention':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
