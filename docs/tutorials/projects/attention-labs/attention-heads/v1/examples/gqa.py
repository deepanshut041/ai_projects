"""Projected GQA: explicit grouping, expanded reference and cached chunk checks."""
import json
import math
import torch


def attention(q, k, v, offset=0):
    query = offset + torch.arange(q.shape[-2],device=q.device)
    key = torch.arange(k.shape[-2],device=k.device)
    scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
    return scores.masked_fill(key[None,:] > query[:,None],-torch.inf).softmax(-1) @ v


def grouped_attention(q, k, v, offset=0):
    if any(t.ndim != 4 for t in [q,k,v]):
        raise ValueError('Expected four named axes')
    b,h,t,d = q.shape
    g = k.shape[1]
    if g < 1 or h % g or k.shape[:3] != v.shape[:3]:
        raise ValueError('KV heads must divide query heads and K/V leading axes must agree')
    if k.shape[0] != b or k.shape[-1] != d:
        raise ValueError('Batch or comparison width mismatch')
    queries = q.reshape(b,g,h//g,t,d)
    out = attention(queries,k.unsqueeze(2),v.unsqueeze(2),offset)
    return out.reshape(b,h,t,v.shape[-1])


def run_checks():
    torch.manual_seed(43)
    q = torch.randn(2,4,5,2,dtype=torch.float64,requires_grad=True)
    results = {}
    for g in [1,2,4]:
        k = torch.randn(2,g,5,2,dtype=torch.float64,requires_grad=True)
        v = torch.randn(2,g,5,3,dtype=torch.float64,requires_grad=True)
        actual = grouped_attention(q,k,v)
        repeat = 4//g
        expanded = attention(q,k.repeat_interleave(repeat,1),v.repeat_interleave(repeat,1))
        torch.testing.assert_close(actual,expanded,atol=1e-12,rtol=1e-12)
        left = torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
        right = torch.autograd.grad(expanded.square().sum(),(q,k,v))
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        first = grouped_attention(q[:,:,:2],k[:,:,:2],v[:,:,:2])
        later = grouped_attention(q[:,:,2:],k,v,offset=2)
        torch.testing.assert_close(torch.cat([first,later],2),actual)
        results[str(g)] = {'head_map':[h//repeat for h in range(4)],
                          'kv_payload_bytes':sum(x.numel()*x.element_size() for x in [k,v])}
    try:
        grouped_attention(q,torch.zeros(2,3,5,2),torch.zeros(2,3,5,3))
    except ValueError:
        pass
    else:
        raise AssertionError('Non-divisible grouping accepted')
    return {'groups':results,'output_and_gradients_equal':True,'cached_chunks_equal':True,
            'invalid_grouping_rejected':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
