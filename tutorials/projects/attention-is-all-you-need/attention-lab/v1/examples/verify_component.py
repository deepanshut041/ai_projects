"""Run with the project environment: uv run --frozen --no-sync python /absolute/path/to/this/file.py.

Uses the frozen source beside this tutorial, not the working model checkout.
All tensors are tiny CPU component probes; no training/data acquisition occurs.
"""
import json
import sys
from pathlib import Path

import torch

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COURSE/'code/src'))
from core.attention import Attention, scaled_dot_product_attention
from arithmetic import checks, matmul, softmax, scalar_loss


def verify():
    results = {"arithmetic":checks(),"torch":torch.__version__,"device":"cpu"}
    torch.manual_seed(7)
    src = torch.randn(2,5,12,dtype=torch.float64)
    tgt = torch.randn(2,3,12,dtype=torch.float64)
    mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,0]],dtype=torch.bool)
    layer = Attention(12,3).double().eval()
    with torch.inference_mode():
        assert layer(src,attention_mask=mask).shape == (2,5,12)
        assert layer(src,tgt=tgt,attention_mask=mask).shape == (2,3,12)
        explicit,weights = layer(src,attention_mask=mask,need_weights=True)
        torch.testing.assert_close(explicit,layer(src,attention_mask=mask),atol=1e-10,rtol=1e-9)
        assert weights.shape == (2,3,5,5)
        changed = src.clone(); changed[~mask] += 100
        torch.testing.assert_close(layer(src,attention_mask=mask)[mask],layer(changed,attention_mask=mask)[mask])
        changed = src.clone(); changed[:,3:] += 100
        torch.testing.assert_close(layer(src,causal=True)[:,:3],layer(changed,causal=True)[:,:3])
        empty = torch.zeros(2,5,dtype=torch.bool)
        torch.testing.assert_close(layer(src,attention_mask=empty),layer.out_proj.bias.expand_as(src))
    for kwargs in ({"tgt":tgt,"causal":True},{"attention_mask":mask[:,:3]}):
        try: layer(src,**kwargs)
        except ValueError: pass
        else: raise AssertionError("Expected invalid interface to fail")
    q=torch.tensor([[[0.]]],dtype=torch.float64,requires_grad=True)
    k=torch.tensor([[[1.],[-1.]]],dtype=torch.float64)
    v=torch.tensor([[[2.],[6.]]],dtype=torch.float64,requires_grad=True)
    context,_ = scaled_dot_product_attention(q,k,v)
    loss=.5*(context-5).square().sum(); loss.backward()
    assert context.item()==4 and loss.item()==.5
    torch.testing.assert_close(q.grad,torch.full_like(q,2))
    torch.testing.assert_close(v.grad,torch.full_like(v,-.5))
    finite_difference=(scalar_loss(1e-6)-scalar_loss(-1e-6))/(2e-6)
    assert abs(q.grad.item()-finite_difference)<1e-8
    assert sum(p.numel() for p in Attention(9,3).parameters())==360
    # Compare independent scalar-list matrix arithmetic with the real projections.
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
    q=x@torch.tensor([[1.,0.],[1.,1.]],dtype=torch.float64)
    k=x@torch.tensor([[1.,1.],[0.,-1.]],dtype=torch.float64)
    v=x@torch.diag(torch.tensor([2.,3.],dtype=torch.float64))+torch.tensor([.5,-.5])
    actual,_=scaled_dot_product_attention(q,k,v)
    expected=torch.tensor([2.1044483707191435,1.296663721960642],dtype=torch.float64)
    torch.testing.assert_close(actual[0],expected)
    results.update(passed=True,query_gradient=2.,component_shapes_checked=True,invalid_interfaces_checked=True,
                   mask_perturbations_checked=True,all_blocked_bias_checked=True,parameter_count=360)
    return results


if __name__=='__main__': print(json.dumps(verify(),sort_keys=True))
