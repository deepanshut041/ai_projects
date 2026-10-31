"""Complete illustrative MQA layer; same parameters, broadcast and explicit loop."""
import json
import math
import torch
from torch import nn


class MQA(nn.Module):
    def __init__(self, width=4, heads=4, head_width=1):
        super().__init__()
        self.heads, self.head_width = heads, head_width
        self.wq = nn.Linear(width, heads*head_width, bias=False, dtype=torch.float64)
        self.wk = nn.Linear(width, head_width, bias=False, dtype=torch.float64)
        self.wv = nn.Linear(width, head_width, bias=False, dtype=torch.float64)
        self.wo = nn.Linear(heads*head_width, width, bias=False, dtype=torch.float64)

    def forward(self, x, loop=False):
        b, t, _ = x.shape
        q = self.wq(x).reshape(b,t,self.heads,self.head_width).transpose(1,2)
        k, v = self.wk(x).unsqueeze(1), self.wv(x).unsqueeze(1)
        future = torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
        if loop:
            weights = torch.stack([
                ((q[:,h] @ k[:,0].transpose(-2,-1))/math.sqrt(self.head_width))
                .masked_fill(future,-torch.inf).softmax(-1)
                for h in range(self.heads)], dim=1)
            context = torch.stack([weights[:,h] @ v[:,0] for h in range(self.heads)], dim=1)
        else:
            scores = q @ k.transpose(-2,-1) / math.sqrt(self.head_width)
            weights = scores.masked_fill(future,-torch.inf).softmax(-1)
            context = weights @ v
        joined = context.transpose(1,2).contiguous().reshape(b,t,-1)
        return self.wo(joined), weights, k, v


def run_checks():
    torch.manual_seed(31)
    layer = MQA()
    x = torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
    y, weights, k, v = layer(x)
    loop, _, _, _ = layer(x,loop=True)
    torch.testing.assert_close(y,loop,atol=1e-12,rtol=1e-12)
    parameters = (x,*layer.parameters())
    a = torch.autograd.grad(y.square().sum(),parameters,retain_graph=True)
    b = torch.autograd.grad(loop.square().sum(),parameters)
    for left,right in zip(a,b):
        torch.testing.assert_close(left,right,atol=1e-12,rtol=1e-12)
    changed = x.detach().clone();changed[:,2] += 100
    torch.testing.assert_close(layer(changed)[0][:,:2],y.detach()[:,:2])
    assert not torch.allclose(weights[:,0,-1],weights[:,1,-1])
    assert k.shape == v.shape == (2,1,3,1)
    assert sum(p.numel() for p in layer.parameters()) == 40
    return {'loop_and_gradients_equal':True,'causal_intervention':True,
            'shared_kv_shape':[2,1,3,1],'distinct_query_patterns':True,'parameters':40}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
