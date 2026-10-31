"""Author check: run with the project Python environment; use frozen tutorial source."""
import json
import math
import hashlib
import sys
from pathlib import Path
import torch

COURSE = Path(__file__).resolve().parents[1]
bundle = json.loads((COURSE/'code/source-bundle.json').read_text())
entry = bundle['files']['src/core/attention.py']
assert hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256']
captured = {}
exec(compile(entry['text'], 'captured-core-attention.py', 'exec'), captured)
Attention = captured['Attention']
scaled_dot_product_attention = captured['scaled_dot_product_attention']


def verify():
    x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [0., 0.]]], dtype=torch.float64)
    layer = Attention(2, 1, fused_attn=False).double().eval()
    with torch.no_grad():
        for projection, multiplier in [(layer.q_proj, 1), (layer.k_proj, 1),
                                       (layer.v_proj, 2), (layer.out_proj, 1)]:
            projection.weight.copy_(multiplier * torch.eye(2, dtype=x.dtype))
            projection.bias.zero_()
    output, weights = layer(x, need_weights=True)
    torch.testing.assert_close(output[0, 0], torch.tensor([1.3395230986533138, 1.0], dtype=x.dtype))
    fast = Attention(2, 1, fused_attn=True).double().eval()
    fast.load_state_dict(layer.state_dict())
    torch.testing.assert_close(fast(x), output, atol=1e-12, rtol=1e-12)
    mask = torch.tensor([[True, True, True, False]])
    changed = x.clone(); changed[:, 3] += 1000
    torch.testing.assert_close(layer(x, attention_mask=mask)[:, :3], layer(changed, attention_mask=mask)[:, :3])
    torch.testing.assert_close(layer(x, causal=True)[:, :3], layer(changed, causal=True)[:, :3])
    assert layer(x, tgt=x[:, :2]).shape == (1, 2, 2)
    with torch.no_grad(): layer.out_proj.bias.fill_(0.25)
    blocked = torch.zeros((1, 4), dtype=torch.bool)
    torch.testing.assert_close(layer(x, attention_mask=blocked), torch.full_like(x, 0.25))

    q = x[0, :1].clone().requires_grad_()
    k, v = x[0], 2*x[0]
    target = torch.tensor([[2., 0.]], dtype=x.dtype)
    context, _ = scaled_dot_product_attention(q, k, v)
    loss = 0.5*((context-target)**2).sum()
    loss.backward()
    expected = torch.tensor([[-0.20659532084217017,0.35355339059327373]], dtype=x.dtype)
    torch.testing.assert_close(q.grad, expected)
    epsilon=1e-6
    for axis in range(2):
        losses=[]
        for sign in [1,-1]:
            probe=q.detach().clone(); probe[0,axis]+=sign*epsilon
            c,_=scaled_dot_product_attention(probe,k,v)
            losses.append((0.5*((c-target)**2).sum()).item())
        assert math.isclose((losses[0]-losses[1])/(2*epsilon),q.grad[0,axis].item(),abs_tol=1e-8)
    return {'passed': True, 'same_fixture_forward': True, 'same_fixture_gradient': True,
            'explicit_sdpa_parity': True, 'padding_causal_influence': True,
            'cross_shape': True, 'all_blocked_bias': True}

if __name__ == '__main__':
    print(json.dumps(verify(),sort_keys=True))
