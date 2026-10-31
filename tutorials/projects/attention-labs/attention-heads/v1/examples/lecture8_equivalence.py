"""The 2x6x6 shape route and independent ModuleList/packed forward/backward checks."""
import torch
from head_loop import LoopMHA
from multihead_torch import PackedMHA


if __name__ == '__main__':
    torch.manual_seed(123)
    loop = LoopMHA(width=6, heads=2, context_len=6).eval()
    packed = PackedMHA(width=6, heads=2).double().eval()
    with torch.no_grad():
        for name in ['query', 'key', 'value']:
            source = torch.cat([getattr(head, name).weight for head in loop.heads], dim=0)
            getattr(packed, name).weight.copy_(source)
        packed.output.weight.copy_(loop.output.weight)
    x_loop = torch.randn(2, 6, 6, dtype=torch.float64, requires_grad=True)
    x_packed = x_loop.detach().clone().requires_grad_()
    y_loop = loop(x_loop)
    y_packed, weights = packed(x_packed)
    torch.testing.assert_close(y_loop, y_packed, atol=1e-12, rtol=1e-12)
    y_loop.square().sum().backward()
    y_packed.square().sum().backward()
    torch.testing.assert_close(x_loop.grad, x_packed.grad, atol=1e-12, rtol=1e-12)
    for name in ['query', 'key', 'value']:
        expected = torch.cat([getattr(head, name).weight.grad for head in loop.heads], dim=0)
        torch.testing.assert_close(expected, getattr(packed, name).weight.grad,
                                   atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(loop.output.weight.grad, packed.output.weight.grad,
                               atol=1e-12, rtol=1e-12)
    q_full = packed.query(x_packed.detach())
    q_view = q_full.view(2, 6, 2, 3)
    wrong = q_view @ q_view.transpose(-2, -1)
    q_heads = q_view.transpose(1, 2)
    right = q_heads @ q_heads.transpose(-2, -1)
    for label, tensor in [('projected', q_full), ('split', q_view), ('wrong scores', wrong),
                          ('transposed', q_heads), ('right scores', right), ('weights', weights)]:
        print(label, list(tensor.shape))
    changed = x_packed.detach().clone()
    changed[:, 4:] += 100
    torch.testing.assert_close(packed(changed)[0][:, :4], y_packed.detach()[:, :4],
                               atol=1e-12, rtol=1e-12)
    print('output', list(y_packed.shape), 'maximum difference', (y_loop-y_packed).abs().max().item())
    print('Input, Q/K/V and output-projection gradients agree; future intervention passes.')
