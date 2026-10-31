"""Adjacent-pair RoPE: norms, relative positions, explicit blocks and cache offsets."""
import json
import math
import torch


def rotate(x, positions, base=10000.0):
    d = x.shape[-1]
    if d % 2 or len(positions) != x.shape[-2] or base <= 0:
        raise ValueError('Even rotated width, matching positions and positive base required')
    frequencies = base ** (-torch.arange(0,d,2,dtype=x.dtype,device=x.device)/d)
    angles = positions.to(dtype=x.dtype,device=x.device)[:,None] * frequencies[None,:]
    pairs = x.reshape(*x.shape[:-1],d//2,2)
    a,b = pairs[...,0],pairs[...,1]
    c,s = angles.cos(),angles.sin()
    return torch.stack([a*c-b*s,a*s+b*c],dim=-1).flatten(-2)


def run_checks():
    unit = torch.tensor([[[[1.0,0.0]]]],dtype=torch.float64)
    scores = []
    for m,n in [(3,1),(5,3),(100,98)]:
        q = rotate(unit,torch.tensor([m]))
        k = rotate(unit,torch.tensor([n]))
        score = (q*k).sum().item()
        assert math.isclose(score,math.cos(2),abs_tol=1e-12)
        scores.append(round(score,6))
    torch.manual_seed(53)
    x = torch.randn(2,3,6,4,dtype=torch.float64,requires_grad=True)
    pos = torch.arange(6)
    actual = rotate(x,pos)
    torch.testing.assert_close(actual.norm(dim=-1),x.norm(dim=-1))
    matrices = []
    for t in pos.tolist():
        blocks = []
        for f in [1.0,0.01]:
            c,s = math.cos(t*f),math.sin(t*f)
            blocks.append(torch.tensor([[c,-s],[s,c]],dtype=x.dtype))
        matrices.append(torch.block_diag(*blocks))
    reference = torch.einsum('tij,bhtj->bhti',torch.stack(matrices),x)
    torch.testing.assert_close(actual,reference)
    left = torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    right = torch.autograd.grad(reference.square().sum(),x)[0]
    torch.testing.assert_close(left,right)
    chunks = torch.cat([rotate(x[:,:,:2],pos[:2]),rotate(x[:,:,2:],pos[2:])],dim=2)
    torch.testing.assert_close(chunks,actual)
    wrong = rotate(x[:,:,2:],torch.arange(4))
    assert not torch.allclose(wrong,actual[:,:,2:])
    # One pair oscillates; position separation does not guarantee monotonic scores.
    assert math.cos(6) > math.cos(3)
    try:
        rotate(torch.ones(1,1,2,3),torch.arange(2))
    except ValueError:
        pass
    else:
        raise AssertionError('Odd rotated width accepted')
    return {'equal_gap_scores':scores,'norm_and_block_gradients_equal':True,
            'cached_positions_equal':True,'reset_position_bug_detected':True,
            'odd_width_rejected':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(),sort_keys=True))
