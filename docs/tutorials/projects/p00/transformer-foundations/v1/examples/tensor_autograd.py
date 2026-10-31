"""Bounded M1 exercises; fits XOR, not a generalization benchmark."""
import json
import torch
from torch import nn
from torch.nn import functional as F


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    w = torch.tensor([[1., 0., 2.], [0., 1., 3.]], dtype=torch.float64)
    bias = torch.tensor([0., 0., 1.], dtype=torch.float64)
    assert torch.equal(x @ w + bias, torch.tensor([[1., 2., 9.], [3., 4., 19.]], dtype=torch.float64))
    scalar = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    loss = (scalar * 3 - 1) ** 2
    loss.backward()
    assert scalar.grad.item() == 30
    assert torch.autograd.gradcheck(lambda v: (v * 3 - 1) ** 2, (scalar,))
    sentinels = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
    heads = sentinels.reshape(2, 3, 2, 4).transpose(1, 2)
    joined = heads.transpose(1, 2).contiguous().reshape(2, 3, 8)
    assert torch.equal(joined, sentinels)
    # A wrong reshape can round-trip too; verify intermediate ownership.
    assert heads[0, 0].tolist() == [[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]]
    assert heads[1, 1, 2].tolist() == [44, 45, 46, 47]
    wrong = sentinels.reshape(2, 2, 3, 4)
    assert not torch.equal(wrong, heads)
    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([0, 1, 1, 0])
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = float(F.cross_entropy(model(inputs), targets).detach())
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    final = float(F.cross_entropy(model(inputs), targets).detach())
    predictions = model(inputs).argmax(-1).tolist()
    assert predictions == targets.tolist() and final < initial * 0.05
    print(json.dumps({"matmul": True, "head_roundtrip": True, "head_ownership": True, "analytic_gradient": 30,
                      "finite_difference": True, "xor_initial_loss": initial,
                      "xor_final_loss": final, "xor_predictions": predictions,
                      "scope": "four-row fit; no held-out claim"}, indent=2))


if __name__ == "__main__":
    main()
