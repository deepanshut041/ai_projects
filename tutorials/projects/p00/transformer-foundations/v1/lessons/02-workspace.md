## Lesson 2 — PyTorch tensors, storage and module state

**Goal:** distinguish mathematical axes from the runtime's shape, dtype, device and storage contracts. Prerequisite: [linear algebra](lessons/01-prediction.md). The existing learner environment is preserved; this lesson does not ask you to recreate the child or start with Python syntax.

### A tensor needs more than a shape

Write a contract such as `features: float32[B,T,D] on cpu` and `ids: int64[B,T] on cpu`. Shape names the coordinates, dtype the representation of each element, and device where the operation executes. A vocabulary ID is a category index, not a continuous feature. Casting ID 37 to 37.0 does not create an embedding. Integer IDs normally enter `nn.Embedding`; floating features enter `nn.Linear`.

For the runnable reference, use the already configured attention project's environment. From workspace home:

```bash
ai_projects/projects/transformer-labs/attention-is-all-you-need/.venv/bin/python \
  ai_projects/docs/tutorials/projects/p00/transformer-foundations/v1/examples/prerequisite_tensors.py
```

For later shell blocks that use `python`, select this reference interpreter in the terminal first. From workspace home, `source ai_projects/projects/transformer-labs/attention-is-all-you-need/.venv/bin/activate` selects the existing environment; it does not recreate either lab. Then change to the stated examples directory. Alternatively replace `python` with that interpreter’s path. Standalone standard-library blocks can use host `python3`.

The checked reference runtime is Python 3.12 with Torch 2.14.1+cpu. This does not install Torch in the separate learner lab. If you configure another environment, record its versions and use the CPU dependency instructions from [uv's PyTorch guide](source:uv-pytorch). Keep environments and downloaded packages out of Git.

A module and its inputs must use compatible dtypes/devices. `model.to(device)` moves registered parameters and buffers; moving only the input can leave the weights elsewhere. Floating tensors require an explicit decision about FP32 versus FP64. FP64 is useful for finite-difference checks; it is not automatically the best training precision.

### Broadcasting follows lengths, not meaning

Trailing dimensions align. Lengths must be equal or one of them must be 1; missing leading dimensions behave as 1. For `[B,T,D]`, bias `[D]` repeats across B and T. A token-valid mask `[B,T]` needs `[...,None]` to multiply features `[B,T,D]`. Adding `[T]` to `[B,T,D]` acts on the last axis if T happens to equal D, which is the wrong interpretation.

```python
import torch
x = torch.arange(24).reshape(2, 3, 4).double()
bias = torch.tensor([10., 20., 30., 40.], dtype=x.dtype)
valid = torch.tensor([[True, True, False], [True, False, False]])
y = (x + bias) * valid[..., None]
assert y.shape == (2, 3, 4)
assert torch.equal(y[0, 2], torch.zeros(4, dtype=x.dtype))
```

A mask is not a label and does not inherently alter gradients. This multiply zeroes a representation; later attention masking removes specific query–key connections. The purpose determines the operation.

### Transpose and reshape answer different questions

An array's strides tell us how an index changes its storage offset. In contiguous `[2,3,4]` storage, strides are `(12,4,1)`. Swapping the last two axes gives shape `[2,4,3]`, strides `(12,1,4)`, and a noncontiguous view. Values move logically to different coordinates without necessarily moving physical bytes.

```python
import torch
x = torch.arange(24).reshape(2, 3, 4)
t = x.transpose(1, 2)
assert t[0, 1, 2].item() == x[0, 2, 1].item() == 9
assert t.stride() == (12, 1, 4)
assert not t.is_contiguous()
flat = t.reshape(2, 12)  # may allocate to preserve t's logical order
assert flat[0].tolist() == [0,4,8,1,5,9,2,6,10,3,7,11]
```

`reshape` preserves logical traversal order; it does not exchange semantic axes. `view` requires compatible strides. `contiguous()` creates compatible storage when needed. A copied reshape and a view can have different mutation behavior: do not rely on aliasing unless you establish it. In multi-head attention, reshaping `[B,T,D]` to `[B,T,H,d]` and transposing to `[B,H,T,d]` are separate operations.

### Modules own learnable and persistent state

`nn.Parameter` marks learnable state. Assigning a submodule as an attribute registers it recursively. Use `ModuleList` for a list of layers: a plain Python list does not provide that registration. A buffer is persistent non-parameter state, such as a fixed position table. `state_dict()` includes registered parameters and persistent buffers; it is not the whole training checkpoint.

```python
import torch
from torch import nn
class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Linear(2, 3)
        self.register_buffer('offset', torch.zeros(3))
    def forward(self, x):
        return self.map(x) + self.offset
m = Probe()
assert sum(p.numel() for p in m.parameters()) == 9
assert set(m.state_dict()) == {'map.weight', 'map.bias', 'offset'}
```

`train()` and `eval()` control layer behavior such as dropout. `no_grad()` controls graph recording. Neither substitutes for the other, and `eval()` does not freeze parameters.

[[practice:tensor-transfer]]

### Notes and checks

Write the coordinate, dtype/device and storage contract for an ID tensor, its embeddings and a head-split tensor. Predict the transpose result before running it. Explain why a same-shaped tensor can still have the wrong axes. The runnable tensor checks are [test_prerequisite_tensors.py](examples/test_prerequisite_tensors.py); use the configured Torch interpreter.

Sources: [broadcasting](source:broadcasting), [views](source:torch-views), [Module](source:torch-modules). Continue with [calculus](lessons/03-numbers.md).
