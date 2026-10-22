## Lesson 10 — Training updates, schedules and gradient scale

**Goal:** control exactly one optimizer update and reason about batch size, accumulation, schedule and clipping. Prerequisites: loss gradients and optimizer state. A forward pass alone is not training.

### One update has a defined order

A training step selects records, clears old gradients, computes logits and loss, calls backward, optionally clips gradients, and applies an optimizer update. The step counter and schedule advance once per optimizer update under the convention used here. A different schedule convention is possible, but checkpoint/recovery must preserve it.

```python
import torch
from torch import nn
from torch.nn import functional as F
model = nn.Sequential(nn.Linear(2,8), nn.Tanh(), nn.Linear(8,2))
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
x = torch.tensor([[1.,1.],[1.,-1.]])
y = torch.tensor([0,1])
model.train()
optimizer.zero_grad(set_to_none=True)
loss = F.cross_entropy(model(x), y)
assert torch.isfinite(loss)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
optimizer.step()
```

Clipping happens after backward and before the update. Logging `.item()` is fine after the scalar has been used in backward; building the loss from `.item()` values would break the graph. A failed nonfinite check should stop before an optimizer step rather than save corrupted state as successful progress.

### Accumulation must preserve the intended objective

Suppose micro-batch A has 2 valid targets and sum loss S₁, while B has 6 and sum loss S₂. The combined token objective is `(S₁+S₂)/8`. Backward each sum divided by 8, accumulate gradients, then step once. Averaging the two means gives `(S₁/2+S₂/6)/2`, which overweights A. Divide by the total supervised count, not automatically by the number of micro-batches.

For equal-sized batches, dividing each mean by the number of accumulation steps is equivalent. Dropout and other random behavior can still make separate forwards differ from a combined forward even when the algebra matches. The reference accumulation check disables such differences and compares unequal batch sizes. Accumulation does not replicate BatchNorm's combined batch statistics.

### Learning-rate schedules change update scale

A learning rate sets a step size in parameter space after optimizer transformation. Too large can overshoot; too small can make useful progress slow. Warmup gradually raises the rate while initial optimizer/activation behavior settles. It is a chosen training protocol, not a universal proof of stability.

The paper uses positive update index s and warmup w:

$$\eta(s)=D^{-1/2}\min(s^{-1/2},s\,w^{-3/2}).$$

For s≤w, the second branch grows linearly; afterward the first decays as inverse sqrt(s). They meet at s=w with peak `1/sqrt(Dw)`. With D=4,w=4, rates at s=1,4,16 are `.0625,.25,.125`. The paper's D=512,w=4000 are source settings, not a requirement to use that long warmup on our tiny classifier. The classifier reference uses its own explicit short linear warmup then constant learning rate; it is not a reproduction of the paper's schedule.

Avoid s=0 division, off-by-one first steps and calling a per-update scheduler once per micro-batch. Save scheduler state and the exact update index. Changing total planned steps halfway through a resumed cosine schedule changes the future trajectory.

### Global clipping is one shared rescaling

Let g collect all parameter-gradient coordinates and c be a norm limit. Global L2 clipping computes r=||g||₂, then scales every coordinate by `min(1,c/(r+ε))`. With g=`[3,4]` and c=2.5, the result is approximately `[1.5,2]`. Its direction is preserved; coordinatewise clipping to ±2.5 would produce `[2.5,2.5]`, a different direction. The returned norm from PyTorch's clip operation is the preclip norm.

A logged gradient norm detects scale changes, not whether every gradient is correct. A disconnected layer can have `None` gradients while another layer has a healthy global norm. Inspect both.

### Precision and measurement

FP16 has limited exponent range; automatic mixed precision may use loss scaling, requiring unscaling before clipping. BF16 has a different exponent/precision tradeoff. Keep FP32 CPU as the course's controlled training reference and FP64 for small gradient checks. A future mixed-precision run must record its scaler/precision state and measured behavior; GPU use alone does not prove faster or correct execution.

[[practice:training-step-transfer]]

In my notes: draw the update boundary, derive the unequal-micro-batch denominator, calculate all three schedule values and compare global with coordinate clipping. The actual loop is [prerequisite_mlp.py](examples/prerequisite_mlp.py), with checks in [prerequisite_mlp_checks.py](examples/prerequisite_mlp_checks.py).

Sources: [original optimizer/schedule](source:transformer-paper), [gradient clipping](source:gradient-clipping). Continue with [data and supervision](lessons/06-data.md).
