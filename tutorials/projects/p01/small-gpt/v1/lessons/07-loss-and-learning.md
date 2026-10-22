## Lesson 7 — make the model learn from its mistakes

You have a full forward path and correctly shifted labels. Now measure how much probability the model gives the right labels, differentiate that measurement and change parameters. Start with one tiny calculation before a training loop.

### Cross-entropy starts with a stable log probability

For logits z and correct index y,

$$\ell(z,y)=-\log p_y=\log\left(\sum_j e^{z_j}\right)-z_y.$$

Subtract the maximum m inside the exponential to avoid overflow:

$$\ell=m+\log\left(\sum_j e^{z_j-m}\right)-z_y.$$

Adding a constant to all logits changes neither the probabilities nor the loss. For logits `[0,log(3)]`, softmax is `[0.25,0.75]`. If class 1 is correct, loss is −log(0.75)≈0.287682. The derivative with respect to each logit is $p_j-\mathbf{1}[j=y]$, giving `[0.25,−0.25]`. Gradient descent lowers the wrong score and raises the right one locally. This derivative is for one example's summed loss; averaging over N valid labels adds division by N.

Create `gpt_objective.py` with the imports, summed loss and schedule/optimizer helpers below. The library cross-entropy already combines stable log-softmax and target selection. Do not feed softmax probabilities into an API expecting logits.

<!-- file:gpt_objective.py mode:write -->
```python
"""Loss sums and denominators stay visible for evaluation and accumulation."""
import math
import torch
from torch.nn import functional as F
from gpt_tokenizer import PAD


def nll_sum(logits, labels):
    count = int((labels != PAD).sum())
    if count == 0:
        raise ValueError("No supervised tokens")
    total = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1), ignore_index=PAD, reduction="sum")
    return total, count


def lr_at(update, total, warmup, peak, floor):
    if not 1 <= update <= total or not 1 <= warmup < total or not 0 <= floor <= peak:
        raise ValueError("Invalid fixed learning-rate schedule")
    if update <= warmup:
        return peak * update / warmup
    fraction = (update-warmup) / (total-warmup)
    return floor + 0.5*(peak-floor)*(1+math.cos(math.pi*fraction))


def optimizer_for(model, lr, decay):
    matrices, vectors = [], []
    for parameter in model.parameters():
        (matrices if parameter.ndim >= 2 else vectors).append(parameter)
    return torch.optim.AdamW([{"params": matrices, "weight_decay": decay},
                              {"params": vectors, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8, foreach=False)
```

`reshape(-1,V)` combines batch and time into one label axis without assuming the input is contiguous. `ignore_index=PAD` removes padded labels from the sum; `count` explicitly removes them from the denominator. If no label is supervised, fail rather than returning a meaningless zero or NaN.

Now run `python -m unittest test_gpt_data test_gpt_model -v`. You should see twelve tests pass. For the logit-gradient fixture, explain why the target entry is negative before looking at the assertion.

[[practice:loss-gradient]]

### Accumulate the right mean

Suppose microbatch A has two labels, each with loss 1, and B has one label with loss 4. The combined mean is `(2×1+1×4)/3=2`. Averaging the two batch means gives `(1+4)/2=2.5`, which overweights the smaller batch.

For an effective update made from several microbatches, sum their valid-token counts first. Backpropagate each **summed** microbatch loss divided by that same total. Autograd adds contributions to parameter gradients. Only after all microbatches do you clip and apply one optimizer step. Dividing each batch mean by the number of batches is equivalent only when valid-token counts match.

[[scenario:denominator]]
[[practice:accumulation]]

### Understand the optimizer state you will later save

Adam tracks a moving average m of gradients and v of squared gradients:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

Starting both at zero biases early estimates toward zero. Correct with $\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$. The adaptive update is proportional to $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. For a first scalar gradient 2 and β1=0.9, β2=0.95, the raw moments are 0.2 and 0.2; corrected moments are 2 and 4, so the normalized update is approximately 1. This is different from plain SGD's update proportional to 2.

AdamW adds decoupled decay: conceptually shrink a weight by `(1−lr×decay)` and apply the adaptive gradient step. Our explicit policy decays matrix parameters, including tied/position embeddings, but excludes bias/norm vectors. Weight tying means the shared matrix must appear in the optimizer once; `model.parameters()` yields that registered parameter once here.

Clip the **global** gradient norm after accumulation. If a two-entry gradient is `[3,4]`, its Euclidean norm is 5. Clipping at 1 approximately rescales it to `[0.6,0.8]`. This preserves direction and bounds magnitude. It is not LayerNorm, which normalizes forward features, and not elementwise clipping. Record the norm returned **before** clipping. A nonfinite gradient should stop the run rather than be saved as valid progress.

### Use a fixed global learning-rate schedule

Warmup slowly raises the learning rate during the first W updates. With peak 0.001 and W=20, update 1 uses 0.00005 and update 20 uses 0.001. After warmup, cosine decay moves smoothly to the floor at the declared total N:

$$lr(t)=lr_{min}+\frac{lr_{max}-lr_{min}}{2}
\left[1+\cos\left(\pi\frac{t-W}{N-W}\right)\right].$$

Cosine starts at 1 and ends at −1 over that interval; the bracket therefore goes from 2 to 0. The schedule's input is the global completed-update count plus one. Resuming a 300-update experiment at update 150 does not start warmup again. A session limit is merely where you pause that fixed experiment.

[[practice:schedule]]

### Fit two strings before the book

Create `fit_tiny.py` with this complete diagnostic. It repeatedly uses two fixed strings, dropout zero and constant LR. These are deliberate differences from the later corpus trainer, so interpret its result as a fitting check only.

<!-- file:fit_tiny.py mode:write -->
```python
"""A fixed-batch fitting diagnostic, deliberately separate from corpus training."""
import json
import torch
from gpt_config import Config
from gpt_data import windows
from gpt_model import GPT
from gpt_objective import nll_sum
from gpt_tokenizer import ByteBPE


def fit(updates=300):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = GPT(Config(dropout=0.0))
    x, y = windows(["red cat", "blue dog"], ByteBPE(), 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    def loss():
        total, count = nll_sum(model(x), y)
        return total/count
    before = loss().item()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    after = loss().item()
    return {"updates": updates, "before_nll": before, "after_nll": after,
            "fitting_target": 0.1, "passed": after < 0.1,
            "scope": "two memorized strings, no held-out claim"}


if __name__ == "__main__":
    print(json.dumps(fit(), indent=2))
```

Run `python fit_tiny.py`. The declared target is NLL below 0.1 after 300 updates. Your exact result belongs in your report; the [author run](project/reference-runs.md) records its own result. If it fails, inspect shifts/masks, finite gradients, parameter changes and effective LR first. Do not begin a large run to compensate for a broken tiny one.

The loop's order matters: clear gradients → forward → summed loss/valid count → backward → clip → step. Forgetting `zero_grad` accumulates old updates; calling `step` before backward uses no new gradient. Dropout is disabled here to make memorization a clean diagnostic.

You now have a model that can change its weights to fit a known batch. Next test whether the changes help unseen text, and make the model use its own predictions during generation.

[Continue to Lesson 8](lessons/08-evaluate-and-generate.md)

### Complete files from this lesson

[gpt_objective.py](examples/gpt_objective.py) · [fit_tiny.py](examples/fit_tiny.py)
