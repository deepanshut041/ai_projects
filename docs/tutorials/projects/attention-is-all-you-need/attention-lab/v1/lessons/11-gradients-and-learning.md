## 9. How the translation error reaches attention

Q/K/V projections were chosen by hand in our worked calculation. A trained translator must learn parameters that help predict the reference translation. To see where that learning signal comes from, follow the decoder beyond attention.

### Start with the actual task objective

For the target row whose prefix ends in “Nous”, the desired next token is “lisons”. Let h_t be that row's final decoder representation. A vocabulary projection calculates logits `z_t = h_t W_U + b_U`, one score for each possible target token. Softmax converts those scores into a next-token distribution p_t. Cross-entropy penalizes low probability on the reference token:

$$L_t=-\log p_t[\text{reference token}].$$

The derivative with respect to logits is `p_t - one_hot(reference token)`. That error flows backward through the vocabulary map, decoder computation, attention outputs, value contributions and query/key comparisons. W_Q, W_K and W_V can all receive gradients. No training file needs to say “this head should attend to source position 2”. Their selection behavior can be learned indirectly from the token-prediction objective.

![Task loss flowing back through decoder features to attention parameters](images/27-learning-path.svg)

This is the real-model learning path. The full translation model is not implemented locally yet. We will magnify one part of that path with a controlled objective on the gathered vector, so its derivative can be calculated completely instead of invoking a missing decoder.

### Isolate one retrieval error

Keep our first query, keys and values. For this derivative probe only, set a desired context `t = [2,0]`. The current context is `[1.339523,1.000000]`, so use half squared error:

$$L=\tfrac12\sum_m(c_m-t_m)^2\approx0.718115.$$

The error vector `r = c-t` is approximately `[-0.660477,1.000000]`. The first feature is too small and the second too large. Giving more relative weight to value `[2,0]` would help both errors. Increasing every score equally would do nothing, because the common shift cancels in softmax.

We are not assigning this desired context to the translation of “We”. It is a local objective chosen to expose the mechanism by which an error changes retrieval.

### Differentiate the computation, one dependency at a time

From the squared loss, `dL/dc = r`. Since `c = sum_j a_j v_j`, increasing weight a_j changes c in direction v_j, giving `dL/da_j = r dot v_j`.

Weights are coupled by the softmax denominator. Its derivative is `da_l/ds_j = a_l(delta_lj-a_j)`, where delta_lj is one when l=j and zero otherwise. Combining the terms gives:

$$\frac{\partial L}{\partial s_j}=a_j\,r\cdot(v_j-c).$$

The subtraction of c accounts for redistributing weight away from the current mixture when a source score increases. It is the reason we cannot treat each weight as an independent knob.

Finally, `s_j = q dot k_j / sqrt(2)`, so:

$$\nabla_q L=\sum_j\frac{\partial L}{\partial s_j}\frac{k_j}{\sqrt2}
\approx[-0.206595,0.353553].$$

This gradient describes how the chosen loss changes for small query changes. Gradient descent subtracts it: increase the first query coordinate and decrease the second. The direction follows from this fixture and objective, rather than being an intrinsic property of the word or query role.

### Update a shared parameter and inspect the consequence

The code makes W_Q trainable and holds W_K/W_V fixed, isolating the query path. `backward()` computes the gradient; it does not update parameters. The explicit subtraction inside `no_grad()` performs the update.

```python
import math
import torch

def attend(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    return weights @ v, weights

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
w_q = torch.eye(2, dtype=x.dtype).requires_grad_()
w_k = torch.eye(2, dtype=x.dtype)
w_v = 2 * torch.eye(2, dtype=x.dtype)
target = torch.tensor([[2., 0.]], dtype=x.dtype)

q = x[:1] @ w_q
k, v = x @ w_k, x @ w_v
context, weights = attend(q, k, v)
loss = 0.5 * ((context - target) ** 2).sum()
loss.backward()

print(context.detach())
print(loss.item())
print(w_q.grad)

with torch.no_grad():
    new_w_q = w_q - 0.1 * w_q.grad
    new_context, new_weights = attend(x[:1] @ new_w_q, k, v)
    new_loss = 0.5 * ((new_context - target) ** 2).sum()

print(new_context)
print(new_loss.item())
assert new_loss < loss.detach()
```

Input row 0 is `[1,0]`, so it selects W_Q's first row. That row receives gradient `[-0.206595,0.353553]`; the second row receives zero in this one-query probe. A rate of 0.1 changes the query to about `[1.020660,-0.035355]`, the context to `[1.345969,0.987501]`, and the probe loss to `0.701457`.

In a full training step, gradients from many target positions and examples accumulate into shared parameter matrices before the optimizer update. A later example calculates new attention weights with the updated parameters. Attention weights are intermediate results, not parameters that the optimizer directly stores for each token pair.

The local loss decrease verifies this derivative path. Translation quality requires the actual token objective, complete model, data, repeated updates and held-out evaluation. We can now read the reusable class knowing both what its forward pass computes and how its parameters could receive a task signal.

**Runnable copy:** [one query-projection update](examples/walkthrough/09_gradient.py).

[[practice:gradient-check]]
