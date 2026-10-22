## Lesson 9 — SGD, momentum, Adam and weight decay

**Goal:** compute updates and explain which state an optimizer must remember. Prerequisites: gradient, objective/reduction and tensor parameters. Optimizers update parameters; they do not evaluate generalization or fix a wrong target/mask.

### SGD and momentum

For parameters θ and gradient gₜ=∇L(θₜ), SGD takes θₜ₊₁=θₜ−ηₜgₜ. A mini-batch gradient estimates a full training objective's gradient under the sampling/reduction assumptions. “Stochastic” refers to sampling, not random updates without a derivative.

One momentum convention is vₜ=μvₜ₋₁+gₜ, θₜ₊₁=θₜ−ηvₜ. Starting v₀=0, μ=.9 and gradients `[2,0]` gives velocities 2 then 1.8. Even a zero current gradient need not produce zero movement. An alternative convention uses `(1−μ)g`; it rescales the accumulator and therefore must be paired with the intended learning rate. Do not combine formulas from different conventions unnoticed.

Momentum can accumulate a persistent direction and damp oscillation in some problems, but it also adds history and another hyperparameter. Finite loss descent on every stochastic update is not guaranteed.

### Adam tracks first and second moments

For each coordinate, initialize m₀=v₀=0 and use

$$\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\hat m_t&=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t),\\
\theta_{t+1}&=\theta_t-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon).
\end{aligned}$$

These are coordinatewise operations. v tracks an exponential second moment, not centered variance. Bias correction compensates for starting the moving averages at zero; it is not the model's affine bias. ε protects the denominator and affects tiny-gradient behavior.

Use θ=2,η=.01,β₁=.9,β₂=.999,ε=1e−8. At gradient30, m₁=3,v₁=.9, corrected moments30 and900, so θ≈1.99. At the next gradient0, m₂=2.7,v₂=.8991; corrected m₂≈14.210526,v₂≈449.774887. The next parameter is approximately1.983299417. History continues to move it. Resetting Adam before that second step produces a different update.

[[diagram:prereq-adam]]

Adam's normalization makes the first step approximately independent of a large positive gradient's magnitude, but ε, history, sign, schedule and interactions with the objective still matter. It does not make gradient reduction or clipping irrelevant. The original Transformer uses Adam with β₁=.9, β₂=.98 and ε=1e−9, rather than the generic values in this hand fixture.

### L2 penalty and decoupled weight decay

Adding `(λ/2)||θ||²` to a loss adds λθ to its gradient. Under plain SGD, θ−η(g+λθ) equals `(1−ηλ)θ−ηg`. With Adam, putting λθ inside g changes both moment accumulators and passes the penalty through adaptive scaling.

AdamW instead applies a direct decay step separately from moment estimation: roughly `(1−ηλ)θ−η·adam_update(g)`. Therefore coupled L2 and decoupled decay are different under Adam. Decay is often restricted to selected parameters; state whether biases and normalization scales are included. Changing a parameter group changes the objective/update contract. The original paper's optimizer specification is Adam; do not silently label it AdamW.

### Implement and inspect a controlled update

The standard-library [Adam reference](examples/prerequisite_math.py) stores parameters, m, v and step count explicitly. Run its known-answer checks before using a large model. In PyTorch, `optimizer.state_dict()` stores moment state and parameter-group settings, while the model's state dict stores weights. Both are necessary for equivalent continuation.

```python
import torch
p = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
o = torch.optim.Adam([p], lr=.01, betas=(.9,.999), eps=1e-8)
p.grad = torch.tensor(30., dtype=p.dtype); o.step()
p.grad = torch.tensor(0., dtype=p.dtype); o.step()
assert abs(p.item()-1.983299417) < 1e-9
assert o.state[p]['step'].item() == 2
```

A missing gradient (`None`) and a zero gradient tensor can differ: some optimizers skip parameters with no gradient, including state/decay behavior. Clearing with `set_to_none=True` is an intentional choice. Never mutate learned tensors with arbitrary `.data` arithmetic to work around autograd; use the optimizer or a deliberate `no_grad()` update for a numerical exercise.

[[practice:optimizer-transfer]]

### Notes and source

Calculate the two moment updates, explain why the zero-gradient step moves and derive the SGD equivalence before explaining AdamW's difference. List which optimizer state a checkpoint needs. Sources: [Adam paper](source:adam-paper), [PyTorch Adam](source:adam), [decoupled decay paper](source:adamw-paper). Continue with [training-step control](lessons/10-training-steps.md).
