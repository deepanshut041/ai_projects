## Lesson 14 — Sequence order and positional encoding

**Goal:** show why content attention needs an order signal and derive the original sinusoidal representation. Prerequisites: embeddings, matrix multiplication, sine/cosine basics. Order and causality are related but distinct: a mask restricts information flow; positions describe location.

### Content operations alone are permutation equivariant

A shared FFN transforms each token independently. Reordering rows then applying it is the same as applying it then reordering. Unmasked self-attention has the same property when Q,K,V all come from the reordered inputs. For permutation matrix P,

$$\begin{aligned}
Q'&=PQ,\ K'=PK,\ V'=PV,\\
S'&=PSP^T,\ A'=PAP^T,\\
O'&=A'V'=PO.
\end{aligned}$$

Row-wise softmax respects the row/column permutation. Thus outputs track the permutation, but the operation has no intrinsic idea of “first” or “third.” Reordering `dog bites person` changes which content occurs at a position, yet content-only attention cannot infer the original arrangement from the unordered set. A causal mask breaks some permutation symmetry, but it is not a substitute for the paper's positional representation. The proof above assumes no position-dependent mask or other positional signal.

### The original sine/cosine table

For position p≥0 and even width D, use coordinate pair i:

$$\begin{aligned}
\omega_i&=10000^{-2i/D},\\
PE(p,2i)&=\sin(p\omega_i),\\
PE(p,2i+1)&=\cos(p\omega_i).
\end{aligned}$$

Sine and cosine are the vertical and horizontal coordinates on a unit circle; an angle of2π radians completes a turn. Each pair changes phase as the position advances. Different pairs have different frequencies. With D=4, ω₀=1 and ω₁=.01. At p=0 the row is `[0,1,0,1]`. At p=1 it is approximately `[.841471,.540302,.010000,.999950]`; at p=2, `[.909297,−.416147,.019999,.999800]`. Angles are in radians. The denominator uses the feature-pair index, not the token position.

```python
import math
D = 4
pe = lambda p: [value for i in range(D//2)
               for value in (math.sin(p*10000**(-2*i/D)),
                             math.cos(p*10000**(-2*i/D)))]
assert pe(0) == [0.,1.,0.,1.]
assert abs(pe(1)[2]-.009999833334166664) < 1e-12
```

For an odd D, define the final unpaired sine coordinate consistently. Do not assume a vectorized assignment of a shorter cosine slice has the same length as its sine slice.

### A fixed shift is a pairwise linear transform

For angle a=pω and shift b=kω, the addition identities give

$$\begin{bmatrix}\sin(a+b)\\\cos(a+b)\end{bmatrix}
=\begin{bmatrix}\cos b&\sin b\\-\sin b&\cos b\end{bmatrix}
\begin{bmatrix}\sin a\\\cos a\end{bmatrix}.$$

For a fixed k, each pair can express the shifted position with a linear transform independent of p. This motivates access to relative offsets; it does not prove the learned model will extrapolate accurately beyond its training lengths. Periodic components, finite precision and training distribution still matter.

### Add position while preserving feature width

The input is `sqrt(D)*E[id]+PE[position]`, shape `[B,T,D]`. Adding preserves D so residual and projection interfaces remain compatible. Concatenation would give width2D and require a different architecture. Position features do not erase token identity; the next learned layers can use both signals, subject to capacity and optimization.

A fixed table can be a registered buffer, moves with the module, and has no learned parameters. A learned absolute table of context length L has LD parameters and needs a rule for unseen positions. Relative biases and rotary encodings are later alternatives with different operations. Recognize them when reading modern models, but implement the original table before substituting a variant.

For padded tokens, adding PE produces nonzero features even if the PAD lookup row is zero. Continue to use proper key and loss masks. At generation time, a new token's position must follow the actual prefix; a cache implementation that restarts every token at position0 changes the model.

The reusable [foundation probes](examples/transformer_foundation_probes.py) implement these small calculations; [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:position-transfer]]

### Notes and source

Derive the permutation relation and pairwise shift identity, calculate one table row and draw addition versus concatenation. Explain why “fixed” does not imply proven length extrapolation. Source: [original positions](source:transformer-paper), [D2L position discussion](source:d2l-position). Continue with [attention](lessons/07-attention.md).
