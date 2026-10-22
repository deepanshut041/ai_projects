## Lesson 25 — read the paper and assemble the foundations

**Goal:** map the original Transformer's equations to concrete tensors, trace a complete translation batch, explain its computational comparisons, and carry reproducible source/training evidence into Course 1. Prerequisites: Lessons 1–24. The course now explains the mechanisms needed to read *Attention Is All You Need*; assembling and training the complete translation model is the next implementation task.

### Read each paper equation as a contract

Use the [original paper](source:transformer-paper), sections 3–5, alongside this table. Let $B$ be batch size, $S$ source length, $T$ decoder length, $D$ model width, $H$ heads, $d_k=d_v=D/H$, $F$ FFN width, and $V_t$ target vocabulary size. The usual equality of head key/value widths is this architecture's choice, not a mathematical requirement of attention.

| Paper operation | Required inputs, operation and output | Foundation lesson |
| --- | --- | --- |
| Scaled attention, Eq. 1 | $Q[B,H,T_q,d_k]$, $K[B,H,T_k,d_k]$, $V[B,H,T_k,d_v]$; permitted scores → key-axis softmax → weighted values $[B,H,T_q,d_v]$ | [Dot-product attention](lessons/07-attention.md), [masks](lessons/16-masks.md) |
| Multiple heads | Separate learned Q/K/V projections, split feature axes into heads, concatenate head outputs, then apply the output projection | [Multi-head attention](lessons/17-multihead.md) |
| Position-wise FFN, Eq. 2 | For each token independently, $\operatorname{ReLU}(xW_1+b_1)W_2+b_2$ with $D\to F\to D$ | [FFN sublayer](lessons/18-positionwise-ffn.md) |
| Residual/normalization | Same-width addition around each sublayer, then feature-axis LayerNorm in the original post-norm order | [Residuals](lessons/19-residual.md), [LayerNorm](lessons/20-normalization.md) |
| Vocabulary interface | Token IDs → learned vectors scaled by $\sqrt D$; decoder states → vocabulary logits, with compatible shared weight identity | [Encoding](lessons/05-tokenizer.md), [embeddings](lessons/13-embeddings.md) |
| Positional encoding | Add sine/cosine features of width $D$ to token embeddings, with a declared position index and frequency rule | [Positions](lessons/14-position.md) |
| Training schedule, Eq. 3 | Positive optimizer-update index, model width and warmup count determine the rate; optimizer history must continue across sessions | [Optimizers](lessons/09-optimizers.md), [training steps](lessons/10-training-steps.md) |

For attention, write the masked score and weighted-value equations with all axes identified:

$$
\begin{aligned}
C_{b,h,i,j}&=\frac{Q_{b,h,i,:}\cdot K_{b,h,j,:}}{\sqrt{d_k}}+M_{b,h,i,j},\\
A_{b,h,i,:}&=\operatorname{softmax}_{j}(C_{b,h,i,:}),\\
O_{b,h,i,:}&=\sum_j A_{b,h,i,j}V_{b,h,j,:}.
\end{aligned}
$$

Here $M$ is zero for allowed keys and negative infinity for forbidden keys, subject to the explicit empty-row policy from Lesson 16. The normalization index $j$ enumerates **key positions**. Vocabulary softmax at the final output enumerates a different axis. Write both on your board; saying “apply softmax” without its axis hides a consequential distinction.

### Keep original settings and later comparisons visible

The paper's base configuration uses six encoder and six decoder layers, $D=512$, $H=8$, $d_k=d_v=64$, and $F=2048$. It uses post-norm, ReLU and sinusoidal positions. Its Adam settings are $(\beta_1,\beta_2,\epsilon)=(0.9,0.98,10^{-9})$, with warmup 4,000; base dropout and label smoothing are 0.1. These are reference choices, not dimensions to allocate implicitly. [Paper settings](source:transformer-paper).

Pre-norm changes the residual computation; GELU changes the FFN activation; AdamW changes decay semantics; learned or rotary positions change the position contract. They can be valid later designs, but silently substituting them does not implement the same specification. [My source notes](project/source-notes.md) records these distinctions and the implementation details the main paper leaves unspecified. A miniature profile can retain the architecture's operations while reducing layer counts, widths, vocabulary and lengths.

### Trace one complete miniature translation batch

Use $B=2,S=3,T=4,D=8,H=2,F=32,V_t=10$, one encoder layer and one decoder layer. Source IDs are `[[3,4,PAD],[5,6,7]]`; shifted decoder IDs are `[[BOS,4,5,6],[BOS,7,PAD,PAD]]`, with labels `[[4,5,6,EOS],[7,EOS,PAD,PAD]]`. These are pedagogical vocabulary IDs as in Lesson 22.

1. Source/decoder IDs enter their embedding lookup as integers. Scaling and adding positions give floating tensors $[2,3,8]$ and $[2,4,8]$. During evaluation, dropout is disabled.
2. Encoder Q/K/V split into $[2,2,3,4]$. Source self-attention scores are $[2,2,3,3]$; the first record's PAD key is excluded. Merge heads, project, add the residual and normalize features. The position-wise FFN and its residual/normalization produce memory $[2,3,8]$.
3. Decoder masked self-attention scores are $[2,2,4,4]$. Query index $i$ can read legal decoder keys $j\leq i$, never its unknown target label through a future position.
4. Cross-attention takes queries from decoder states and keys/values from encoder memory. Scores are $[2,2,4,3]$; source padding is excluded. Residual addition returns to the decoder stream. The FFN preserves decoder shape $[2,4,8]$.
5. The output projection gives logits $[2,4,10]$. Six valid labels, including two EOS labels, contribute to summed loss. Divide by six for this token-mean objective, backpropagate, then apply the one-update contract from Lesson 10.

At generation time, the labels disappear. Begin with BOS and produce each next prefix token until EOS or the length cap. Source encoding can be reused for that request; it must not leak between requests. Draw the source memory, target prefix, all three masks and the loss boundary before implementing a complete model. The [sequence contract diagram](lessons/22-encoder-decoder.md) supplies the corresponding visual structure.

### Count parameters and working tensors separately

For this miniature design, explicitly choose biases on all Q/K/V/output projections and both FFN maps, five affine LayerNorms, one shared $10\times8$ source/target/output matrix, and a vocabulary output bias. Shared vocabulary identity is a precondition of this choice. Then:

$$
\begin{aligned}
P_{\mathrm{MHA}}&=4D^2+4D=288,\\
P_{\mathrm{FFN}}&=2DF+F+D=552,\\
P_{\mathrm{LN}}&=2D=16,\\
P_{\mathrm{total}}&=3(288)+2(552)+5(16)+80+10=2138.
\end{aligned}
$$

This is an authored count for the stated specification, not the instantiated classifier's 42 parameters or a claim about every library Transformer. Fixed sinusoidal values are buffers rather than learned parameters. Untied or separate-vocabulary embedding tables change the count.

The three explicit attention-score arrays contain $2×2×(3^2+4^2+4×3)=148$ scalars, or 592 bytes in FP32. Scores, probabilities, saved backward values and workspace are additional allocations; optimized attention can avoid retaining full matrices. Doubling length at fixed width approximately quadruples explicit self-attention score storage, while parameter count stays unchanged. A checkpoint, peak device allocation and whole-process RAM are different measurements.

Run this complete arithmetic probe from `examples/` with the frozen interpreter:

```python
from transformer_foundation_probes import parameter_counts, transformer_learning_rate

c = parameter_counts(vocabulary=10, width=8, hidden=32, heads=2)
total = 3*c['multihead'] + 2*c['ffn'] + 5*(2*8) + c['one_tied_vocabulary_matrix'] + 10
assert c['multihead'] == 288 and c['ffn'] == 552 and total == 2138
score_scalars = 2*2*(3*3 + 4*4 + 4*3)
assert score_scalars == 148
rates = [transformer_learning_rate(s, width=4, warmup=4) for s in (1, 4, 16)]
assert rates == [0.0625, 0.25, 0.125]
print('parameters:', total, 'explicit score bytes:', 4*score_scalars, 'schedule:', rates)
```

The schedule uses $D^{-1/2}\min(s^{-1/2},s w^{-3/2})$ at $s\geq1$, with warmup $w$. At $s=w$, both branches meet; $s=0$ is undefined for the inverse-square-root term. The classifier's warmup-then-constant trainer is a simpler separately named schedule.

### Interpret the paper's computational comparison

For equal sequence length $n$ and width $D$, dense attention's score and value products require order $n^2D$ arithmetic, while a recurrent feature update costs order $nD^2$ over the sequence. An attention layer can calculate its legal query rows in parallel; a recurrent state depends on the preceding state, requiring order $n$ sequential operations. Global unmasked attention directly connects two token positions in one layer, whereas a recurrent chain has order $n$ path length. These are operation/dependency comparisons, not a guarantee of constant wall-clock time. [Paper section 4](source:transformer-paper).

The comparison table also describes convolutional feature work of order $knD^2$ for kernel width $k$. Expanding receptive fields across a suitable stack can give logarithmic path growth; ordinary undilated contiguous kernels may instead need order $n/k$ layers to span the sequence. We need that distinction to read the comparison, not a separate CNN/RNN implementation project.

Include projection and FFN work in actual budgets: Q/K/V/output projections cost order $nD^2$, and the FFN costs order $nDF$. For one length-four, width-eight record with $F=32$, the two attention-core matrix products use 256 multiply-accumulate terms, four projections use 1,024, and two FFN products use 2,048. Counting each multiply plus addition as two FLOPs changes the unit; declare it. Autoregressive inference still generates dependent target tokens sequentially even though teacher-forced training evaluates all masked rows together. Measure actual runtime and memory rather than converting those orders into an unmeasured speed claim.

### Connect the inference model to its reported evaluation

The paper's section 6.1 uses beam width four and length-penalty parameter $\alpha=0.6$. It also averages the last five base-model checkpoints or twenty big-model checkpoints to obtain inference weights. Those choices belong to the evaluated system, alongside its data/tokenization and metric settings. [Inference specification](source:transformer-paper). Lesson 22 explains beam search; Lesson 23 explains why a BLEU number requires its reference corpus and metric contract.

Parameter averaging forms $\bar\theta=K^{-1}\sum_{k=1}^{K}\theta^{(k)}$ in the same parameter coordinates. Averaging vectors `[1,3]` and `[3,5]` gives `[2,4]`; running a nonlinear model with that vector does not generally equal averaging the two models' predictions. Require compatible source/configuration, vocabulary identities, parameter names/shapes and the intended shared training trajectory. Matching widths alone cannot establish interchangeable learned coordinates.

An averaged inference model is a separate artifact. It does not contain the optimizer moments, schedule/RNG/sampler position or metric history required for the exact next update in Lesson 24. Keep the latest full resumable generation and the chosen averaged evaluation weights distinguishable; do not overwrite the recovery copy with a weight average.

### Carry portable source and personal notes into Colab

The [source exporter](examples/prerequisite_export.py) packages the same classifier/checkpoint sources and numerical/attention probes used locally; it excludes environments, data, weights and generated runs. From `examples/`, choose fresh destination filenames:

```bash
python prerequisite_export.py /tmp/tutorial0-topic-source.zip \
  --notebook /tmp/tutorial0-topic-colab.ipynb \
  --study-notebook /tmp/tutorial0-topic-study.ipynb
```

The report prints archive/member hashes and identifies the actual working-tree source snapshot. The cleared study notebook follows the topic lessons and imports source probes; its explicit export option refuses to overwrite an existing personal notes notebook. Keep your edited copy. The separate thin Colab notebook performs source verification, setup/checks and trainer calls rather than duplicating implementation code.

The hosted reference pins NumPy 2.5.4 and PyTorch 2.14.1+cpu, verifies actual imports and uses FP32 CPU with one thread. An unavailable wheel or changed imported version requires resolving/restarting the environment, not silently relabeling it compatible. Colab hardware/resources are temporary and variable. This CPU reference does not establish GPU compatibility or full translation training capacity. [Colab FAQ](source:colab-faq).

The generated notebook requests explicit Drive mounting, verifies a small write/read round trip, and chooses a persistent run root and stable run ID. Its initial slice is 25 additional updates toward an unchanged goal of 1,000; longer slices stay within 1,000 updates/60 minutes, with saves after 100 completed updates or five minutes and at normal end. These are workspace budgets, not guaranteed service limits. Persistence failure stops the trainer.

### Prove hosted recovery by removing hidden runtime state

1. Verify the uploaded source hash, dependencies and source checks. Select `new` for an empty Drive run, finish a short slice and record its verified generation/global update.
2. Choose **Disconnect and delete runtime**. Reconnect and repeat source acquisition, environment checks and tests from the cleared notebook.
3. Mount the same Drive location, retain the same run ID and fixed experiment configuration, and explicitly choose `resume`.
4. Record the selected generation, restored update, any skipped generation/rollback, advanced update, actual runtime/resource identity, save bytes and persistence latency.

A mere cell interruption or local directory round trip does not perform this drill. Source/dependency mismatch is a separate experiment unless compatibility is explicitly demonstrated. Hosted recovery remains pending until those actual observations exist; authoring this procedure does not close it.

[[diagram:prereq-handoff]]

### Record readiness with concrete evidence

Use the [course outline](project/course-outline.md) and [prerequisite map](project/prerequisite-map.md) to find a missing concept; use [my study notes](project/study-notes.md) for your derivations and changed-input predictions. Keep actual work in the [implementation record](project/implementation.md), compare only the stated conditions in [reference runs](project/reference-runs.md), and record open gates in [readiness](project/p01-readiness.md). [Source notes](project/source-notes.md), [research](research.md) and [local reference evidence](reference-verification.json) distinguish documented facts, authored examples and executed evidence. Use the [native practice collection](practice/questions.md) and [explain-my-model boards](practice/explain-your-model.md) when reviewing the topics. These records support the lessons; they are not a second required theory course.

You should be able to derive a stable loss/gradient, explain embeddings and positions, work through rectangular masked attention and head splitting, distinguish feature mixing from token mixing, and trace a shifted translation batch. Independently demonstrate a complete small training/evaluation run and full-state new-process recovery. Begin Course 1's complete translation/RoBERTa implementations; then Course 2 develops Small GPT. Opening all lessons or passing author tests does not establish your explanation, implemented model or hosted capability. The existing foundations child still resumes its saved `linear()` exercise until actual learner work changes that checkpoint.

[[practice:handoff-transfer]]

In [my study notes](project/study-notes.md), annotate paper equations 1–3 with axes, required saved state and one failure invariant each. Draw the complete batch above with all masks and six loss targets. Name the next component you will implement, its independent check, and the learner/hosted evidence that is still pending. Retain that specific next action in [the implementation record](project/implementation.md).
