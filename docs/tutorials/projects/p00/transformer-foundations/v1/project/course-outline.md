# Tutorial 0 — foundations for Attention Is All You Need

This is the outline used for the complete course. Each numbered lesson owns a specific topic. Essential explanations belong in that lesson: definitions, a derivation, worked numbers, tensor shapes, runnable checks, failure cases and exercises. Python programming is assumed. The order follows prerequisites; it does not stretch a small prediction story into a course.

The goal is to understand every operation and training choice needed to read the original Transformer paper and begin its implementation. This includes the concepts behind the equations, not just how to call PyTorch. Course 1 then develops the complete translation and RoBERTa builds; Course 2 develops Small GPT.

## The lesson outline

| Lesson | Main topic and coverage | Worked calculation and practical check |
|---|---|---|
| 1 | **Linear algebra and neural maps**: scalars, vectors, matrices, tensors; indices, dot products, matrix multiplication, transpose, norms, affine maps, parameter sharing and operation counts. | Multiply a rectangular matrix by hand; implement `linear()`; distinguish feature and example axes and PyTorch's stored weight orientation. |
| 2 | **PyTorch tensor and module contracts**: shapes, rank, dtype, device, indexing, broadcasting, views/strides, reshape/transpose, parameters, buffers and registered submodules. | Trace an unequal-dimension tensor through a transpose; reject a legal broadcast with the wrong meaning; inspect a module's state. |
| 3 | **Calculus for learning**: derivatives, partial derivatives, gradients, chain rule, directional derivatives, Jacobians, vector–Jacobian products and local descent. | Derive both weight and bias gradients; compare central differences; calculate when a descent step overshoots. |
| 4 | **Probability, statistics and information**: categorical distributions, mean/variance, expectation, conditional probability, likelihood, independence assumptions, entropy, cross-entropy and KL. | Turn sequence probabilities into summed log likelihood; distinguish entropy, cross-entropy and the uniform baseline. |
| 5 | **Feedforward networks and activations**: affine composition, nonlinear expressiveness, hidden features, ReLU, sigmoid, tanh, GELU and parameter counts. | Show why stacked affine maps collapse; calculate a two-layer forward pass and an opposite-sign classification counterexample. |
| 6 | **Initialization and signal scale**: zero/symmetry failure, fan-in/fan-out, Xavier/He assumptions, activation/gradient variance and the attention scaling connection. | Derive variance of a weighted sum; compare initialization scales; explain why a dot product grows with its dimension. |
| 7 | **Backpropagation and autograd**: computation graphs, saved forward values, local derivatives, shared-gradient sums, matrix backward rules, accumulation, detach and gradient checks. | Derive every parameter and input gradient of a complete 2→2→2 network; compare analytical values, finite differences and float64 autograd. |
| 8 | **Loss functions and numerical stability**: MSE, binary and categorical cross-entropy, logits versus probabilities, log-sum-exp, loss reduction, ignored targets and label smoothing. | Derive the softmax/cross-entropy gradient; calculate a confident wrong prediction; check extreme logits without underflow. |
| 9 | **Optimizers**: SGD, momentum, Adam, moment estimates, bias correction, epsilon, coupled L2 and decoupled AdamW decay. | Hand-compute two Adam updates; change gradient magnitude and optimizer history; distinguish weight decay from a loss penalty. |
| 10 | **Training steps and learning-rate control**: forward/backward/update order, gradient clearing, mini-batches, accumulation, warmup, the paper's inverse-square-root schedule, global clipping and precision. | Calculate schedule values and clipping; compare unequal micro-batches to one combined batch; trace one optimizer update. |
| 11 | **Datasets, batching and supervision**: train/validation/test roles, leakage, sampling, variable lengths, collation, token/record denominators, padding IDs, labels and masks. | Pad unequal sequences; keep PAD, ignored labels and attention masks distinct; compute a valid-token mean. |
| 12 | **Text encoding and tokenization**: Unicode, code points versus bytes, normalization, vocabulary IDs, character/byte/subword tokenization, BPE, special tokens, frozen artifacts and offsets. | Encode `café`; train deterministic boundary-safe BPE merges; preserve original text coordinates and reload the exact tokenizer. |
| 13 | **Embeddings and the vocabulary interface**: one-hot lookup equivalence, learned tables, repeated-ID gradient sums, padding rows, vocabulary identity, output projection and weight tying. | Look up a small table; backpropagate repeated IDs; calculate tied/untied parameter counts and the paper's embedding scale. |
| 14 | **Position and sequence order**: permutation equivariance without position, sinusoidal encoding, frequencies, addition versus concatenation, learned positions and brief relative-position context. | Calculate sinusoidal values and a position shift; show an unordered attention operation cannot recover sequence order. |
| 15 | **Dot-product attention**: query/key/value roles, similarity, learned projections, row-wise softmax, scaled scores, weighted values and attention gradients. | Work through QKᵀ, scaling, softmax and AV with rectangular query/key lengths; change only the values. |
| 16 | **Attention masks and causality**: valid-key masks, causal masks, decoder prefixes, cross-attention masks, padded queries, all-masked rows and API conventions. | Draw rectangular and triangular masks; verify future-token invariance and a declared safe policy for an all-masked row. |
| 17 | **Multi-head attention**: separate projections, head splitting, axes, parallel mixtures, concatenation, output projection, head width and cost. | Trace B×T×D through all heads; calculate parameter counts; show splitting alone does not create learned specialization. |
| 18 | **The position-wise feedforward sublayer**: expansion, activation, contraction, shared per-position weights, token mixing versus feature mixing and capacity/cost. | Apply one FFN to two positions; calculate the paper's D→4D→D parameters and compare its cost with attention. |
| 19 | **Residual connections and depth**: matching dimensions, identity paths, residual Jacobians, gradient transport, pre-norm/post-norm and architectural consequences. | Differentiate x+F(x); compare zero-sublayer pre-norm and post-norm outputs without claiming guaranteed trainability. |
| 20 | **Layer normalization**: normalization axes, population variance, epsilon, learned scale/shift, gradient coupling, train/eval behavior and BatchNorm comparison. | Normalize a token's feature vector by hand; distinguish normalizing features from averaging tokens; test a constant vector. |
| 21 | **Regularization and generalization controls**: dropout, inverted scaling, attention/residual/embedding placement, weight decay, label smoothing and early stopping. | Derive dropout expectation/variance and a smoothed target; explain the difference between `eval()` and `no_grad()`. |
| 22 | **Encoder–decoder learning and inference**: source/target streams, self- versus cross-attention, BOS/EOS, target shift, teacher forcing, autoregressive factorization and greedy/beam decoding. | Build decoder inputs/labels from one target; work through a short conditional sequence score and stopping rule. |
| 23 | **Evaluation, debugging and a complete small training experiment**: overfit-a-batch, baselines, validation, perplexity, accuracy, calibration limits, leakage probes, gradient/activation checks and reproducibility. | Train the existing small nonlinear classifier with the shared trainer; diagnose failure modes; explain why this is not translation evidence. |
| 24 | **Checkpointing and reproducible recovery**: complete training state, optimizer/schedule/RNG/sampler position, identity, verified immutable generations and fresh-process continuation. | Interrupt at an update boundary; recover the same next batch/update; test corrupt and incompatible checkpoints. |
| 25 | **Read the paper and assemble the foundations**: map the original architecture/equations/hyperparameters to the lessons, trace one translation batch, distinguish original choices from later variants, measure compute and use the same source in Colab. | Annotate the original paper's attention/FFN/position/schedule equations; produce a shape-and-mask diagram and a readiness record for Course 1. |

## What each lesson must contain

1. A topic-specific learning goal and the prerequisite concepts it uses.
2. An explanation of what the operation means, why it is needed and what assumptions it makes.
3. Defined notation and dimensions before formulas, followed by a derivation and small worked numbers.
4. A runnable reference probe or a precise implementation exercise, with expected outputs and meaningful checks.
5. Common mistakes and a changed-condition exercise with a revealable worked answer.
6. A notes prompt that asks for my own derivation, prediction and connection to the Transformer.

## Coverage and boundaries

Lessons 1–11 establish mathematical and training foundations. Lessons 12–22 explain representation, attention and all the original architecture's core sublayers and sequence contracts. Lessons 23–25 verify those foundations, preserve work and connect them to the complete implementation.

The original paper is the architecture reference: post-norm, ReLU FFN, sinusoidal positions, scaled dot-product/multi-head attention, shared embedding/output weights and embedding scaling, Adam with its stated betas, warmup schedule, dropout and label smoothing. Later practices such as pre-norm, AdamW and GELU are explained as comparisons, not silently substituted into the 2017 design.

CNNs, advanced measure theory, distributed training, diffusion and reinforcement learning are outside this prerequisite course. Relative positions, mixed precision and decoding alternatives receive enough context to recognize the distinction; they do not displace the foundations or become extra model projects.

## Study sequence

Work through each topic in the order above. Derive and predict before running the probes or revealing the worked answer, then keep your own notes. The source export and optional cleared study notebook follow the same topics. The implementation/readiness records help resume actual exercises and distinguish authored reference checks from your work. In recovery, verify complete-generation acceptance separately from storage durability; the reference uses a completion marker, not an fsync/atomic-rename guarantee.

Existing lesson URLs are compatibility identities, so some filenames retain their older numbers. The displayed lesson numbers and navigation follow this outline. The actual learner child still resumes its saved `linear()` exercise; authoring this course does not perform learner work or establish hosted recovery.
