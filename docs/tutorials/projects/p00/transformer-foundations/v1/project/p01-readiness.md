# My readiness for the Transformer builds

Tutorial 0 prepares the mathematics, representation, architecture and training contracts needed for Attention Is All You Need. Availability of the written course does not complete my learner implementation, model training or hosted recovery. This filename stays stable for existing links.

| Foundation | Evidence I will produce |
|---|---|
| Linear algebra/tensors | Hand-computed rectangular maps, parameter counts, axes/transpose/broadcast/storage and runtime contracts |
| Calculus/backpropagation | Derive scalar and full-network gradients; compare central differences and FP64 autograd; explain shared paths |
| Probability/loss | Conditional sequence likelihood, entropy/cross-entropy/KL, stable logits loss and valid-target denominators |
| Networks/optimization | Affine-collapse/expressiveness reasoning, initialization assumptions, Adam history, correct steps/schedules/clipping |
| Data/text | Split/identity policy, original Unicode/byte/token coordinates, exact frozen tokenizer and unequal-length collation |
| Embedding/position | Lookup gradient sums, padding/tied-table paths, vocabulary identity, scale and sinusoidal/ordering reasoning |
| Attention/heads/masks | Rectangular Q/K/V numerical calculation, analytic backward, future/padding invariance, empty-row policy, head split/merge/counts |
| Core sublayers | Position-wise nonlinear FFN, residual derivative/cancellation and original post-norm feature normalization |
| Regularization/seq2seq | Dropout expectation/placement/modes, explicit smoothing convention, BOS/target shift/EOS and source-versus-prefix contracts |
| Evaluation/debugging | Bounded nonlinear learning, simple baseline, one diagnosed failure, held-out totals and honest task/metric limits |
| Recovery | Fresh-process parity for model/optimizer/schedule/sampler/RNG, verified compatible fallback and remaining hosted drill |
| Paper explanation | Trace one source/target batch through the original design and explain each equation, shape, mask and training choice |

I will work the native exercises in the lessons and use [study notes](project/study-notes.md) for my own derivations. [Explain my model](practice/explain-your-model.md) gives a review board; [prerequisite map](project/prerequisite-map.md) connects the foundations to paper operations. No learner answers are prefilled.

## Course 1 and Course 2

[Course 1 — Transformer Foundations: Translation and RoBERTa](/records/tutorials/p00-transformer-course-v1/chapters/start/) applies these operations to the complete translation Transformer and MLM→extractive QA. Its source recipes, corpus/model/task checks and recovery requirements remain its own work. Tutorial 0 explains the operations; it does not supply a trained translation checkpoint.

[Course 2 — Small GPT](/records/tutorials/p01-small-gpt-v1/chapters/next-token/) follows actual required P00 learning/model/recovery gates. Its decoder-only architecture and pre-norm/learned-position choices are explicit differences. A small classifier is a foundation exercise, not a GPT model.

My actual implementation checkpoint remains P00-M1 in-progress, setup complete, `linear()` next in the existing foundations child. The course rewrite retains its contract and old continuation URL. Course preparation and author checks do not advance this status.

## Hosted evidence still to collect

Use the same trainer/source identity in the generated thin Colab notebook. Check persistent storage, save a complete verified generation, delete the runtime, recreate exact source/dependencies and resume/advance the same run. Record that evidence separately. Local fresh-process continuation does not establish Free resource availability, Drive survival, GPU performance or deleted-runtime recovery. The later complete-model projects retain their own hosted gates.
