# Explain my foundations and Transformer contract

I will draw and explain these boards using my own calculations. A passing reference test is supporting evidence; it does not provide my answer.

1. Draw an affine→activation→logits→loss graph. Name the shared parameters and derive all gradients on the controlled two-layer fixture. Show one finite-difference comparison and a changed target.
2. Draw text→tokenizer artifact→IDs→embedding/position→Q/K/V. Distinguish original coordinates, integer row selection, floating features and contextual representations. Explain tied input/output gradient paths.
3. Draw one attention head with unequal query/key lengths, then split/merge multiple heads. Work row probabilities and weighted values. Change a key, a value, a padded position and a future token separately and predict the effect.
4. Draw the original encoder/decoder sublayers. Explain residual stream alignment, feature-axis LayerNorm, position-wise ReLU FFN and dropout placement. Contrast pre-norm as a different formula.
5. Draw a teacher-forced source/target batch and a generated prefix. Name BOS, EOS, PAD, ignored labels and each visibility mask; explain why parallel training does not expose future targets.
6. Draw a training update and recovery boundary. Explain valid-target reduction, gradients, moments, schedule/clipping, split-safe evaluation and every saved state needed for the next update.

For each board, identify one failure that a shape-only test would miss and propose a changed-condition check. Explain what the small classifier demonstrates and what remains untested for translation and hosted recovery. Return to the relevant topic lesson when I cannot derive a step rather than memorizing a diagram.
