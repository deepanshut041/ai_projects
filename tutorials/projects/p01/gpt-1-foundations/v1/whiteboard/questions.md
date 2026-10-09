# Challenge bank

Use one question at a time in a requested live rehearsal. Ask the learner to draw or predict first; the worked answers in the lesson are a later check.

1. Trace one token through the decoder and label every shape.
2. Draw the exact order of attention, residual addition and normalization.
3. Why does replacing a forbidden score with zero fail to mask it?
4. Which outputs change when a future value changes? Demonstrate with the example.
5. Construct inputs/targets for four token IDs and explain the last target.
6. How can loss look good while generation is wrong? Name a falsifying check.
7. Derive K/V storage and say which allocations are excluded.
8. Which additional measurements would an SSD offload study need?

Feedback records strong/partial/needs-investigation against the relevant objective and evidence. Never mark mastery or a model milestone from availability of the answer bank.

For a requested walkthrough, begin with `visible-prefix` only if its prerequisite is unclear. After a correct answer use `causal-check` with a changed future value; after confusion trace one position before widening the scope. Use `zero-score-check` to distinguish score from weight. Wait for each real answer before hints/feedback, accept unknown/skip, and keep any saved human answers opt-in. These are proposed routes, not recorded learner results.
