## Make the arrays small enough to inspect

Use B=2, T=8, D=64, H=4 and V=260 solely to work out shapes. The project proposes a maximum context of 64; this worked batch only contains eight positions. No tensor allocation or parameter count is claimed here.

[[table:shapes]]
[[diagram:attention]]

### Derive the restriction from the task

Suppose position 0 is meant to predict the next token using only its permitted prefix. If its output can use a later input value, changing that value can change the prediction before the prefix changes. This would violate our prefix-only task. Therefore the mixture must give forbidden future values zero weight. This is a consequence of the stated task and attention mixture, not a claim that every attention model must be causal.

For a hand calculation, use one scalar value per token, V=[2,4,8], and zero attention scores. A visible row with n keys assigns each weight 1/n. This removes learned projections so we can isolate the mask and weighted sum.

[[table:visibility]]

Before the explanation, test a tempting implementation shortcut:

[[practice:zero-score-check]]

Softmax turns a finite score s into a positive factor exp(s). Zeroing a forbidden score gives exp(0)=1, so it still receives positive weight after normalization. In this toy, all-zero unmasked scores distribute the mass uniformly across all three keys. A forbidden key needs zero weight; the idealized mask uses negative infinity before normalization. The last row averages 2+4+8; the first row can use only its first value. The causal rows here always have at least one allowed key; an all-masked row would need an explicit policy.

[[scenario:leakage]]
[[practice:causal-check]]
[[diagram:targets]]

Work out the alignment before a loss function is written. At input position t, our next-token task asks for token t+1. This motivates shifting the target array by one position. A causal mask and correct targets solve different problems: the mask prevents seeing later inputs, while the shift specifies the answer each position must predict. A correct mask cannot repair training against the wrong target.

[[practice:shift-check]]

### Run both language examples after predicting

Read the [Python example](examples/causal_trace.py) or [C++ example](examples/causal_trace.cpp). Both print the causal and unmasked averages, changed-prefix behavior and shifted arrays as JSON. They use the standard libraries and no learned parameters. Compare with your hand calculation, then change the values and recompute the expected result.

From workspace home, run:

```bash
./ai learn examples --lesson projects/p01/gpt-1-foundations/v1
```

This checks toy semantics in both languages. It does not verify model projections, autograd, a tokenizer or training.
