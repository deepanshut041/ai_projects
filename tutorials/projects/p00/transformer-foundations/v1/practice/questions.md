# Predict before running

Try each changed case before reading its worked answer. Explain the coordinate, axis or state owner; naming an API alone is not enough. These are review prompts, not stored learner answers or a mastery score.

1. **Input or parameter?** With x=3,w=2,target 1, change x to 4. Prediction 8/error 7. The parameter and target did not change.
2. **Bias broadcast:** Add 1 to one output bias for two input rows. That output coordinate changes in both rows; the other coordinates stay fixed.
3. **Rectangular product:** X[4,3] times W[3,2] produces[4,2]. Three input features are summed; four examples and two output features remain.
4. **Stored weight:** nn.Linear(3,2).weight is[2,3]. The forward expression uses X times the stored weight's transpose plus bias.
5. **Population variance:** For[1,3], mean 2 and population variance 1. Dividing squared deviations by one instead of two gives a different statistic.
6. **Norms:** Vector[3,4] has L2 norm 5. Normalizing that vector, clipping a gradient vector and normalizing an attention probability row solve different problems.
7. **Stable probabilities:** [0,ln 3] gives[.25,.75]. Adding 1000 to both logits changes neither probability; evaluate after subtracting the maximum.
8. **Underflow-safe loss:** For logits[0,−1000] and target 1, negative log probability is 1000 under log-sum-exp. Computing log of a rounded probability 0 would fail.
9. **Gradient sign:** A positive dL/dw says a sufficiently small increase in w raises L locally. A huge step in the opposite direction can still overshoot.
10. **Loss derivative:** For a softmax/cross-entropy row, changing the target switches which coordinate subtracts 1 from the probability gradient. The gradient is not just the probability vector.
11. **Accumulated gradients:** Two backward calls add contributions unless gradients are cleared. Reset once at the intended update boundary; do not accidentally erase earlier microbatches.
12. **Weighted loss:** Microbatch counts 2 and 6 require denominator 8 across their loss sums. Equal weighting of their means assigns each example in the small group three times the weight.
13. **Byte coordinate:** café has 4 code points and 5 UTF-8 bytes. The two bytes for é share original character interval[3,4).
14. **Unknown content:** If a character vocabulary maps both unseen b and z toUNK, decoding that ID cannot recover which original character was supplied. Serialization cannot undo information loss.
15. **Mapping identity:** Two vocabulary tables of equal size with different order require different embedding interpretation. Compare the saved order/hash, not only dimensions.
16. **Mask or label?** A False key-mask position controls information access. A label−100 controls ignored supervision under the declared loss. One does not automatically implement the other.
17. **Half-open span:** Original text[4:8] in The café is open. returns café; UTF-8 bytes[4:8] end in a partial character. The full byte span is[4:9).
18. **Affine composition:** Removing the nonlinearity from Linear→activation→Linear leaves an affine map. Increasing its width alone does not make that composition nonlinear.
19. **Eval or no_grad?** eval selects evaluation behavior for layers such as dropout. no_grad controls graph recording. Either can be used without automatically applying the other.
20. **Resume:** Identical loaded weights do not guarantee an identical next Adam update. Restore moments/counters, scheduler, sampler and RNG after compatible construction and before requesting the next batch.
21. **Metric boundary:** Perfect classification of this tiny structured held-out grid is evidence for those points/rule, not a probability of success on arbitrary real data.
22. **Hosted recovery:** A local directory round trip and cleared notebook do not prove Drive recovery. Delete the hosted runtime, recreate exact source/dependencies and require restored then advanced updates.

For detailed worked calculations, return to the corresponding lesson's revealable practice. [Explain the complete system](practice/explain-your-model.md) ties these changed cases into one model trace.
