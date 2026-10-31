# Explanation and changed-condition review

Try each question before revealing its worked answer. No learner response or score is stored here. These are a return-to-review aid; the same transfer checks appear beside their lessons.

## 1. An answer says “3 hours and a five-year warranty,” citing lumen-1. Which checks can pass, and what still fails?

<details><summary>Reveal worked answer and rubric</summary>

The charging fact matches and lumen-1 is a valid ID. The warranty claim is absent from the supplied evidence, so complete-answer support fails. The first exact fixture scorer is insufficient for this extra-claim case; review all material claims or use a stricter declared evaluator.

**Rubric:** Distinguish fact correctness, ID membership and semantic support; name the fixture limitation.

</details>

## 2. The TUI status runner exits 0 and prints Gemma 4. May we mark model compatibility verified?

<details><summary>Reveal worked answer and rubric</summary>

No. It validated/read configuration and printed the planned source. It did not load weights or generate tokens. Keep model_execution_verified false; answer remains exit 2. Compatibility needs actual load/template/EOS/placement/reload receipts.

**Rubric:** Trace status through config/dispatcher and name missing runtime evidence.

</details>

## 3. A backend loads the weights, but its template adds a different generation prefix. Is this the same frozen prompt experiment?

<details><summary>Reveal worked answer and rubric</summary>

No. Visible question equality does not establish token-sequence equality. Capture rendered messages/template/token IDs, fix or separately version the template policy, then repeat the compatibility probe before comparing artifacts.

**Rubric:** Explain why template identity changes the input, rather than treating load success as enough.

</details>

## 4. Using the four-bit fixture, what happens when cache allowance grows from 2 to 8 GiB?

<details><summary>Reveal worked answer and rubric</summary>

Weights remain 4.191 GiB. Add 8 GiB cache, 1 GiB workspace and 2 GiB reserve: 15.191 GiB exceeds 12 GiB. The 6 GiB increase is independent of weight compression. A measured backend allocation may differ from this budget model.

**Rubric:** Use the same pool and units; include reserve exactly once and distinguish estimate from measurement.

</details>

## 5. Two identical paragraphs occur in one manual. Why does the forward search cursor matter?

<details><summary>Reveal worked answer and rubric</summary>

Searching from the document start each time would attach both records to the first occurrence. Searching after the prior interval preserves occurrence offsets. The invariant D[start:end] equals passage text still needs the correct occurrence identity; retain full document digest and offset convention.

**Rubric:** Identify repeated-content ambiguity and Unicode character versus byte/token offsets.

</details>

## 6. Question A and B use different labels but the same manual. Can A calibrate/tune while B is a document-disjoint final example?

<details><summary>Reveal worked answer and rubric</summary>

No. Split document groups before questions. Shared text/facts violate this protocol even if labels differ. Group duplicate/related manual versions, freeze membership and keep final labels separate. A transductive design is possible only as a separately declared protocol.

**Rubric:** State the unit of splitting and why hidden labels alone do not prevent overlap.

</details>

## 7. Add 1000 to every logit. What should change in softmax and greedy choice?

<details><summary>Reveal worked answer and rubric</summary>

In exact arithmetic neither changes: a common exponential factor cancels. Subtracting max avoids overflow. Ties still need a deterministic policy and floating implementations can differ at tiny margins. A changed next token can alter later history.

**Rubric:** Derive shift cancellation and distinguish local numerical invariance from full runtime determinism.

</details>

## 8. With s=0.5, z=0 and integer codes -7 through 7, encode 1.25 and 10 using ties-to-even.

<details><summary>Reveal worked answer and rubric</summary>

1.25/0.5=2.5 rounds to even code 2, reconstructing 1.0 with error -0.25. 10/0.5=20 clips to 7, reconstructing 3.5 with error -6.5. The half-step bound applies to unclipped nearest rounding, not the clipped outlier.

**Rubric:** Show division, rounding, clipping and reconstruction separately.

</details>

## 9. Why do five four-bit codes need the count as well as three bytes?

<details><summary>Reveal worked answer and rubric</summary>

Three bytes contain six nibbles. The unused last nibble is padding, so decoding all six would invent an element. Preserve shape/count. Also preserve scale/zero point to recover real values; packing alone only recovers codes.

**Rubric:** Compute ceil(5/2)=3 and identify shape and numeric metadata.

</details>

## 10. Keep E from Lesson 10 but change x from [4,1] to [1,4]. Predict the output errors.

<details><summary>Reveal worked answer and rubric</summary>

Row 1 becomes -0.24(1)-0.02(4)=-0.32. Row 2 becomes -0.24(1)+0.01(4)=-0.20. Weight MSE is unchanged, but output error changes because the input weights error columns differently.

**Rubric:** Use E x with correct row/column axes, not the old output difference.

</details>

## 11. Q4 uses a GPU and BF16 uses the CPU. What can their timing comparison establish?

<details><summary>Reveal worked answer and rubric</summary>

It compares two deployment configurations. Quality on matched questions remains informative, but a timing difference cannot isolate weight quantization because placement changed. Keep a same-runtime/placement mechanism comparison where feasible and report the cross-placement row separately.

**Rubric:** Name the confound and preserve useful evidence without claiming a causal mechanism.

</details>

## 12. Duplicate every calibration sample once. What happens to G and average reconstruction loss?

<details><summary>Reveal worked answer and rubric</summary>

Neither changes because numerator and sample count both double. A sum would double; the mean does not. The observations are not new independent information, so duplication does not improve uncertainty or rank coverage.

**Rubric:** Explain the denominator and distinguish data volume from independent evidence.

</details>

## 13. Apply scale [4,1] to weights but forget inverse input scaling. Is the float operation preserved?

<details><summary>Reveal worked answer and rubric</summary>

No. W becomes [0.96,1.02] while x remains [4,1], giving 4.86 instead of 1.98. The identity requires multiplying weights by S and input by S inverse. Exact float equivalence does not imply equal quantized reconstruction.

**Rubric:** Calculate the counterexample and state both sides of the identity.

</details>

## 14. For G=[[4,0],[0,2]] and fixed e1=0.25, what e2 minimizes loss?

<details><summary>Reveal worked answer and rubric</summary>

e2=0 because the cross term is zero. Derivative is 4 e2; a nonzero second error only adds loss. The minimum is 4(0.25)^2=0.25. Correlation, not merely a large diagonal, makes nonzero compensation useful.

**Rubric:** Use the constrained derivative and explain the missing cross term.

</details>

## 15. The byte cap falls from 900 to 700 in Lesson 15. Can the 848-byte winner remain selected?

<details><summary>Reveal worked answer and rubric</summary>

No. Reapply feasibility. The 848-byte choice fails, and both group-32/four-bit layers cost 720 and fail too. First layer four-bit/group-128 plus second four-bit/group-32 costs 528+144=672 and fits, with proxy loss 0.12+0.30=0.42. Both group-128 costs 660 with loss 0.52. Enumerating the choices selects 672 bytes under the 700-byte cap; do not preserve the prior winner or silently relax the cap.

**Rubric:** Reject the prior winner, count metadata and enumerate rather than silently relax the cap.

</details>

## 16. A run omits the two questions that crashed. May the evaluator report a score on only its remaining IDs?

<details><summary>Reveal worked answer and rubric</summary>

Not as the frozen full comparison. Reject membership mismatch or retain explicit failure rows under a predeclared policy. Preserve the full denominator and raw failures. A partial-run score can be descriptive if labeled partial with completed/expected counts.

**Rubric:** Explain alignment, failure retention and denominator rather than hiding missing work.

</details>

## 17. Five output tokens arrive between t1=0.7 and tn=2.7. Is the defined decode rate 2.5 tokens/s?

<details><summary>Reveal worked answer and rubric</summary>

No. The first token is present at t1, leaving four inter-token intervals. The specified rate is (5-1)/(2.7-0.7)=2 tokens/s. A different tokens-over-runtime metric may use another denominator but must be named distinctly.

**Rubric:** Identify the observed interval and count, and handle n=1 as undefined for this metric.

</details>

## 18. At T=4096 the sliding owner still retains 512 tokens. What changes if batch doubles?

<details><summary>Reveal worked answer and rubric</summary>

Under the same-length independent-sequence fixture both owner allocations double, from 4,456,448 to 8,912,896 bytes. Sliding retention stays 512 per sequence. Physical page/static allocation and sharing policies still require actual backend inspection.

**Rubric:** Count owners once per declared sharing relation and batch/token axes correctly.

</details>

## 19. With probabilities [0.25,0.75], change values from [0,4] to [2,4] while holding keys fixed.

<details><summary>Reveal worked answer and rubric</summary>

Probabilities stay the same. Output changes from 3 to 0.25(2)+0.75(4)=3.5, so delta is 0.5. A key change would instead alter scores and renormalize the whole mixture. Do not attribute both pathways to one value-error formula.

**Rubric:** Compute the weighted value error and keep key probabilities fixed.

</details>

## 20. Newest complete generation is corrupt and no older verified generation exists. What should the loader do?

<details><summary>Reveal worked answer and rubric</summary>

Fail with no verified generation and retain rejection reasons. It must not label corrupt bytes loaded or silently choose a different source model. Atomic publication prevents partial visibility; integrity and compatibility still need checks, and crash durability is separate.

**Rubric:** Name explicit failure/fallback semantics and avoid claiming hash checks prove model quality.

</details>

## 21. A malformed response raises during citation validation. Which lifecycle action must still occur?

<details><summary>Reveal worked answer and rubric</summary>

The request-owned sequence/cache must reset or release in finally, then the error remains an error. Do not fabricate a refusal. A begin failure on another active session needs ownership-aware handling rather than blindly resetting someone else's state.

**Rubric:** Trace try/finally and distinguish response validation from semantic support.

</details>

## 22. The loopback demo returns 200 and “3 hours.” What evidence of an LLM application does that provide?

<details><summary>Reveal worked answer and rubric</summary>

Only the tested HTTP fixture boundary. Its response states backend=fixture and model_executed=false. It proves parsing/response/error/shutdown behavior for that named fixture, not arbitrary document QA, Gemma loading, browser UX or model quality.

**Rubric:** Identify exactly what ran and the missing model/application evidence.

</details>

## 23. Q8 passes quality but exceeds memory; Q4 fits memory but misses the frozen quality threshold. Which wins?

<details><summary>Reveal worked answer and rubric</summary>

Neither is feasible, so choose none. A smaller context, another model/placement or revised threshold creates a new decision/protocol version; it does not retroactively pass the original criteria. Final labels stay unopened during that revision.

**Rubric:** Apply all constraints jointly and keep development selection separate from final assessment.

</details>

## 24. All teaching examples pass. Which real implementation phase is complete?

<details><summary>Reveal worked answer and rubric</summary>

No model phase is completed by those checks. They verify authored arithmetic/contracts. The existing scaffold is runnable; actual compatibility, frozen data, model artifacts/results, context study and saved application remain pending. Begin phase 00 and attach real receipts before advancing its state.

**Rubric:** Separate software/reference checks, model results, learner explanation and hosted execution.

</details>

## Whiteboard trace

Draw the exact route for one question: canonical bytes → passage spans → split membership → template/token budget → selected artifact/cache → next-token output → response validation → support score → timing/memory receipt → immutable saved generation. At each arrow name one failure and the retained evidence that distinguishes it. Then change one factor—weight format, placement, calibration text or cache type—and state which comparison remains attributable.
