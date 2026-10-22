# P01 review questions

Attempt these before reopening the worked answers embedded in the lessons. The original practice IDs keep the answer/rubric beside their teaching context.

1. For bytes of ok=[111,107], write x and y including BOS/EOS. Which token does x position zero predict? (lesson objective: next-token; practice `shift`)
2. Correct-token probabilities are 1/2, 1/4 and 1/8. Compute sequence probability, mean NLL and perplexity. (lesson objective: next-token; practice `perplexity`)
3. If D=96 and H=4, what is head width? Why does D=63,H=4 fail? (lesson objective: workspace; practice `heads`)
4. You train BPE on all twelve chapters before splitting. Why is a frozen final artifact still insufficient? (lesson objective: text; practice `tokenizer`)
5. For BOS,a,b,c,d,EOS with T=3, give both x/y windows and count supervised labels. (lesson objective: windows; practice `windows`)
6. Weights and optimizer restore correctly, but the next random window differs. What is missing? (lesson objective: windows; practice `batch-rng`)
7. With all scores zero and values 2,4,8, what is row 1 under a causal mask? What if forbidden scores are set to zero? (lesson objective: attention; practice `mask`)
8. Outputs match SDPA but Q gradients differ. What must you check before widening the tolerance? (lesson objective: attention; practice `gradient-check`)
9. Does a repeated token at positions 2 and 5 share both embedding rows? (lesson objective: model; practice `positions`)
10. Zero both sublayer outputs. What remains of a pre-norm block, and a post-norm block? (lesson objective: model; practice `prenorm`)
11. Why can GELU(-1) be negative while ReLU(-1)=0? (lesson objective: model; practice `gelu`)
12. For logits [0,ln(3)] and correct class 1, compute the two logit derivatives. (lesson objective: learning; practice `loss-gradient`)
13. Microbatch A has 2 tokens with NLL sum 2; B has 1 with sum 4. How should backward be scaled? (lesson objective: learning; practice `accumulation`)
14. A 300-update run pauses at 150. What schedule step follows, and can --updates 150 change total to 150? (lesson objective: learning; practice `schedule`)
15. Training loss falls but validation rises. Is greedy sample readability enough to call the model improved? (lesson objective: generation; practice `evaluation`)
16. For logits [0,ln(3)], compare probability of the second token at temperatures 1 and 0.5. (lesson objective: generation; practice `temperature`)
17. A generated prefix is longer than context 64. What does this baseline do, and why is that relevant to a cache? (lesson objective: generation; practice `request-reset`)
18. The newest directory has state.pt but no COMPLETE. An older completed generation verifies. What should resume do? (lesson objective: checkpoint; practice `recovery`)
19. Why restore RNG after constructing and loading the model? (lesson objective: trainer; practice `state-order`)
20. You load only weights at update 150 and create a new optimizer. Is that a resumed run? (lesson objective: trainer; practice `resume-vs-warm-start`)
21. A local /tmp save/reload and a successful Drive mount passed. What remains before Colab recovery is established? (lesson objective: colab; practice `hosted-proof`)
22. What changes in the parameter count if you untie a bias-free vocabulary head? (lesson objective: handoff; practice `parameter-count`)
23. Doubling context from 32 to 64 does what to explicit attention-score storage? Does step time necessarily quadruple? (lesson objective: handoff; practice `profile`)
24. A one-token query follows five cached keys. Why can a naive triangular 1x6 mask be wrong? (lesson objective: handoff; practice `cache`)

After each answer, change one input or assumption and predict the result. Missing explanations point back to the matching lesson; a memorized sentence does not replace an implementation or run.
