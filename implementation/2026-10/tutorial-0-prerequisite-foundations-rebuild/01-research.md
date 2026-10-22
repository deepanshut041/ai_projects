# Prerequisite coverage research

## Written maths source pass — 2026-10-12

Nielsen's *Neural Networks and Deep Learning* (2015; site dated December 2019), chapters 1–3, is the requested written reference. Chapter 1 motivates learned parameters and gradient descent; chapter 2 expands reverse chain-rule dependencies and shared-weight contributions; chapter 3 derives the sigmoid/cross-entropy output cancellation. Direct HTML retrieval returned 502; the source-domain indexed chapter excerpts were inspected through web search on 2026-10-12. Record that access limit rather than claiming a fresh full-book download. D2L 1.0.3 §5.3 was read directly: forward intermediates, computation graph, reverse order and activation derivative. Keep its column-oriented/no-bias example distinct from our row-oriented/bias-inclusive Tanh fixture. No book prose, diagrams, exercise wording or Python 2 implementation is imported.

Original additions map into existing lessons:

| Starting result | Next problem | Maths/example | Application and independent check | Next dependency |
| --- | --- | --- | --- | --- |
| One row of XW+b, Lesson 3 | Why is a bias needed? | Zero input and an affine offset; row/column convention | Existing rectangular linear fixture; changed bias | Shared-parameter derivatives |
| Scalar chain rule, Lesson 4 | One weight has several paths to the loss | `(2w-1)^2+(3w-1)^2`, w=1: two contributions 4+12=16 | Forward finite differences; small-step Taylor reasoning | Reverse pass through layers |
| Softmax/NLL, Lesson 7 | Why use this loss for a class probability? | Conditional likelihood product → summed NLL; Bernoulli cross-entropy derivative p-y | p=.01,y=1 gives -0.99 versus half-squared probability loss -0.009801 | Logit signal feeding the hidden layer |
| Tanh/affine modules, Lesson 8 | Derive every parameter gradient | A controlled 2→2→2 fixture: hidden [0,.5], logits [0,ln3], target1 | Hand values, every-coordinate finite differences, Torch float64; unequal 2→3→3 fixture too | Same chain applied to the 42-parameter trainer |
| Opposite-sign task | Why one affine boundary cannot fit it | Sum the two positive and two nonpositive score-gap inequalities | Contradiction on the four fixed corners | Need for a nonlinear hidden feature map |
| Each lesson's worked result | Remember and transfer the idea | One topic-specific prompt, equation, own example and changed-case question | Editable blank Markdown/notebook; no invented learner answers | Next existing lesson |

The course remains eleven lessons with optional source reading beside original inline explanations. Notes workbook and study notebook aid the same route; the Colab training notebook and learner lab remain separate. Protected baseline: 133 later-course files and 49 tracked/nonignored child files with exact Git head/status, plus the stable lesson paths and pre-existing tutorial references.

Updated: 2026-10-12. Primary Python3.12 and PyTorch2.14 API sources inspected; learner child state and both later courses verified locally before editing. Their versions and exact source hashes are preserved in before-rebuild.json.

## Teaching boundary

Tutorial 0 now prepares the numerical/programming/experiment foundations before full attention, translation and RoBERTa in Course 1, then Small GPT in Course 2. Eleven existing paths/IDs remain stable; advanced architecture teaching moves out of this active prerequisite route. A verified recovery ZIP retains the previous complete teaching package and its dated results. Original code files stay preserved.

## Prerequisites at first use

Prediction precedes setup. Python tools2 prepare functions, containers/imports. Linear maps/numerical notation3 preserve the exact existing learner linear(rows,weights,bias) API and stop afterA. Stable exp/log/NLL precede gradients4. Text5 separates characters/bytes/IDs/pieces and original spans. Batches6 state axes/splits/padding/targets. Probability/weighted information7 prepares normalized rows, conditional branches, basic period/distribution functions. Modules8 separate parameters/buffers/mode and compose a small classifier. Training9 demonstrates honest counts/heldout diagnosis; recovery10 and source/Colab preparation11 preserve exact experiment state.

## Reference design and first checks

Own synthetic nonlinear classification: class1 iff two features have opposite signs. Training grid64points and distinct heldout grid36points; no realworld quality claim. Actual classifier is Linear2→8→Tanh→Linear8→2,42parameters168FP32parameterbytes, rawlogit output. Tanh is an explicit smooth nonlinearity rather than a source-compatibility requirement. CPUFP32/trainer/state identities are declared. Author evidence and learner/hosted gates remain separate.

Ten text checks already pass: exact UTF8, invalid/truncated bytes, original halfopen coordinates/NFC changes, frozen vocabulary/unknowns/artifact identity and document-bounded merges. Other author groups are testing arithmetic/autograd/tensors/modules/training/recovery/export while drafting. Final sources/results/reader/preservation proof will be in 06-review.md.

Latest scope correction: user is advanced in Python. Omit Python syntax/collections/import/function primers, retain PyTorch runtime/tensor/module concepts and mathematical foundations. Lesson 2, introductory prose, title/native practice and prerequisite map updated accordingly.
