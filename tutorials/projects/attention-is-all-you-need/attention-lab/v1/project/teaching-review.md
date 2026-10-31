# Why the whole tutorial was rebuilt

On 26 October 2026, the learner rejected the tutorial and explicitly asked for an entire-course comparison with good human-authored attention tutorials. The review read all twelve chapters, their code, practice, supporting pages and the rendered opening.

[Jay Alammar’s Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) establishes the translation system and successively opens its components. [Harvard NLP’s Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), inspected in its historical 2018 version, connects architectural responsibilities to implementation. [D2L’s attention chapters](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) motivate retrieval through a specific encoder/decoder limitation. These are useful teaching patterns, not text or figures copied into this course.

The previous route mostly formed an arithmetic ladder: supplied values, supplied query, matrices, projections, heads and masks. Its local numbers could be correct while the reader still lacked the task, input ownership, inadequate design and reason for each operation. The full model context appeared too late. A large exact source block did not itself explain execution, and figures available on disk were mostly absent from the pages.

The replacement establishes translation and token-to-vector input first, demonstrates the missing dependency of independent row processing, then derives input-dependent gathering. It introduces Q/K/V as transformations of owned input rows, interprets matrix axes, distinguishes parameter state from activations, motivates heads and information boundaries, connects token loss to attention, and follows an actual class call before the full source.

In the subsequent review, the learner said these detailed explanations were good and requested an overview-to-depth progression. The opening now shows the whole translation process before opening a component, introduces encoder/decoder names after explaining their jobs, locates attention in plain language and maps the successive zooms through the course. The accepted detailed calculations and programs are retained.

The numerical fixture and exact programs remain useful as inspectable arithmetic. They are explicitly separated from learned linguistic representations. New diagrams show actual data paths and axes. Practice asks about influences, pairing, leakage and supervision rather than merely repeating names. Setup remains a reference.

All Attention video attachments were removed. Prior delivery checks remain historical records and do not establish that the teaching was effective. The revised text’s teaching acceptance remains for the learner to judge.
