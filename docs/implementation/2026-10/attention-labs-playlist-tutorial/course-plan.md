# Course plan

Audience: Python/C++ engineer learning attention deeply; no Python syntax primer. Purpose: derive the small mechanisms, inspect intermediate arrays, understand what changes with masks/projections/heads, and distinguish component learning from a full trained translation model.

| Starting point | Source lecture | New understanding and output | Folder |
| --- | --- | --- | --- |
| Need contextual token information | 1 | Locate scoring, connectivity, head/KV layout, efficiency, position and input-domain choices without pretending the categories are exclusive | attention-foundations |
| Whole translation task | 2 | Follow one recurrent encoder and decoder and quantify a fixed-summary limitation | attention-foundations |
| Retain source states | 3 | Derive a query-dependent context with additive scores, stable softmax, alignment and output stopping | attention-foundations |
| Learned additive scorer | 4 | Compare dot/scaled scores, norms, variance assumptions, temperature and softmax gradients | attention-foundations |
| Weighted retrieval | 5 | Compute every row of a small self-attention matrix; connect source/query axes, permutation and limitations | attention-foundations |
| Raw input geometry | 6 | Separate Q/K retrieval from V content, trainable projections and one checked gradient update | attention-foundations |
| Full visibility | 7 | Derive causal masking before normalization, shifted targets, safe padding and dropout expectation | attention-foundations |
| A complete single head | 8 | Derive independent heads and output projection, then prove loop/packed equivalence with all tensor axes; future GQA remains planned | attention-heads |

Illustrative numeric fixtures are independently authored; they are not the lecturer's embeddings, learned semantics or the existing accepted Attention course fixture. Every meaningful formula is introduced with symbols, a small calculation, code and changed-condition check. Each included lecture gets its own source video and original slide deck mapping. Lecture 9 has no chapter/deck/embed. No generated narration is requested.
