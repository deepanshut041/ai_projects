# Attention tutorial: teaching route and acceptance

Updated: 2026-10-23, America/Toronto. The user confirmed setup and implemented attention now, with later data/model/training teaching added as those parts are implemented. No model implementation, training, paid media or remote publication is in scope. Lead: workspace developer-learning-tutorials; supporting planning, visual-learning and source-backed whiteboard guidance.

The course starts with an inspectable weighted result, then connects the arithmetic to the existing locked project and actual attention implementation. Twelve lessons explain the implemented component without inventing translation results. Canonical sources live in `docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1/`. Project-local phase/implementation records expose the material through the existing grouped Projects reader. Python implementation, existing notebook, captions and Bike content remain unchanged.

Source audit: final reusable attention is implemented; tokenizer/data/model/trainer/checkpoint/config and later top-level entry points are comment-only. Existing 35 CPU checks and the demo passed in this conversation. The notebook has 37 saved executed cells and 28 saved figures; notebook execution has not been repeated for this documentation request.

| Previous result | Next problem | Maths and idea | Source | Observable check | Next connection |
| --- | --- | --- | --- | --- | --- |
| Two values | Pick useful information | Weighted sum, normalized weights | scaled_dot_product_attention | [4,6] from equal weights | Make weights depend on a query |
| First numerical result | Reproduce it | Environment, imports, component boundaries | pyproject/lock/demo | 35 checks and PASS | Read the tensors |
| Vectors | Many queries/keys | Axes, dot products, transpose, batched contraction | _heads and low-level attention | Rectangular matrix known answer | Normalize scores |
| Scores | Comparable positive weights | Exponential, softmax, scale and saturation | scores/weights | Zero/equal/shifted scores | Learn representations |
| Raw input | Learn query/key/value roles | Affine maps, bias, orientation | projection_example/SingleHeadAttention | Three-token Q/K/V and context | Independent heads |
| One head | Multiple retrieval views | Split/transpose/merge/output projection | MultiHeadAttention/Attention | Loop and vectorized parity | Mask invalid keys |
| Full attention | Padded batches | Boolean broadcasting, blocked rows | make_attention_mask | Padded perturbation isolation | Restrict future access |
| Padding | Decoder future leakage | Lower triangle, intersections | causal path | Future perturbation isolation | Retrieve from another sequence |
| Self-attention | Source/target lengths differ | Rectangular query-key matrix | CrossAttention/Attention | B=2,S=5,T=3 output | Reusable API |
| Components | Choose explicit/efficient path | Dropout semantics and public contract | _context/final Attention | SDPA parity, 360 parameters | Backward path |
| Forward result | Can parameters learn? | Softmax derivative, chain rule, finite differences | autograd and parity tests | Nonzero value/query gradients | Interpret evidence |
| Checks | Decide next action | Independent reference, boundaries and unknowns | tests/placeholders | Evidence matrix and handoff | Later data phase |

Maths first-use coverage: vector coordinates and sums in lesson 1; tensor axes/products in 3; exponentials/probabilities/root scaling in 4; affine maps/transposes in 5; reshape/axis permutation in 6; boolean logic/broadcasting in 7; temporal constraints in 8; derivatives/chain rule in 11. All use tiny labelled synthetic inputs rather than claimed linguistic alignments.

Acceptance: complete worked text, source snapshots and diagram provenance; independently executed important numbers and changed-condition checks; unchanged model/Bike files; rendered manifest with linked project records and revealable practice; source downloads; root/prefixed desktop/mobile equations/diagrams/navigation/containment; standard reader tests/build/browser checks; final continuity/maths/implementation/readiness review. Source videos remain references; no new narrated media is generated. Learner understanding and hosted recovery remain unassessed.
