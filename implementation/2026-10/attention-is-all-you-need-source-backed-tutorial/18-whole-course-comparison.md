# Whole-course teaching review — 26 October 2026

The learner rejected the explanation and videos, then explicitly corrected a paragraph-level response: compare the entire tutorial with good human-authored attention tutorials before deciding how to repair it. This review covers all twelve chapters, the complete lesson, learning points, comparisons, practice, implementation guide, actual component and rendered opening. It supersedes the current teaching/delivery recommendation, without erasing historical approvals or delivery receipts.

## What was compared

Read Jay Alammar's [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) from the translation black box through embeddings, scalar attention, matrices, heads, positions, decoder, vocabulary output and training. Its strength is successive zooming: the reader knows where a component sits before inspecting it. Its linguistic examples and some broad statements are not copied; attention need not assign the largest weight to the current token.

Read Harvard NLP's [Annotated Transformer, 2018 version](https://nlp.seas.harvard.edu/2018/04/03/attention.html): architecture, encoder/decoder, attention, heads, feed-forward, embeddings/positions, masking, training and decoding. The current replacement URL failed fetching, so the inspected version is explicitly historical. Its strength is that each implementation belongs to an already located architectural responsibility. Do not copy its old package commands or its fully masked-row behavior into this project.

Read the authors' [D2L Q/K/V chapter](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html) and [Bahdanau attention chapter](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html). They distinguish selection from content and develop attention as a remedy for a specific encoder/decoder limitation. D2L's abstract database analogy is not required for this learner. Avoid its unqualified statement that softmax gradients never vanish.

Check historical motivation against [Bahdanau, Cho and Bengio](https://arxiv.org/abs/1409.0473), and architecture/math against [Attention Is All You Need](https://arxiv.org/html/1706.03762v7), especially sections 1–4. Separate earlier recurrent encoder/decoder attention from Transformer self-attention. The comparison is my editorial assessment, not a measured educational outcome.

## The central defect

The old route is primarily an arithmetic ladder. It starts from an asserted need for context, immediately abandons the text for a supplied numerical fixture, and then explains operations on that fixture. Most local calculations are sound. What is missing is the explanatory connection between task, available information, an inadequate design, the required capability, the chosen operation and the meaning of its result. A correct executable block cannot supply those missing connections.

The heading “start with the problem” does not establish a problem. The important question is whether the reader has actually been shown the task and data before being asked to accept the proposed solution.

## Chapter-by-chapter audit and replacement requirements

| Chapter | Existing teaching issue | Required explanation |
| --- | --- | --- |
| 1. Weighted information | Word ambiguity appears before a task; input representations and their origin are missing; hand-set weights become the apparent subject. | Establish translation input/output, token IDs, embeddings and row ownership. Demonstrate why a row-local function cannot use a changed neighboring row. Compare common pooling and per-position retrieval before the weighted sum. |
| 2. Scores and softmax | Q/K/V are names for supplied arrays; the actual input-to-query connection is deferred. | Derive comparison and payload roles from one input matrix, show which row supplies each factor, and trace score → scale → normalize → contribution → output. Explain what training must learn. |
| 3. Tensors and products | Correct dimensions lack a view of who receives what; positional information is absent. | Explain each score cell and output row, then demonstrate row-wise normalization, batch isolation and permutation equivariance without positions. |
| 4. Learned projections | Identity maps verify arithmetic while hiding what separate learned maps can change. | Distinguish shared parameters from per-input activations and attention weights. Explain asymmetric reading, bilinear compatibility and how a parameter change affects many rows. |
| 5. Heads | Layout dominates; the limitation of one shared attention distribution is brief. | Start with the one-distribution constraint, derive independently weighted feature groups, then explain the split/transpose/join and output map with the existing numbers. |
| 6. Padding | The mask is introduced without showing the variable-length batch that requires it. | Show stored rows versus real tokens, denominator contamination even with a zero value, pre-softmax exclusion, empty-row policy and query/loss distinction. |
| 7. Causality | A new text example interrupts the running task; training versus generation is underdeveloped. | Use the same translation task to show shifted decoder inputs and desired tokens, the exact cheating path, allowed pairs and an independent future-perturbation check. |
| 8. Cross-attention | Translation becomes concrete only here; two arbitrary target rows lack architectural ownership. | Show decoder-prefix state, encoder memory and output-to-vocabulary continuation. Explain rectangular geometry and why source indices have no target-time ordering. |
| 9. Learning | Context MSE may be mistaken for how attention is actually supervised; full task loss is distant. | First trace translation labels → vocabulary loss → decoder/attention parameters. Then label the isolated MSE probe as a magnified derivative calculation and show the missing chain. |
| 10. Project source | A large dependency-closed block substitutes for an execution walkthrough. | Follow one actual call through named helpers and tensor shapes, validation/masks/branches/output. Keep complete exact source available after the walkthrough. |
| 11. Architecture/handoff | The system map and meaning of the title arrive after the mechanism. | Return to the architecture established at the beginning; distinguish three attention uses and surrounding residual/norm/FFN/position/output work. State exact implemented scope once. |
| Setup reference | Packaging/demo inventory overwhelms the simple run path. | Put the minimal reproducible commands first, then ownership/troubleshooting/Colab details. Keep it a reference. |

## Cross-cutting issues

- Figures exist on disk, but the current conceptual pages mostly do not show them. Add focused diagrams tied to the current input, computation and axes, with adjacent readable explanations.
- Repeated identical numerical answers establish continuity of the fixture, not understanding of the task. Every chapter must interpret the result and explain what remains unresolved.
- Repeated “not a translator / not mastery” delivery prose disrupts teaching. Keep one early component-scope statement and a precise final readiness record, with only locally necessary distinctions in between.
- Practice mostly checks arithmetic or restates the just-presented definition. Add design/perturbation questions: what input change can influence which output, and through which path?
- A full code block remains necessary for runnable completeness, but execution should be explained before it. Avoid introducing another hidden-helper problem while fixing readability.
- The complete chapter, project notes, overview, review pages and current checkpoints must agree. Historical delivery records retain their dates and are marked superseded for present recommendations.

## Rewrite contract

Use one real task, English-to-French translation, throughout. A single short sentence establishes actual source/prefix/label ownership; no unrelated ambiguity, search-engine, pronoun or shopping analogy is needed. The retained three-row numerical fixture is explicitly an inspectable stand-in for one layer's activations, never claimed to be trained embeddings or a computed translation.

The opening must establish the system and the missing communication path before defining attention. Every subsequent operation must answer a design question from that system and expose inputs, computation and output. Preserve stable URLs, exact source helpers, ten complete executable programs and checked numerical values. Keep full source after an execution walkthrough. Rewrite review/practice material to assess causal reasoning.

Remove all current Attention video bindings from the tutorial and its project notes/library. Preserve recoverable original media and registry snapshots. No video production or reattachment is authorized by this rejection. No source implementation, training, learner checkpoint, scientific result, commit or publication is part of this editing work.

## Recovery and verification

Pre-edit course, private notes and full registry are saved under `.ai/attention-course-review/2026-10-26/before/`. `protected.json` pins actual Attention/Bike source, tests and Attention notebook; `programs-before.json` pins all ten standalone Python blocks. Verify complete-fence execution, independent scalar/gradient checks, actual-source AST parity, declared references and publication. Inspect every revised chapter and authored figure in the browser at desktop/mobile widths; check reveal, source/navigation/download, complete route and zero Attention players. Technical checks support factual and artifact correctness; teaching acceptance remains for the learner.
