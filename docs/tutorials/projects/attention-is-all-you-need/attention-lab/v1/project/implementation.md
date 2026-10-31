# Read the model problem, then trace the computation

Start with [the whole picture](lessons/01-weighted-information.md): the translation process, its two jobs and attention's place in them. The opening map shows how each pass adds detail. Then zoom into one source-side information-sharing calculation and build on it through calculated weights, all-position matrices, learned maps, heads, availability boundaries, target/source relationships, task gradients and the actual class. The final chapter returns those understood operations to the Transformer.

The conceptual route is chapters 1–11; the last setup page is a run reference. Each numerical program is complete and downloadable. The four-row fixture is deliberately small so every contribution can be inspected; its row labels track ownership without pretending the coordinates were learned from text.

The [whole-course comparison](project/teaching-review.md) explains the structural problems found in the previous version and how this route addresses them. Its assessment covers all chapters, not just the opening paragraph.

All Attention companions were removed from this course and its project understanding notes on 26 October 2026. Recoverable original media remains local. No replacement video production is part of this revision.

The existing implementation remains the attention component. Tokenizer/data, complete model, training, decoding/evaluation and hosted recovery remain future work. [Readiness](project/readiness.md) records that boundary.
