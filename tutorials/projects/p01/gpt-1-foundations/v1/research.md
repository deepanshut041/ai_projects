# Authoring evidence and limits

2026-10-09: Rechecked GPT-1 paper sections 3/4.1 and pinned train.py revision a69b5c43b0452462890bca8ff92fb75dee9290cf. Inspected block normalization order, embeddings, head splitting, shifted LM targets and shared output weights. No original training/provider operation was replayed.

The 2x8x64 worked dimensions, scalar zero-logit attention and cache calculation are independently authored teaching derivations, not published benchmark results. The intended tiny project uses byte/special IDs, an explicit tokenizer simplification. The C++ and Python examples have identical declared JSON expectations.

Six semantic diagrams cover architecture, block order, attention, target shift, training workflow and memory/access. Chapters separate source statements, illustrative calculations and planned model checks. Whiteboard preparation is reference-only. Project progress remains P01-M1 not-started.

User correction: all shared Projects documentation is consolidated under ai_projects/docs; the former project_docs directory is gone. Tutorials remain Git-eligible under this same owner. Source Dashboard ignore/private behavior is unchanged.

2026-10-09 adaptive improvement: inspected amosblomqvist/learn at 7cfd8942f82ab9476e63572387e1fe9bcea5082c for teaching design. Added an original proposed prerequisite route, concept-dependency board, visible-prefix entry practice and zero-score misconception/transfer practice. Motivated causality and target shift from the explicit task; retained conditions and source-specific architecture limits. The upstream repository informs pedagogy, not GPT model claims or proven learning outcomes. Exact provenance and comparison live in implementation/2026-10/learn-inspired-adaptive-tutorial-improvements/.

Seven semantic diagrams and six practice records now include the learning route. Existing toy examples/model-source claims are preserved. No human diagnostic answer has been collected; learner knowledge/mastery stays unassessed and P01-M1 remains not-started.
