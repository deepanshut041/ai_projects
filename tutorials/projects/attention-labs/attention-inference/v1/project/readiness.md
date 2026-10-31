# Cache readiness

Explain the full prompt-to-next-token loop, why old causal states remain unchanged, why old queries need not be retained, and how a new query retrieves from all permitted stored values. Recalculate one score and cache byte count. Draw absolute positions for a multi-token append.

Technical readiness requires both executable references to pass and the cached chunks to match the full causal result. The implementation does not support padding/ragged caches, eviction, quantization or production prefix sharing. It makes no speed claim and is not a complete model. Learner understanding and actual Colab execution remain unassessed.
