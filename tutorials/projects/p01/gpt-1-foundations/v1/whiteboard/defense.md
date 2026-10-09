# GPT-1 foundations: whiteboard story

Boundary: reference-design study. No complete local GPT implementation or training exists yet. Primary sources are linked in lesson.json; toy examples have their own execution evidence.

In two minutes: draw IDs -> token plus position embeddings -> repeated causal decoder blocks -> vocabulary scores -> shifted loss. Expand a block into attention, add and normalize, FFN, add and normalize. Write [B,T,D], [B,H,T,D/H] and [B,T,V]. Draw the triangular visibility grid and calculate the prefix averages 2, 3 and 14/3.

Explain two failures: an unmasked future changes an earlier output; copied rather than shifted targets change the task. Distinguish residual-stream width from FFN intermediate width. Describe the missing model checks: finite/reference gradients, full causality, generation, training and save/resume.

Finally draw weights, K/V, workspace and training state as separate memory categories. Derive the ordinary MHA cache formula; state that GQA/compression changes it. SSD capacity alone supplies no speed claim. A performance improvement belongs to a focused Research protocol after a measured baseline.

Do not claim original author intent without recorded evidence. Do not score understanding from reading this story. At the actual implementation milestone replace reference-only statements with inspected owner/revision/test results.
