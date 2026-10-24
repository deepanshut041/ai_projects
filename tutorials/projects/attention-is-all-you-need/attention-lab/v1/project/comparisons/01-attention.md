# Compare attention choices through their consequences

| Comparison | Hold fixed | What changes or what to check |
| --- | --- | --- |
| Raw Q=K=V versus learned projections | Input rows and attention formula | Learned affine maps create separate comparison/content spaces |
| One head versus multiple heads | Declared model width and task | Head width/partitions and distributions; no promised head specialization |
| Loop versus vectorized heads | Mapped identical parameters and inputs | Organization of equivalent calculations, checked values/weights |
| Padding versus causality | Input sequence and model | Invalid keys versus future keys; combine with AND |
| Self versus cross-attention | Public dtype/width conventions | Origin of queries and output length; source-aligned masks |
| Explicit versus SDPA | Parameters, inputs, masks, dtype, dropout state | Numerics/gradients within tolerance; hardware speed needs measurements |
| Pre-dropout weights versus used weights | Scores, probability and training state | Inspection weights stay normalized; context uses dropped/rescaled weights |
| Zero context versus zero final output | Blocked row and model | Output projection bias remains |
| Correct component versus successful translation | Exact claim being assessed | Full data/model/training/evaluation evidence still needed |

The comparison is meaningful only when its controlled inputs are declared. Two random initializations are two different functions. A different output is not evidence that vectorization is wrong until the parameter mapping and axis layout match.
