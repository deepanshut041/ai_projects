# Comparisons: Follow tree leaves, forests and boosted corrections

The task and information set stay fixed: the original 45 calendar inputs, identity rental target, 2011 training and the frozen 4,357-hour later validation population. Trees add conditional regions; forests average varied regions; boosting adds residual corrections. They are not different optimizers for the same linear weight vector. Linear additive and interaction references, plus the constant and grouped means, make that distinction testable.

| Choice | What changes | Executed evidence | Consequence |
| --- | --- | --- | --- |
| Shallow versus deep bounded tree | Region detail and leaf support | Training MAE 93.70 → 25.83; mean later MAE 140.74 → 100.00 | More capacity helps this sweep, while still losing to groups at 91.57. |
| One tree versus bootstrap forest | Average 64 varied fitted trees | Best family mean MAE 99.75; seed-17 snapshot about 16.1 MB | Small improvement over one tree does not justify replacing the stronger baseline. |
| Forest versus boosting | Independent averaging versus sequential corrections | Boosting 300-stage raw/clipped MAE 104.84/102.22 | The aggregation rule and negative-output behavior differ. |
| Raw versus nonnegative | Output policy after fitting | Tree/forest tie; boosting improves after clipping | A model's reporting policy accompanies its score. |
| Detailed tree versus broad groups | More historical conditioning | Busy forecast 180.75 versus 349.50, target 420 | Lower training error can coexist with worse transfer. |

Each family gets three configurations and seeds 17/29/43, nine main fits. Within a family, choose the configuration by mean validation MAE over those seeds, never by the best seed. Exact ties prefer raw policy, then the declared configuration order. Seed 17 is the representative by prior declaration. Its score is kept separate from the configuration mean. The four deterministic references get one fit each and no newly tuned settings. Equal search counts do not make the model families computationally identical; costs and structural limits remain explicit.

The timing pilot uses 1,024 training rows and chooses no quality setting. Structural caps are 1,024 leaves per component, 64 forest trees and 300 depth-three boosting stages. The five/sixty-second timing guards are post-fit checks, not hard interrupts. All reviewed fits complete, and all final receipts are warning-free. The earlier run with a deprecated boosting criterion argument is retained; removing that inactive argument changed no grid or selection criterion.

Source/environment/data/protocol identities, raw and clipped predictions, tree support, node counts, snapshot bytes, fit and replay times are retained. JSON size is storage, not peak RAM. Replay timing describes the transparent snapshot path, not sklearn native serving latency. Algorithm seeds on one validation population are not independent temporal replications.

Groups/raw remains selected at 91.5729. This is a development reference, not final-test evidence. Deeper error slices, more estimator tuning, alternative ordinal/cyclic tree inputs, observed-weather tasks and serving changes remain outside this implementation. The [understanding notes](lessons/07-trees-and-ensembles.md) explain the calculations and sources in context.

Return to the [worked understanding notes](lessons/07-trees-and-ensembles.md).
