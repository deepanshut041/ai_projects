# What each comparison can explain

| Comparison | Change | Keep fixed | Can establish | Cannot establish alone |
| --- | --- | --- | --- | --- |
| Converted float vs source float | Container/runtime interpretation | Source weights, prompt/template/decoding contract | Bounded conversion agreement | Full task preservation or same-runtime speed |
| Floating vs Q8 vs Q4 | Weight artifact/recipe | Source, prompt IDs, runtime/placement/cache, budgets, decoding | Same-runtime weight tradeoff where feasible | Benefit of another placement/backend |
| General vs domain calibration | Calibration text distribution | Token/sample budget, method/group/precision/runtime | Effect under that calibration protocol | Universal best corpus |
| Basic vs supported advanced method | One method/recipe | Source/data/budget/runtime factors if possible | Applied method tradeoff with actual coverage | A causal method claim when runtime also changes |
| Group/selective precision | One representation choice | Method/calibration/task/runtime | Budgeted precision allocation | Additive toy loss predicts full-model support |
| Cache types/policies | Verified cache behavior | Weight hash, prompts within length, decoding/placement | Cache-quality/context-memory effect | Weight-method advantage |
| Context length/evidence placement | Declared prompt length/content/position | Model/artifact/cache policy/output allowance | Behavior under those constructed inputs | General long-document reliability |
| CPU vs device or runtime A vs B | Deployment configuration | Task identities and stated output policy | Deployment tradeoff | Quantization-only speedup |
| Before vs after restart | Process and loading lifecycle | Generation/prompt/runtime contract | Saved-artifact recovery/replay | Training/resume state or unrelated hosted execution |

Every row needs actual identities and receipts before it becomes a result. Current course numbers are authored fixtures. For ambiguous answers use the frozen claim/evidence rubric and retain reviewer disagreements. Keep the floating reference even when a smaller artifact wins a practical budget; it remains a quality anchor.
