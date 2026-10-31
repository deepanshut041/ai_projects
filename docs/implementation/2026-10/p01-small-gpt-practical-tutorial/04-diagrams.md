# Teaching boards

The editable semantic definitions are in the P01 lesson.json; generated SVG/JSON remain in its diagrams/ directory. Each is placed beside its teaching explanation.

| Board | Relationship being taught |
| --- | --- |
| prefix | Known prefix -> causal scores -> shifted-label comparison; labels do not enter their own forward call |
| data-flow | Raw bytes -> chapter split; only train fits tokenizer; all partitions use frozen encoding |
| attention | Projection/head geometry -> scaled/masked scores -> value mixture -> merged channels |
| model | IDs -> token+position embeddings -> blocks -> final norm -> tied vocabulary scores |
| prenorm | Direct residual bypass plus normalization/sublayer branch meeting in addition |
| recovery | Completed update -> local state -> verified persistent generation -> completion marker |
| memory | Distinct weight/Adam/activation/optional-cache lifetimes; a grid, not an execution sequence |

Scenarios independently change causal visibility, unequal reduction, temperature and checkpoint validity. Read each selected case before treating its control as verified; a rendered graph is not automatically a correct graph.
