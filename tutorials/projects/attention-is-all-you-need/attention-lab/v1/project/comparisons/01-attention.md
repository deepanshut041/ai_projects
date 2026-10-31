# Compare designs by their information paths

Use these comparisons after [the main route](lessons/01-weighted-information.md). The important difference is which inputs can influence which outputs, through which calculation.

| Decision | What it permits or constrains | Consequence to check |
| --- | --- | --- |
| Independent row transform | Output i reads only input i | A neighbor-only change cannot alter that row |
| Common average | Every receiver gets the same mixture | Communication exists, but the mixture is not receiver-dependent |
| Query-dependent attention | Each receiver derives weights against available keys | A changed query can redistribute the same available values |
| Recurrent communication | State passes sequentially through positions | Context is possible, with a position-dependent computation chain |
| Separate Q/K/V maps | Selection and contributed content can differ | Halve only W_V: context halves with identical weights |
| Multiple heads | Projected feature groups have separate distributions | All heads keep every allowed position; joining must preserve receivers |
| Source padding mask | Storage-only key rows are absent from normalization | A padded-row perturbation cannot affect real receivers |
| Target causal mask | Future target inputs cannot reveal labels | Test future perturbation while keeping those keys real |
| Cross-attention | Target states retrieve from a separate source memory | Rectangular weights; source mask length follows memory |
| Explicit/library execution | Two implementations of the same formula | Match exact state and evaluation inputs before comparing |

A mask’s appearance does not establish isolation, and matching tensor shapes do not establish correct key/value pairing. Perturbations and row ownership make those properties concrete.

Parallel attention still has n² pair comparisons for a length-n sequence. Parameter sharing across positions does not remove activation work, and increasing heads at fixed model width narrows each head. These are engineering tradeoffs, not automatic quality improvements.
