# Transfer review: head layout changes

Use the two-head numerical fixture to answer these before checking the code.

## Suppressing one head at the output

If the rows of the row-vector output matrix corresponding to Head 2 are zero, do Head 2's attention weights have to change?

<details><summary>Worked reasoning</summary>

No. The output matrix operates after the head distributions and value mixtures are computed. It can suppress Head 2's contribution without changing that head's current weights. A later learning update could change upstream parameters, but that is an additional operation rather than a consequence of this forward assignment.

</details>

## Equal axis lengths can hide a bug

Suppose the number of heads equals the number of tokens. A mistaken head-by-head product now has the same numerical shape as the intended token-by-token product. What check can still expose the error?

<details><summary>Worked reasoning</summary>

Compare values against an explicit per-head token-attention reference using the same parameters and nontrivial inputs. Shape checks alone cannot distinguish axes that happen to have equal lengths. Output-order checks, gradient comparisons and a future-token intervention add independent evidence.

</details>

## A faster packing versus fewer KV heads

An implementation removes its Python loop and preserves the same matrices and outputs. Has it become grouped-query attention?

<details><summary>Worked reasoning</summary>

No. It has changed the execution layout. GQA requires fewer key/value heads than query heads and an explicit sharing relation. Packing independent MHA heads preserves the original head counts and parameter relationships.

</details>
