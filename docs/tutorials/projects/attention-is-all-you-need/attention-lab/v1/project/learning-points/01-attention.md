# Reconstruct the attention design from the translation task

A translation model receives source tokens and predicts each next target token from that source and its available target prefix. Token IDs select embedding rows; positional input supplies order. Encoder and decoder layers transform those rows into representations that support vocabulary predictions.

Applying f(x_i) independently to each row cannot make output i depend on a changed neighboring input. The missing capability is communication between positions. A common average communicates one shared summary, while attention gives each receiver its own input-dependent mixture of available source values. This explains the operation before naming its parts.

The query is derived from the receiving representation. Each available representation supplies a key for comparison and a paired value for contribution. Their separate projection matrices can adapt selection and content independently. The matrices are shared model parameters; the calculated Q/K/V, scores and attention weights are activations for the current input. Useful comparison features must be learned rather than assigned by names.

In the fixture, query [1,0] compares with keys [1,0], [0,1], [1,1], [0,0]. The scaled scores normalize to [0.334881,0.165119,0.334881,0.165119]. Applying them to [2,0], [0,2], [2,2], [0,0] gives [1.339523,1.000000]. That result is a gathered feature row. Its two numbers are not a token prediction, and the labels on the rows do not give their coordinates linguistic meaning.

Stacked queries preserve one result per receiver. Score cell (i,j) is a comparison from receiver i to source j. Row-wise softmax gives each receiver a distribution; multiplying by V sums out the source axis. Batch examples remain independent. Paired permutations of source keys and values preserve a fixed receiver’s result; unrestricted self-attention needs positional input to represent order.

One head uses one weight per source row across all its value features. Several heads let projected feature groups gather independently weighted mixtures. Every head still sees all permitted positions. Restoring the position axis before joining heads is essential to preserve row ownership.

Masks are information boundaries. Padding removes storage-only source rows before normalization; even a zero padded value otherwise consumes denominator weight. Decoder causality blocks future reference tokens that would reveal the next-token label during training. A causal mask requires the correct shifted input/label contract. Cross-attention instead lets target receivers read all valid encoded source rows, whose indices do not represent future target time.

Reference-token cross-entropy supplies gradients through the vocabulary projection and decoder into attention. The course’s explicit squared-context probe isolates one part of that derivative path. Its falling loss explains a local update, rather than establishing a trained translator.

The actual class packages projection, head layout, allowed masks, explicit/library computation and output mapping. The surrounding Transformer still needs positional input, residual/norm/FFN sublayers and vocabulary prediction. Return to [the execution trace](lessons/10-reusable-attention.md) for exact source behavior and [the architecture](lessons/12-evidence-and-next-phase.md) for component boundaries.
