## 11. Put attention back into the Transformer

We began with an English sentence and a decoder that must generate its French translation using the source and an available target prefix. We can now name the three information-sharing operations in that system and trace their outputs.

| Location | Receiver Q | Available K/V | Information boundary |
| --- | --- | --- | --- |
| Encoder self-attention | Source representations entering the layer | The same source representations | All real source tokens are available |
| Decoder self-attention | Target representations entering the layer | The same target representations | Real target-prefix tokens through the current input position |
| Decoder cross-attention | Decoder states after its preceding sublayer | Final encoder memory | All real source tokens, regardless of target index |

The formula is shared, but vector origin and allowed connections determine its role. Describing a matrix only as “attention” without identifying those two facts leaves the important part unexplained.

### The surrounding network still has work to do

In the original 2017 Transformer, token embeddings and positional encodings enter stacked encoder/decoder layers. Encoder layers have self-attention and a position-wise feed-forward network. Decoder layers have causal self-attention, encoder–decoder attention and a position-wise feed-forward network. Sublayers use residual addition followed by LayerNorm in the original post-norm design. The final decoder representations pass through a vocabulary projection to predict the next token. [Paper, sections 3.1–3.5](https://arxiv.org/html/1706.03762v7).

Residual addition preserves a direct path for the receiving row alongside its computed update. The feed-forward network transforms features within each row; it does not replace attention's cross-position communication. Positional encodings supply order information that unrestricted attention alone lacks. LayerNorm controls per-row feature normalization. Those responsibilities make attention useful inside a model rather than turning the weighted-sum function into a complete translator.

The title “Attention Is All You Need” describes the paper's replacement of recurrent/convolutional sequence mixing. It does not mean that embeddings, positions, feed-forward layers, residuals, normalization, output predictions or training objectives disappear.

### What each construction actually resolved

An independent row function had no neighbor-to-output path. A weighted sum created such a path. Query/key projections made the weights depend on current receiving and source representations; the value projection chose what traveled. Matrix multiplication retained one result per receiver. Heads allowed different projected feature groups to use different distributions. Padding and causal masks removed invalid information paths. Cross-attention connected decoder receivers to source memory. A loss supplied gradients to the shared parameters.

These are the reasons for the operations, not just their execution order. To explain a layer, start from its input ownership and allowed information, then follow a score, a normalized contribution and an output row.

### The exact local implementation boundary

The existing project contains the attention module, its demo, CPU checks and a Colab execution notebook. The ten standalone programs in this course make the arithmetic and parameter behavior inspectable. They do not supply a tokenizer, training corpus, complete Transformer, translation trainer or measured translation quality.

The next implementation phase must establish actual source/target tokenization and batches: dataset/split identity, special tokens, input/label shift, source-key validity and target-loss exclusion. After that come the full model, training, decoding, evaluation and persistent recovery. Actual hosted execution remains a separate check.

The conceptual attention route ends here. The following setup page is a run/reference page. The [project return page](project/implementation.md), [learning points](project/learning-points/01-attention.md), [comparison review](project/comparisons/01-attention.md), [optional review questions](practice/explain-attention.md), [readiness](project/readiness.md) and [source/check record](project/reference-runs.md) support a return visit. The [whole-course teaching review](project/teaching-review.md) explains why this route was rebuilt.

The previous Attention videos were removed from this tutorial and its project notes at the learner's request on 26 October 2026. This rewritten explanation is available for review; technical verification does not establish its teaching acceptance or the learner's understanding.

[[practice:handoff-check]]
