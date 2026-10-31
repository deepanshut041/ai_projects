# Attention Labs reading route

Begin with the seven [Attention Foundations lessons](../tutorials/projects/attention-labs/attention-foundations/v1/lessons/01-context-and-map.md), then the single complete [MHA tutorial](../tutorials/projects/attention-labs/attention-heads/v1/lessons/01-independent-heads.md). Continue in the order below. Categories organize related mechanisms; one complete lesson file owns each mechanism.

| Order | Tutorial | Next problem |
| --- | --- | --- |
|1|[KV cache](../tutorials/projects/attention-labs/attention-inference/v1/lessons/01-kv-cache.md)|Reuse prefixes and preserve absolute positions.|
|2|[MQA](../tutorials/projects/attention-labs/attention-heads/v1/lessons/02-multi-query-attention.md)|Independent queries share one retained KV head.|
|3|[GQA](../tutorials/projects/attention-labs/attention-heads/v1/lessons/03-grouped-query-attention.md)|Choose and verify query-to-KV grouping.|
|4|[RoPE](../tutorials/projects/attention-labs/attention-position/v1/lessons/01-rope.md)|Rotate comparisons using absolute positions.|
|5|[ALiBi](../tutorials/projects/attention-labs/attention-position/v1/lessons/02-alibi.md)|Add head-specific relative-distance penalties.|
|6|[MLA](../tutorials/projects/attention-labs/attention-heads/v1/lessons/04-multi-head-latent-attention.md)|Compress content and retain the positional path.|
|7|[FlashAttention](../tutorials/projects/attention-labs/attention-kernels/v1/lessons/01-flash-attention.md)|Merge tiled dense softmax without full intermediates.|
|8|[Sparse, local, global and sliding-window attention](../tutorials/projects/attention-labs/attention-sparse/v1/lessons/01-sparse-local-global.md)|Define visibility, reachability and safe eviction.|
|9|[Linear attention](../tutorials/projects/attention-labs/attention-linear/v1/lessons/01-linear-attention.md)|Accumulate a normalized positive-kernel state.|
|10|[Performer](../tutorials/projects/attention-labs/attention-linear/v1/lessons/02-performer.md)|Approximate exponential similarities with random features.|
|11|[DeltaNet](../tutorials/projects/attention-labs/attention-linear/v1/lessons/03-deltanet.md)|Correct the association the memory currently predicts.|
|12|[Gated DeltaNet](../tutorials/projects/attention-labs/attention-linear/v1/lessons/04-gated-deltanet.md)|Forget old state before correcting the remaining prediction.|
|13|[PagedAttention](../tutorials/projects/attention-labs/attention-inference/v1/lessons/02-paged-attention.md)|Allocate, share and reclaim KV blocks safely.|
|14|[Ring Attention](../tutorials/projects/attention-labs/attention-distributed/v1/lessons/01-ring-attention.md)|Circulate source blocks with global softmax normalization.|

Each new lesson contains the motivating failure, source visuals and original worked figures, worked math, a complete executable Python/PyTorch reference, changed-condition practice and references. The 27 additional editable figures trace the numerical states, memory lifetimes and controlled changes that overview pictures alone did not show. Read the relevant axes and calculations before running code; predict a changed case before revealing its answer. Colab file execution instructions accompany each reference.

Public Vizuara workshop/article/parallelism lessons anchor the continuation. ALiBi and Performer use explicitly identified primary-author figures where a matching public Vizuara lesson was unavailable. The ring panel is a passive vector export of the visible source geometry; the primary architecture/capacity figures are original. Original Lecture 9 remains excluded.

Local arithmetic, reference execution and reader checks prepare the material. They do not establish learner mastery, full-model task quality, optimized GPU performance, actual hosted Colab execution or a completed learner project. The existing Attention Foundations boilerplate and P00 checkpoint remain separate.
