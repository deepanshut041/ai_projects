# Sources, teaching choices and evidence boundaries

Inspected 23 October 2026. The learner requested the depth and reader organization of Bike Demand Planner, scoped to setup and already implemented attention. Later text/data/model/training material will be added with implementation evidence. The application source belongs to ai_projects at 0fdd16c03c650e4f74dcfe32cc030f9254f4b2b9; this course is a separately authored Projects Docs artifact.

## Source hierarchy

The captured attention module and existing tests are the authority for current behavior. src/utils/attention_visuals.py supplies the controlled Q/K/V and output-projection fixtures plus the original drawing code. The Colab execution notebook lives in notebooks/ outside the core/data/training/utils packages, with entry scripts directly under src/. Its six operational code cells verify/extract source, install the locked environment, record runtime identity, run demo/tests and download logs. Tutorial prose and figures replace the former concept notebook. Local runner verification is separate from an actual hosted run. The runnable project source archive excludes raw captions, environments, data, weights and run trees.

The [original paper](https://arxiv.org/html/1706.03762v7) supplies the component definition and self/cross-attention roles; our explanations and figures use original toy tensors and the actual local implementation. The [PyTorch 2.14 SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) was checked for API semantics. No paper performance numbers are presented as our results.

The project follows Priyam Mazumdar's three supplied videos. Its reference code is pinned to PyTorch-Adventures revision 2f18a5470c4ef0a3d413dd7b18d07941f45779e9 under MIT. The [attention source video](https://www.youtube.com/watch?v=JXY5CmiK3LI), [data video](https://www.youtube.com/watch?v=UkBggwQsbF8) and [model/training video](https://www.youtube.com/watch?v=DOqg_QxInd8) retain their original roles. Videos 2-3 are future implementation inputs; no new media is generated here.

## Important local choices

The final API preserves src/tgt and q_proj/k_proj/v_proj/out_proj names. True/1 means an allowed key. Source padding and causality combine explicitly; cross-attention rejects the causal flag. Entirely blocked rows have zero context; output bias can remain. Returned inspection weights precede dropout. Efficient mode uses SDPA with evaluation dropout disabled, while requesting weights selects explicit calculations.

These details are source-specific contracts. Q/K/V notation, normalized weighted sums and chain-rule arithmetic have mathematical meaning beyond the implementation, but they do not establish useful language representations or training success. The head-width scaling rationale names its independence/variance assumptions rather than treating them as measured properties.

## Authored evidence and departures

The twelve-lesson route is a new source-backed explanation of the existing component. Standard-library scalar/list probes check selected numbers independently; PyTorch probes compare them with frozen actual module behavior. Existing tests include an independent PyTorch MultiheadAttention reference. The former concept notebook was replaced with a Colab project runner; its operational cells are checked locally with simulated upload/download and real commands. Attention arithmetic and fixed-seed demo outputs remain unchanged; selected figures are copied with source provenance.

The route begins with a weighted result, then setup, axes, normalization, projections, heads, padding, causality, cross-attention, public API, gradients and evidence/handoff. The later full model is described only to explain the component boundary. Learner understanding, hosted recovery and translation quality remain unassessed. Runtime figures are synthetic/random demonstrations, not trained alignments or hardware benchmark plots.
