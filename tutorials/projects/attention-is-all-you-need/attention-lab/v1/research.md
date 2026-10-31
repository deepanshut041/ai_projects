# Four-row fixture follow-up — 27 October 2026

The learner requested all four binary pairs to make the chosen input rows easier to follow. The source is now “We read books daily,” with the fourth illustrative row `[0,0]`. This focused numerical revision reuses the authored attention explanations inspected below. It changes the arithmetic, diagrams and exact runnable examples consistently while preserving the established teaching progression. These coordinates are illustrative rather than learned word meanings; a real zero row is distinct from masked padding. Independent checks and rendering are recorded in the owning four-position follow-up. No source model, media or learner milestone is changed.

# Current research and whole-course comparison — 26 October 2026

The latest learner instruction rejects a paragraph-level repair: compare the complete course with good attention tutorials, diagnose its teaching structure and remove its videos. The [full comparison](project/teaching-review.md) summarizes the review; the owning implementation record retains the twelve-chapter audit. The revised route starts from translation input/output and the missing cross-row dependency, then derives selection/content, layouts, masks, learning and the actual implementation.

The learner subsequently described the detailed explanations as good and requested a big overview followed by progressively deeper passes. The entry and opening diagram now establish the whole process and explain the two jobs before naming their components. A chapter map makes the successive zooms explicit; the detailed calculations remain intact.

Read the primary authored Illustrated Transformer, the historical 2018 Annotated Transformer, D2L Q/K/V and Bahdanau chapters, and the original Bahdanau/Transformer papers on 26 October 2026. External examples and figures are not copied. Numerical calculations and exact source remain independently checked local material. The separate Q/K/V choices and component semantics are supported by actual source; claims about teaching organization are editorial assessment. New SVGs are original diagrams of this route’s data and arithmetic.

All Attention video bindings are removed; original local media/provenance is recoverable. Previous approvals, byte checks and playback receipts are historical evidence, not current teaching acceptance.

## Earlier inspection record

# Sources, teaching choices and evidence boundaries

Inspected 23 October 2026. The learner requested the depth and reader organization of Bike Demand Planner, scoped to setup and already implemented attention. Later text/data/model/training material will be added with implementation evidence. The application source belongs to ai_projects at 0fdd16c03c650e4f74dcfe32cc030f9254f4b2b9; this course is a separately authored Projects Docs artifact.

## Source hierarchy

The captured attention module and existing tests are the authority for current behavior. src/utils/attention_visuals.py supplies the controlled Q/K/V and output-projection fixtures plus the original drawing code. The Colab execution notebook lives in notebooks/ outside the core/data/training/utils packages, with entry scripts directly under src/. Its six operational code cells verify/extract source, install the locked environment, record runtime identity, run demo/tests and download logs. Tutorial prose and figures replace the former concept notebook. Local runner verification is separate from an actual hosted run. The runnable project source archive excludes raw captions, environments, data, weights and run trees.

The [original paper](https://arxiv.org/html/1706.03762v7) supplies the component definition and self/cross-attention roles; our explanations and figures use original toy tensors and the actual local implementation. The [PyTorch 2.14 SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) was checked for API semantics. No paper performance numbers are presented as our results.

The project follows Priyam Mazumdar's three supplied videos. Its reference code is pinned to PyTorch-Adventures revision 2f18a5470c4ef0a3d413dd7b18d07941f45779e9 under MIT. The [attention source video](https://www.youtube.com/watch?v=JXY5CmiK3LI), [data video](https://www.youtube.com/watch?v=UkBggwQsbF8) and [model/training video](https://www.youtube.com/watch?v=DOqg_QxInd8) retain their original roles. Videos 2-3 are future implementation inputs. The new teaching companions explain the implemented attention component.

## Important local choices

The final API preserves src/tgt and q_proj/k_proj/v_proj/out_proj names. True/1 means an allowed key. Source padding and causality combine explicitly; cross-attention rejects the causal flag. Entirely blocked rows have zero context; output bias can remain. Returned inspection weights precede dropout. Efficient mode uses SDPA with evaluation dropout disabled, while requesting weights selects explicit calculations.

These details are source-specific contracts. Q/K/V notation, normalized weighted sums and chain-rule arithmetic have mathematical meaning beyond the implementation, but they do not establish useful language representations or training success. The head-width scaling rationale names its independence/variance assumptions rather than treating them as measured properties.

## Authored evidence and departures

The twelve-lesson route is a new source-backed explanation of the existing component. Standard-library scalar/list probes check selected numbers independently; PyTorch probes compare them with frozen actual module behavior. Existing tests include an independent PyTorch MultiheadAttention reference. The former concept notebook was replaced with a Colab project runner; its operational cells are checked locally with simulated upload/download and real commands. Attention arithmetic and fixed-seed demo outputs remain unchanged; selected figures are copied with source provenance.

The revised route begins with the need for context, then a supplied weighted result, query-dependent normalization, all queries, projections, heads, padding, causality, cross-attention, a same-fixture gradient, the complete dependency-closed project class, and a finish. Setup is a run reference after this conceptual route. The later full model is described only to explain the component boundary. Learner understanding, hosted recovery and translation quality remain unassessed. Runtime figures are synthetic/random demonstrations, not trained alignments or hardware benchmark plots.

## Teaching revision — 24 October 2026

The learner rejected the previous flow and hidden helper calls. The revision uses X=[[1,0],[0,1],[1,1]] and V=2X throughout; identity Q/K projections reproduce the earlier retrieval rather than introducing a new fixture. Every Python fence is independently executable and the final class includes its exact current helper definitions. Programs are copied directly from visible fences. Independent standard-library arithmetic and finite differences check the numerical claims. Current source is inspected at Projects commit 4013627331c881a12082f71b2861a4d631b68443; its numerical module is unchanged. Original source captures above retain their historical provenance.

The paper's architecture/attention sections and PyTorch 2.14 SDPA documentation were rechecked on 24 October 2026. No external figures or copied explanation are used. The previous fourteen Attention video bindings are detached and archived. The learner approved the rewritten text and requested new videos on 24 October 2026. Eleven newly authored companions follow that approved route. Source checks and verified playback do not establish learner mastery or acceptance of the new video content.
