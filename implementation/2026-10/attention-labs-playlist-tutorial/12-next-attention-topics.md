# Attention Labs continuation

User request: add every recommended next topic, one at a time, with the teacher's explanatory progression and matching original diagrams. Keep one complete lesson per mechanism. Shared tutorials belong under Docs, grouped by category; this does not request a new learner model implementation.

Lead: developer-learning-tutorials; planning coordinator supplies ownership and checkpoint discipline. Each topic follows source reading → teaching/visual coverage → worked arithmetic → executable reference → rendering and review before starting the next. Original explanations preserve the source's problem/diagram/calculation order. Image attribution stays in end references; no additional provenance JSON.

## Source boundary

Chrome confirms the original Attention Family playlist has nine public lectures plus one hidden unavailable video. Lectures after 9 are not published there. The learner was asked asynchronously whether Vizuara's other public DeepSeek, inference and parallelism teaching material should anchor the continuation, or whether specific videos will be supplied. Meanwhile inspect those public materials and primary sources. Do not label workshop content as a nonexistent Attention Family lecture or claim unread subtitles. Lecture 9 remains excluded.

## Ordered route and stopping checks

| Prior result | Next problem | Lesson | Observable check | Next handoff |
| --- | --- | --- | --- | --- |
| Causal MHA | Recomputing prefixes | KV cache | Cached chunks equal full causal outputs; bytes and invalid masks checked | Retained state makes sharing useful |
| KV cache | Duplicate per-head KV state | MQA | Independent queries share one KV head; cache equality | Partial sharing |
| MQA | Intermediate KV capacity | GQA | Explicit grouping equals expanded reference; endpoints MHA/MQA | Position-aware retrieval |
| Q/K/V and masks | Represent distance | RoPE, then ALiBi | Rotation/distance identities and cached positions | Latent positional split |
| GQA and RoPE | Compress features | MLA | Expanded/absorbed reference agrees; positional cache counted | Exact execution |
| Dense softmax | Avoid full intermediates | FlashAttention | Blockwise online softmax equals dense, including masks | Change connectivity |
| Dense visibility | Bound selected pairs | Sparse/local/global attention | Masks, pair counts and receptive fields | Summarize history |
| Weighted retrieval | Recurrent kernel state | Linear attention, then Performer | Parallel/recurrent equality; approximation error disclosed | Selective memory updates |
| Additive memory | Correct/forget associations | DeltaNet, then Gated DeltaNet | Overwrite, correction and decay checks | Serving/state ownership |
| KV cache | Fragmentation and sharing | PagedAttention | Logical/physical mapping and copy-on-write match contiguous reference | Distributed blocks |
| Online softmax | Distribute sequence blocks | Ring Attention | All simulated rounds equal dense output; causal empty blocks | Review limitations |

## Progress

KV caching authored and rendered first. One lesson includes all 23 original Chapter 5/7 workshop figures, sixteen checked score cells, a completely specified fifth-token mixture, absolute-position masks, state ownership/reset and memory accounting. Standard-library arithmetic and CPU float64 Torch references pass; three chunks match full causal outputs, invalid owner/position updates are rejected without mutation and the 960-byte payload is verified. Several source chart labels are inconsistent: retain the exact visual and explain the corrected dimensions/units beside it. Original Attention Family lessons and learner/model/hosted checkpoints remain unchanged. The older DeepSeek KV video page was inspected; its complete subtitles have not been read and the workshop figures are not claimed to be its slides. Desktop reader: all 23 figures decode, two practice prompts, no KaTeX errors or page overflow. Phone checks pass at 390/320 as well as desktop 1440: all 23 figures decode, both practice reveals work and there is no page overflow. Two long inline numerical rows were moved into scrollable display maths. MQA may start after this topic check.

## Completed sequential topic checks

MQA: one lesson, three original workshop figures, explicit scalar-head calculations and a complete 40-parameter layer. CPU float64 broadcast/loop outputs and input/all-parameter gradients agree; future intervention and actual singleton KV-head storage pass. Desktop/390/320 reader checks pass.

GQA: one lesson, five matching workshop figures, explicit contiguous mapping and parameter/cache calculations. Grouped/expanded outputs and QKV gradients agree for G=1/2/4, cached chunks agree and non-divisible grouping is rejected. Desktop/390/320 checks pass.

RoPE: one lesson, all thirteen original public Vizuara article diagrams captured in Chrome. Late animation states for the exact rotation/result/context-extension diagrams were retaken and visually inspected. Equal-gap, norm, explicit-block/output-gradient and cached-offset checks pass; desktop/390/320 image/practice/containment checks pass.

Source selection resolved: the learner selected Vizuara's other public lessons and matching visuals. Use public Vizuara sources where available; ALiBi and Performer use explicitly identified primary-source figures where a matching public Vizuara lesson is unavailable. Authenticated/member-only Pods content is not accessed or reconstructed.

ALiBi: authored after RoPE checks, with the original primary-author method and evaluation figures; the numerical row, chunk/common-shift and future-intervention checks pass. Desktop/390/320 reader checks pass.

MLA: three original source figures; expanded/absorbed outputs and all input/coefficient gradients agree with a decoupled rotary path. Chunks, wrong-offset detection and 400-byte latent-plus-position payload pass. User Chrome desktop/390/320 image/equation/practice checks pass.

FlashAttention: all fourteen original Chapter 10 figures. Corrected source row/state arithmetic and memory-unit/IO annotations. CPU dense/online outputs and gradients, block widths 1/3/4/8, both visit orders, causal masks/offsets and large logits pass. User Chrome desktop/390/320 image, equation and practice checks pass.

Sparse/local/global: eight original diagrams, selected/dense outputs and gradients, causal future intervention, 26/30/36 pairs, rolling cache and exact reachability pass. Linear: four original diagrams, pairwise/recurrent outputs and gradients, continuation/reset/future probes and distinct-history collision pass. User Chrome desktop/390/320 image/practice/math/layout checks pass for both before the next topic.

Performer: original primary method/variance figures; orthogonal-block Gaussian-radius features, same-feature output/gradient/chunk equality and finite-seed softmax error table pass. DeltaNet: three original teacher figures; residual/matrix outputs/gradients/chunks, unit-key overwrite/partial/zero writes and independent online-SGD identity pass. Gated DeltaNet: complete original scalar/channel diagram; residual/affine outputs/gradients/chunks, alpha/beta boundaries and wrong-order detection pass. User Chrome desktop/390/320 checks pass for each before the next topic.

PagedAttention: all ten original workshop figures; CPU scattered/contiguous retrieval, partial-tail COW, reference counts, release/reuse, stale-token exclusion and allocation-failure integrity pass. Ring: passive vector export of observed initial teacher panel and complete causal matrix/workload bars plus both complete primary figures; all source-owner rounds, both directions/assignments/masks, outputs/gradients and 480-byte per-rank payload count pass. User Chrome desktop/390/320 figure/practice/equation/containment checks pass for each.

## Current delivery

All fourteen new topics were completed sequentially and individually checked before starting the next. There are 22 lesson pages in eight tutorial categories. Original Foundations bytes and MHA worked math/programs/figures are preserved; only stale MHA future-scope wording and shared navigation/support metadata are refreshed. Source visuals number 96 new placements, including the passive source-panel export. Final collection-wide checks pass, as recorded below.

## Final verification — 2026-10-30

- All fifteen new self-contained Python references were executed as each topic was completed. Torch references use the prepared Python 3.12.13 / PyTorch 2.14.1+cpu environment and CPU float64; standard-library arithmetic is separate. All linked new reference sources match their displayed complete code.
- `npm --prefix website test`: 27/27 pass. `npm --prefix website run build`: 286 records and 13 catalog source references. The latter count is the site's shared catalog, not the number of references in the new lessons.
- Complete `npm --prefix website run test:browser`: 286 record pages, search/downloads/links/print and 1440/768/390/320 containment; project navigation across five collections and 163 local pages; media acceptance; syntax checks across 207 highlighted pages all pass.
- `test:collections` passes at `/` and `/project_docs/`, widths 1440/390/320, all five collections and 14 site-wide tutorial courses. Attention Labs contributes eight categories with 22 chapter pages. The collection search assertion now derives its expected count from the discovered collection instead of assuming two courses.
- The user's connected Chrome session was used for source inspection and final review after their browser preference. All fourteen new chapter URLs also pass root-hosted focused checks: every one of 96 inline visual placements decodes, equations have no KaTeX errors, all 28 revealable practice controls are present, and page containment holds. Each topic's prefixed desktop/390/320 reveal/containment checks preceded authoring the next topic; the final Ring additions were rechecked at root. The ordered reading guide links directly to all fourteen chapters and the category page has eight cards.
- Root/prefixed collection screenshots and native browser output stay ignored under website/test-results. No extra source/visual provenance JSON was added to tutorials. Original source PNGs are retained; the two interactive-guide panels are passive vector exports of observed geometry/labels/colors.
- Preservation audit: 184 of 193 entry-time Attention tutorial files are byte-identical; the nine intentional changes are Heads manifest/render/research, the single MHA page's stale future-scope wording and three support pages. Every Foundations file and every existing MHA program/image remains byte-identical. All 33 preexisting unrelated tracked tutorial/site diffs and the CLI working diff are preserved. Saved staged diffs across Docs, Projects, CLI and the attention collection are identical to entry.

Source limitations: the original Attention Family playlist does not publish these fourteen continuation lectures. New content follows inspected public written Vizuara lessons and primary methods, without claiming new complete subtitle/video audits. ALiBi/Performer use primary-author figures. The public ring guide's controls did not advance during inspection; only observed panels are exported, while balanced rounds are derived and independently checked. Model/task quality, optimized GPU runtime, distributed hardware execution, learner acceptance and actual hosted Colab execution remain unclaimed. No commit or external publication was performed.

Managed reader restored at http://127.0.0.1:4273/project_docs/. Final connected-Chrome check confirms eight Attention Labs category cards, fourteen linked continuation rows, no unresolved local links in the reading route and no page overflow. The reading route is left open in the user's Chrome. Temporary root-hosting review server stopped.


Subsequent visual completion: at the learner’s request, 27 original worked SVGs were added across the fourteen continuation lessons. Existing source figures remain intact. See [the visual completion record](13-original-visual-completion.md) for arithmetic, sequential review, browser and preservation evidence.
