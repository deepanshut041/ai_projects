# Complete the missing worked visual steps

The learner authorizes original figures where matching source teaching visuals are missing. Keep existing source diagrams and accepted explanations; add precise, editable passive SVG figures beside the relevant worked calculations. Lead: repository developer-learning-tutorials; planning coordinates ownership and preservation. No new videos or learner model implementation.

## Coverage audit and sequential authoring order

Existing Foundations and MHA include the complete earlier presentation/lab audit and remain unchanged. The fourteen continuation lessons have source overview pictures, but several numerical transitions and failure cases exist only in prose. Each row below is authored, numerically checked, rendered and visually reviewed before moving to the next lesson. Original figures use the same concrete inputs as the complete executable references. Source attribution stays at the end; no added provenance JSON.

| Lesson | Missing visual understanding | Planned original boards |
| --- | --- | --- |
| KV cache | Reuse still computes one full retrieval row; chunk masks need absolute positions | Fifth-token weighted mixture; correct and incorrect chunk mask |
| MQA | Shared KV can produce different head answers | Four query rows, their weight bars and outputs |
| GQA | Head-to-group mapping determines storage and retrieval | Contiguous head mapping and comparable byte bars |
| RoPE | Equal distance survives a common position shift | Three rotations with the same relative angle |
| ALiBi | Penalties become probabilities; slopes give different priors | Score-to-weight calculation and multi-head distance slopes |
| MLA | Content absorption and positional path have different lifetimes | Expanded/absorbed calculation; decoupled cache flow |
| FlashAttention | Rescaling is required when the running maximum changes | Correct numerical tile merge; common-boundary transfer ledger (the local-average counterexample is drawn in Ring) |
| Sparse/local/global | Direct connectivity differs from layer reachability and storage slots | Causal visibility masks; layer reach and absolute slot labels |
| Linear attention | S and z must update together; histories can collide | Two writes and normalized read; collision (the existing dimensioned storage formula remains) |
| Performer | Positive features approximate a kernel; finite feature error varies | Worked feature products; all measured seeds on one error scale |
| DeltaNet | Residual write corrects a specific prediction and can interfere elsewhere | Full/partial overwrite; overlap-dependent change |
| Gated DeltaNet | Decay precedes prediction and differs from write strength | Exact two-association transition; gate boundaries and lifetime |
| PagedAttention | Address mapping and partial-tail copy-on-write | Concrete logical address translation; fork/append/release lifecycle |
| Ring Attention | Source-owner rounds, unequal softmax mass and causal balance | Four source-owner rounds; weighted block merge; balanced assignment |

## Verification and preservation

Entry digests and Git working/index snapshots are saved under ignored .ai/attention-labs-source/2026-10-30/original-visuals/. Every existing example and source image must remain byte-identical. Check all SVG labels, arrows, bounds and values, re-render each changed package, inspect each board and run the required reader checks. Existing hosted, learner and full-model boundaries remain unchanged.

## Sequential results

All fourteen rows are authored in the recorded order. Each topic’s boards were rasterized and visually reviewed before the next topic was authored. There are 27 new passive SVG figures, explained inline in the fourteen existing lesson files. No chapter or category was added. Existing source figures remain available.

Self-review checked every arrow, matrix row/column, token index, unit, bar scale and plotted quantity against the adjacent explanation. ALiBi’s vertical coordinate was corrected so negative biases descend on the conventional axis. The linear-collision board uses an explicit fraction so numerator and denominator cannot be misread as consecutive operations. Generated boards have DejaVu Sans text, accessible title/description, full-size links and locally scrollable rendering through the existing reader; text/geometry bounds were checked during authoring.

All fifteen complete references passed again in Python 3.12.13 / PyTorch 2.14.1+cpu. Coverage includes numerical outputs, gradients, cached chunks, absolute masks, ownership/reset, error samples, allocation failure and causal source ownership as applicable. Raw run output stays ignored as original-visuals/reference-checks.txt. This is author verification, not independent learner testing.

The exact twelve Performer output-error samples were reproduced before chart review; bar heights use those six-decimal results on one common scale. FlashAttention’s corrected denominator/numerator and output match the executable eight-token fixture. Gated DeltaNet’s matrix states and wrong-order failure, PagedAttention’s allocation IDs/reference counts, and Ring’s rounds/136-pair balance match their complete references.


## Final reader and preservation checks

- Seven changed category packages validate and render. All 27 new assets appear in the existing fourteen lesson files; one lesson still owns each mechanism. There remain eight categories and 22 lesson pages. No Foundations/MHA explanation, source image or complete reference program changed.
- All fifteen complete references pass again. The twelve Performer plot samples are independently compared to the fresh executable output after six-decimal rounding. Twenty-seven passive SVGs have accessible title/description and 710 checked text elements within their view boxes. Every original source image/example and every Foundations/MHA file protected at entry remains byte-identical (292 protected files).
- Native tests: 27/27 pass; build: 286 records. The complete browser suite passes all 286 records, equations, image decoding, local links, downloads, print and widths 1440/768/390/320; five project collections/163 local project pages, media acceptance and 207 syntax-highlighted pages also pass.
- Collection checks pass at both / and /project_docs/, all five collections and fourteen site-wide courses, widths 1440/390/320.
- The user’s connected Chrome separately visits all fourteen continuation chapters in the static root build at 1440/390/320: 42 page/width checks, all 27 added figures at every width, loaded/full-size links/focusable scroll regions, zero equation errors and no document overflow. Both Ring practice reveals work at 320 px; keyboard scrolling moves the wide diagram without moving the document outside the viewport. The managed prefixed reader also displays the new GQA and Gated DeltaNet figures with full-size access. The full-size Gated DeltaNet SVG was visually inspected in Chrome. The live reader’s reload stream caused navigation waits; the static build allowed complete inspection without changing the browser or claiming a render failure.
- All four staged diffs and the CLI working diff match entry. Unrelated working patches remain identical: 37 Docs, three Projects and two attention-collection tracked diffs. Temporary rasterizations, browser output and run logs stay ignored. No extra tutorial provenance JSON, commit or external publication.

This is author teaching/technical review, with no independent learner acceptance claim. Model training, actual optimized GPU/distributed execution and hosted Colab results remain separate. Next learner action: read one tutorial’s explanation, trace its original worked figures and predict the changed case before revealing the answer.

Managed reader is running at http://127.0.0.1:4273/project_docs/. The reading route is left open in the user’s Chrome with all fourteen continuation rows and the 27-figure update visible. Temporary viewport overrides are reset and the temporary root review server is stopped.
