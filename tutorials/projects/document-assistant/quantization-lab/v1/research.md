# Sources, departures and unknowns

Authored on 26 October 2026 after inspecting the existing Attention/Bike teaching format and the actual Document Assistant boilerplate. The complete outline was saved before lesson prose. The course covers the public README's phases 00–06 without starting model implementation or changing the existing learner checkpoint.

## Source authority

The pinned official Gemma configuration/model card informs model architecture and source identity. llama.cpp architecture/quantizer source is inspected at 9b8446b91434e0c0cdc293baf0b29c099eaaa0c7. GPTQ code is inspected at 2d65066eeb06a5c9ff5184d8cebdf33662c67faf, AWQ at d6e797a42b9ef7778de8ee2352116e0f48a78d61. AWQ arXiv v6, GPTQ v2 and KIVI v2 inform named method mechanisms; ONNX defines its integer operator rounding/saturation; GGUF defines a container. Version-sensitive cache/task documentation is dated in lesson.json. Primary source links appear beside the claims they support and in the rendered source ledger.

The loader/dispatcher/pending workflow and entry-point snapshots include exact repository/working-file digests. The public .ai-project.toml was read in full, manually captured as evidence/launcher-profile.txt and hash-compared: the generic source-capture helper rejects any dotfile. Its conservative guard was retained; no private configuration or credential was captured. The project-config TOML snapshot contains only public planned model/budget values.

## Deliberate departures

Lumen is an authored three-passage fixture, not acquired public QA data. Token packing uses a whitespace counter only in its visible arithmetic fixture; the real protocol requires the complete rendered prompt and actual model tokenizer. Exact aliases/evidence IDs teach counting, not semantic entailment. Quantization uses tiny affine/symmetric/ternary rules; signed nibble packing is not GGUF Q4_K. Scaling/compensation demonstrate mathematical steps, not complete AWQ/GPTQ. Cache owner layout is hypothetical and not a guessed Gemma allocation map. HTTP and answer adapters are explicitly fixtures and do not answer arbitrary documents.

All numerical latency, memory allowances, precision losses and bootstrap examples are illustrative. No paper's reported speedup/accuracy is presented as our result. Complete standard-library/C++ files and a separate arithmetic/failure audit establish the stated component scope only.

## Unknown until implementation

No source weights were downloaded. Runtime architecture presence does not prove complete conversion, per-layer embedding handling, kernels, device placement, cache support or quality. Actual Gemma artifact inventory, load/peak memory, QA accuracy/support/refusal rates, timing, advanced-method coverage and saved application remain pending. Recheck releases/support at implementation kickoff, then freeze a new explicit source/runtime decision if needed. Model/results, learner explanation and hosted proof stay distinct.

## Preservation and visibility

This shared tutorial is deliberately Git-eligible. Private project study notes and runtime data/weights/runs remain ignored. Existing Attention text/media/source, Bike results, the sibling search project and CLI behavior are preserved. Source changes are limited to tutorial links/preparation snapshots; no model action is enabled by authoring.
