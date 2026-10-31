# Research and inspected evidence

The public project README defines phases 00–06 and pins google/gemma-4-E4B-it at ee0ef6023621cff504d758262d4e04895a5af4a2. Actual core/config.py, core/workflow.py, utils/commands.py and .ai-project.toml are read-only setup plumbing. Data/quantization/evaluation packages are pending. The tutorial will capture these exact working files.

Primary sources inspected on 2026-10-26: pinned official Gemma config/model card; ONNX QuantizeLinear specification; GGUF specification and llama.cpp quantize documentation; AWQ arXiv 2306.00978v6; GPTQ 2210.17323v2; KIVI 2402.02750v2; Hugging Face KV-cache guidance; LongBench task source. Exact URLs and proof limits belong in course research.md/lesson.json.

Named source commits: llama.cpp 9b8446b91434e0c0cdc293baf0b29c099eaaa0c7; gptq 2d65066eeb06a5c9ff5184d8cebdf33662c67faf; llm-awq d6e797a42b9ef7778de8ee2352116e0f48a78d61. These identify inspected source, not compiled or validated model execution.

The pinned Gemma text config is hybrid: 42 layers (35 sliding, seven full), sliding_window 512, head_dim 256 and global_head_dim 512, two KV heads, 18 KV-shared layers. Cache formulas must model actual owners and backend allocation rather than multiplying 42 by one uniform head dimension. It also has per-layer inputs; effective parameter counts alone do not determine stored bytes. Capture config only; no weights requested.
