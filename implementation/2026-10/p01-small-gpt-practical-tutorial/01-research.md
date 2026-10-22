# Source inspection and consequential choices

- Roadmap: required P01 is a complete pre-norm causal decoder, learned positions, GELU and final LayerNorm. 2 layers/D64/H4/T64/F256/V260 first; M1–M3/M5 required, M4/M6 optional. No actual gpt-lab exists.
- P00 readiness: frozen tokenizer/source handoff; no transferred conditional-model weights, no live sibling imports. Its local author task failure and pending hosted/learner gates remain unchanged.
- build-nanogpt train_gpt2.py inspected at `6104ab1b53920f6e2159749676073ff7d815c1fa`: fused QKV, pre-norm, learned positions, final norm, tied head, tanh GELU. Our educational implementation uses explicit reference attention and independent SDPA comparison; no pretrained weights/distributed defaults. Source URL in course research record.
- PyTorch installed author runtime: 2.14.1+cpu / Python 3.12, no NumPy; official 2.14 API docs and checkpoint/reproducibility guidance inspected. Reference uses standard library + Torch and records RNG/device/version; no AMP in the first trainer.
- Text: Project Gutenberg ebook 11 (Alice), retrieved 2026-10-09 local date; raw UTF-8 bytes 174311, SHA256 `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`. Keep raw bytes ignored. Verify the pinned hash before parsing. Chapter I–IX train, X validation, XI–XII untouched test; split before fitting any BPE/windowing. This one-book split is a bounded teaching corpus, not broad-language evaluation.
- Authoring reference must implement local recovery transparently while naming remaining hosted proof. Full state, fixed total schedule, immutable verified generations, older fallback and retention budgets follow current COLAB.md. A checksum is not proof of deleted-runtime Drive survival.

Research and checks will be updated during authoring.

## Declared CPU author checks (before running)

The full text demonstration uses the roadmap 2-layer/D64/H4/T64/F256/V260 model, dropout 0.1, seed 17, AdamW, 300 total updates, 20 warmup updates, LR 0.001→0.0001, batch 8. On untouched validation chapter X, require an NLL improvement of at least 0.05 over the measured untrained model; separately report comparison with add-one training-unigram NLL. This is an author demonstration criterion, not a universal language-quality or learner gate. Preserve failures and samples. Test chapters XI–XII remain unused.

The fixed-batch debugging drill uses dropout 0, seed 17, 2 short training examples, 300 constant-LR 0.003 updates. Declare final mean NLL below 0.1 as its fitting target, separate from held-out quality. Recovery parity uses a fresh process, nonzero dropout and the same identity, comparing uninterrupted 8 with 4+4 updates exactly on CPU; corruption/interrupted-copy/storage-budget guards are separate tests.
