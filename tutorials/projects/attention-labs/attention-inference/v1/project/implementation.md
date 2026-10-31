# Cache implementation and local checks

The KV-cache lesson contains the complete arithmetic and PyTorch references. Run cache_arithmetic.py with Python 3.12 and kv_cache.py with PyTorch available. Both files include their own inputs and executable checks. The Torch reference accepts projected Q/K/V, uses dense causal attention, stores separate K/V tensors, validates ownership/absolute positions and tests prefill/chunk equivalence, wrong offsets, failed-update integrity, reset, bytes and valid zero tokens.

This is illustrative component code, not the Attention Foundations learner model or a complete generator. Colab file execution is supported; actual hosted execution is not claimed. No training, application result or learner gate is advanced.

The separate PagedAttention lesson adds `examples/paged.py`: logical/physical block mapping, valid lengths, reference-counted prefix forks, partial-tail copy-on-write, release/reuse and failed-allocation integrity. It checks gathered CPU retrieval against contiguous retrieval and is not a fused paged GPU kernel.
