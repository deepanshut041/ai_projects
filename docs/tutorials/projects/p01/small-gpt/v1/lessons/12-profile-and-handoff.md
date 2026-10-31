## Lesson 12 — explain the cost and prepare for GPT-2

You now have a full small language-model pipeline and a recovery protocol. Finish P01 by showing what it contains, what it costs and which claims the evidence supports. This turns a collection of working scripts into a project you can explain and extend.

### Count actual parameters before estimating memory

For our biased linear layers, two LayerNorms per block and tied vocabulary head:

$$P=VD+T_{max}D+L(4D^2+2DF+9D+F)+2D.$$

Derive it rather than memorizing it:

| Component | Parameters |
| --- | --- |
| Token embedding and tied head together | VD |
| Learned positions | Tmax D |
| QKV plus attention output matrices | 3D² + D² |
| Their biases | 3D + D |
| FFN matrices | DF + FD |
| FFN biases | F + D |
| Two block LayerNorm scale/bias pairs | 4D |
| Final LayerNorm | 2D |

With V=260, Tmax=64, D=64, F=256 and L=2: embedding 16640, positions 4096, each block 49984, final norm 128, total **120832**. Count the model's unique parameters and compare; a second 16640-parameter vocabulary head would indicate accidental untying.

[[practice:parameter-count]]

### Separate storage lifetimes

FP32 weights alone occupy 4P=483328 bytes, about 0.461 MiB where 1 MiB=1024² bytes. Weights, gradients and two FP32 Adam moments suggest roughly 16P=1933312 bytes. This omits optimizer step tensors, activations, temporary workspaces, allocator reservation and Python overhead. It is a ledger entry, not a peak-RAM prediction.

An explicit attention score tensor uses BHT² elements. At B=8,H=4,T=64 in FP32 it needs 524288 bytes **per score tensor**. Doubling T quadruples that quantity. An optimized attention implementation may avoid materializing the same full intermediate; its peak behavior must be measured. Do not claim an optimization wins from the formula alone.

[[diagram:memory]]

Weights and optimizer moments live across updates. Activations needed for backward live during a training step. A KV cache, if added, lives across decoding calls for one request and must be cleared between requests. These are different lifetimes and access patterns; the current uncached model does not secretly contain a KV cache.

### Measure a bounded CPU operation

Save [gpt_profile.py](examples/gpt_profile.py) as `gpt_profile.py` and run `python gpt_profile.py`. It fixes one thread and seed, warms up three times and reports ten forward/backward timings at T=32 and 64. It also checks parameter counts. The T=32 model has fewer position parameters, so record that configuration difference.

The measured region excludes optimizer update, data loading and persistent saving. It uses synthetic IDs. Report median, range, batch, shape, dtype, threads, runtime and the actual CPU model from your environment record. Throughput of the complete trainer needs a separate end-to-end measurement. Repeated tests on one host are not a hardware benchmark across platforms.

Create `docs/parameter-memory-ledger.md` with formulas and measured quantities. Create `docs/error-analysis.md` around one observed failure: for example, a causal-mask bug that changes earlier logits, a tokenization change rejected by resume, or repetitive generated text despite falling loss. State the prediction, evidence, change and what happened afterward. Preserve a failed run before describing an improvement.

[[practice:profile]]

### Review the required route

Use the [project readiness record](project/readiness.md). M1–M3 and M5 are required: provenance/data/config, complete checked model, actual training/generation/recovery, and resource/diagnostic explanation. The reference tests and author runs are available to compare with; they do not mark your own rows passed.

Draw the model from memory using [the explanation practice](practice/explain-your-model.md). Label `[B,T]`, `[B,T,D]`, `[B,H,T,d]`, `[B,H,T,T]` and `[B,T,V]`, then trace one label's gradient to the tied token embedding. Explain why changing a future input cannot change an earlier logit, why PAD labels are excluded from the denominator, and why optimizer/RNG state matters on resume. Use [the question bank](practice/questions.md) when you need targeted review.

### What GPT-2 adds

P02 is not “add pre-norm again”; you already used it. The next project establishes GPT-2-specific tokenizer/preprocessing, configuration and weight-layout compatibility where claimed, reviews source-specific initialization/activation/details, and makes KV-cache parity and resource comparisons required. Your P01 language-model trainer, identity contracts and tests are useful starting artifacts, but reverify them after changes.

For the optional early cache exercise, create a separate `gpt_cache.py` only after uncached correctness. Store each layer's keys/values as `[B,H,t,d]`, append along t, and align new queries to absolute positions `past_length … past_length+q−1`. For a one-token query after a prefix, all cached prefix keys are past; blindly using a top-left triangular `[1,t]` mask would hide almost everything. Require logits parity with full-prefix execution inside Tmax, multi-token append checks, batch handling, bounds and fresh-request reset. Only then compare reallocating concatenation with preallocated storage. Rolling-window position reset requires a separate carefully specified parity protocol; it is not solved by merely dropping the oldest keys.

[[practice:cache]]

The optional GPT-1 history exercise belongs in its preserved tutorial: implement the historically appropriate post-norm variant/classification head and explicit pooling/objective contracts as a separate model. Neither historical classification nor the early cache is a missing required P01 step. Keep those choices visible in your exit record.

### Your next action

Return to the first incomplete required gate in [readiness](project/readiness.md). If all have actual passing implementation/run/hosted/explanation evidence, write a versioned `docs/p02-handoff.md` with source revision/hashes, tokenizer/config/split identities, test commands, numerical tolerances, profile limits and remaining GPT-2 differences. P02 starts from that evidence. Reading this page alone does not advance the roadmap.

[Project home and implementation](project/implementation.md) · [Readiness](project/readiness.md) · [Reference runs](project/reference-runs.md) · [Sources](research.md)

[Compact author result identities](project/reference-results.json)
