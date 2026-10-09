## Connect correctness to the eventual research goal

First build and train the complete tiny model. Inspect a bounded batch, losses and gradients, then test generation and checkpoint/resume. Use a held-out split and causal checks: a suspiciously low loss can come from a broken problem.

[[diagram:training]]

The memory board is a set of questions to investigate during the model project, not a proposed optimization result. Start by measuring weights, retained state, temporary tensors and transfers separately. Prefill processes a sequence; cached decoding adds positions while retaining previous keys and values.

[[diagram:memory]]

For ordinary MHA, each layer's K and V each contain B*T*D elements. Multiplying by two tensors, L layers and the dtype's bytes gives the cache's raw storage. This accounting changes with grouped heads, quantization or compression. Actual process memory also includes allocator overhead and workspace.

[[practice:cache-check]]

Putting weights or state on SSD changes where bytes live and how they move. Capacity alone does not establish usable latency or throughput. Study the implementation and measure a declared baseline before making a focused Research claim.

### Prepare to explain, then implement

Use the [whiteboard story](whiteboard/defense.md) and [challenge bank](whiteboard/questions.md). During a requested live defense, answer one question at a time before looking at answers. Save learner mastery only after an actual explanation or implementation exercise.

Next, follow P01-M1: create the independent GPT project, pin sources, declare its tiny configuration and data contracts, and implement one component at a time. This tutorial is the starting material; future chapters should follow actual model code and evidence.
