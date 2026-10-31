# Opening learning route

Updated: 2026-10-09
This diagram describes planned dependencies, not completed implementations.

```mermaid
flowchart LR
    T["P00-M1
Tensors, gradients, tiny MLP"] --> B["P00-M2
Byte/BPE tokenizer"]
    B --> A["P00-M3
Attention and block primitives"]
    A --> E["P00-M4
Full encoder–decoder"]
    E --> R["P00-M5
Task training and persistent recovery"]
    R --> H["P00-M6
Inspect, explain and transfer"]
    H --> G["P01
Small decoder-only text LM"]
    G --> G2["P02
GPT-2 coverage and cache"]
```

P00 encoder states supply decoder cross-attention; P01 removes that path. P00 uses paper post-norm/sinusoidal/ReLU, P01 uses pre-norm/learned positions/GELU. Versioned tokenizer/trainer contracts transfer; model weights do not. Canonical architecture previews live under cli/docs/architecture/ and their existing owner package.
