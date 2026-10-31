# Flows — proposed learning path and Project 1

These diagrams describe the plan, not implemented code.

```mermaid
flowchart LR
  A["1 GPT-1"] --> B["2 GPT-2"]
  B --> C["3 Llama"]
  C --> D["4 Qwen dense"]
  D --> E["5 MoE"]
  E --> F["11 MLA / DeepSeek"]
  E --> G["6 Hybrid recurrence"]
  C --> H["7 Vision-language"]
  C --> I["10 Gemma text"]
  B --> J["18 Diffusion objective"]
  G --> K["8 → 9 Qwen hybrid"]
  F --> L["12 → 15 DeepSeek capstone"]
  F --> M["14 Kimi capstone"]
  G --> M
  G --> N["16 / 17 recurrent branches"]
  H --> O["19 Unified multimodal"]
```

```mermaid
flowchart TD
  T["Byte IDs and learned positions"] --> E["Token + position embeddings"]
  E --> A["Causal multi-head attention"]
  A --> R["Residual addition then LayerNorm"]
  R --> F["GELU feed-forward"]
  F --> S["Residual addition then LayerNorm"]
  S --> LOOP["Repeat complete block"]
  LOOP --> LM["Tied token projection: next-token loss / generation"]
  LOOP --> CLS["Gather explicit CLS position: classifier loss"]
  LM --> JOINT["Optional combined fine-tuning objective"]
  CLS --> JOINT
```

Projects 1–5 establish the foundation; architecture learning continues through the remaining path. Branch arrows indicate learning prerequisites, not source-code dependencies between repositories.

```mermaid
flowchart LR
  A["Understand mechanism"] --> B["Implement complete tiny model"]
  B --> C["Train / inspect / debug"]
  C --> D["Explain behavior"]
  D --> E["Map memory and tensor access"]
  E --> F["Profile and optimize one question"]
  F --> G["Verify / record / next architecture"]
  G --> A
  E --> R["Larger-model SSD / RAM / GPU research"]
```

The learning cycle is required. Real large-model experiments additionally require pretrained checkpoints, enforced memory budgets, measured I/O and quality/latency evidence.
