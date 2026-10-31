# Recovery flow — Colab Free training

```mermaid
flowchart LR
  Source[Exact child source and config] --> Setup[Fresh runtime setup and data]
  Setup --> Select[Explicit new or resume]
  Select --> Train[Common trainer: bounded slice]
  Train --> Local[Stable local update snapshot]
  Local --> Persist[New persistent generation]
  Persist --> Verify[Read back, verify, complete last]
  Verify --> Train
  Kill[Runtime deleted or abrupt stop] --> Setup
  Verify -. latest valid state .-> Select
```

COLAB.md is the canonical detailed contract. Completed markers require matching manifests/files. Failed copies preserve older verified state; runtime-local snapshots alone cannot recover a deleted VM. This is the required design, not an implemented training workflow.
