# Documentation ownership and learning flow

```mermaid
flowchart TB
  P[ai_projects: entry points and controls] --> D[docs: one documentation repository]
  P --> M[projects: independent model repositories]
  D --> R[Roadmap, checkpoints and setup]
  D --> T[Tutorials: editable source, diagrams and C++/Python examples]
  D --> E[Implementation and verification records]
  D --> S[Existing documentation site]
  T --> S
  M -->|actual source and verification| T
  T --> L[Implement, train, inspect and explain]
  L -->|bounded focused question| Q[Separate Research study]
```

The source Diagram JSON for the lesson stays in its lesson.json; six compiled SVGs are beside it. Root architecture sources/images were refreshed using the existing renderer after the move.
