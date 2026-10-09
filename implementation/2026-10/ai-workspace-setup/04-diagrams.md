# Workspace architecture

```mermaid
flowchart TD
  Home[AI home: launcher and ownership] --> CLI[AI Projects CLI]
  CLI --> Projects[Projects umbrella]
  CLI --> Research[Research umbrella]
  CLI --> State[Ignored .ai state, logs, environments and local Git caches]
  Projects --> ProjectDocs[Independent Projects documentation]
  Projects --> Labs[Independent Linear, Retrieval and Attention labs]
  Research --> ResearchDocs[Independent Research documentation and source catalog]
  Research --> Generalization[Independent Generalization lab]
  ResearchDocs --> Sources[Ignored source archive]
```

Documentation builds read authored Markdown, local assets and the tracked catalog, never the raw archive. Child histories and parent pointers are separate. The master's area has no branch in this architecture.
