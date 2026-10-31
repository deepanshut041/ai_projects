# Return to this route

This is the `attention-heads` tutorial under Attention Labs. The runnable files are illustrative references inside this package; no learner model has been created or trained. Read the single MHA lesson in order: independent heads, packed tensors, then implementation boundaries. Continue with separate MQA, GQA and MLA lessons. The complete source and figures are available through the tutorial downloads.

The MHA source is Vizuara's The Attention Family playlist, Lecture 8, with its original linked presentation. The later head-layout lessons use Vizuara's public inference workshop and their primary papers, identified in each lesson's references. Read the source map for the exact mapping and evidence limits. Lecture 9 is omitted. All fundamental mechanisms stay together; MHA lives in the separate Attention Heads route.

From this directory run the displayed `python3 examples/...` commands, and the displayed C++ command where applicable. Every helper is included in the package and shown before its first consequential use. The PyTorch packed reference, when present, uses its separately stated dependency and execution record.

Author arithmetic and rendering checks are recorded in reference-runs.md and the owning implementation package. Learner explanation is unassessed. No dataset, fitted translator, benchmark, hosted notebook execution, model training or new learning milestone is implied.

The revised reading route follows the teacher one lecture at a time, with complete worked examples and inline diagrams/charts. Attribution stays in references; source acquisition metadata is not part of the lesson. Project implementations and learner checkpoints are unchanged.

## Added head-layout references

`examples/mqa.py` implements a complete tiny projected MQA layer; `examples/gqa.py` checks grouped/expanded projected retrieval and cached query offsets; `examples/mla.py` compares expanded/absorbed latent retrieval with decoupled RoPE. Each is displayed completely in its matching lesson and can be uploaded to Colab and run with PyTorch. Their local CPU float64 output/gradient/state checks are recorded in research.md and the continuation record; actual hosted execution and trained-model quality remain separate.
