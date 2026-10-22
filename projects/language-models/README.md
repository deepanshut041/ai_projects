# Language Models

This group is for complete small language-model implementations, starting with Small GPT and GPT-2 and expanding into other model families as I reach them in the roadmap.

The prerequisite mechanism exercises live in [Transformer Labs](../transformer-labs/README.md). Here I'll assemble those ideas into complete models, train on text, inspect generated samples and held-out loss, and verify persistent training recovery.

**Status:** group prepared; no language-model implementation has been created here yet. The Small GPT tutorial already exists, but its learner implementation and hosted training are still pending.

| First builds | Main focus | When to start |
| --- | --- | --- |
| Small GPT | A complete causal decoder, next-token training, generation and recovery | After the actual P00 foundation gates |
| GPT-2 | Architecture-specific initialization/tokenization, weight mapping and KV cache behavior | After Small GPT |

Later model families follow the [canonical roadmap](../../docs/guide/roadmap.md); this directory name does not limit the group to GPT architectures. No extra model folders are being created just to reserve names.

Each implementation will be an ordinary directory tracked by `ai_projects`, with its own source, dependencies, configs, notebook, tests and notes as needed. It will not have a nested `.git` directory or submodule registration. Shared checkpoints and decisions remain in [Projects guidance](../../docs/guide/resume.md).

When I actually begin a new implementation, the default creation route is:

```bash
./ai new project small-gpt --title "Small GPT"
```

That command creates `projects/language-models/small-gpt/` as a local Python scaffold. It does not train the model or complete a learner milestone. Use the [Small GPT course](../../docs/tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md) and [Colab contract](../../docs/guide/COLAB.md) for the build and recovery requirements.
