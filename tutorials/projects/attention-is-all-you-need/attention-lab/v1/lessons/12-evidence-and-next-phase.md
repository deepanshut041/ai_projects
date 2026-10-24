## Explain what works and prepare the next implementation

We can now follow an input through projections, head layout, scores, masks, normalized weights, retrieved values, concatenation and output projection. We can also explain a backward example. This completes the tutorial's current component scope. It does not fill the project's later placeholder files.

### Use each check for the claim it can support

| Evidence | What it checks | What it does not establish |
| --- | --- | --- |
| Hand-calculated weighted average and raw-X example | Known numerical forward results | Learned word meaning |
| Projection and matrix fixtures | Q/K/V orientation, bias, scores, context and head merge | Real-data generalization |
| PyTorch MultiheadAttention comparison | A separate implementation gives matching outputs/weights with mapped parameters | Full encoder/decoder correctness |
| Explicit/SDPA forward and gradient parity | Selected paths agree for exercised cases | GPU speed or universal bitwise equality |
| Padding and future perturbations | Forbidden positions cannot affect protected outputs in those cases | Correct future data splitting or label alignment |
| Entirely blocked and dropout tests | Defined edge behavior, finite gradients and train/eval distinction | Arbitrary malformed-input safety |
| 35 existing tests and runnable demo | Current local component is executable and checked | Trained translation quality or learner mastery |

The code has been reorganized by responsibility; the attention module's numerical implementation and demo outputs are unchanged. Its captured source and [reference-run evidence](project/reference-runs.md) let us distinguish an existing application capability from a new authored explanation. The Colab runner's operational cells are verified locally; actual hosted execution is recorded separately from the attention tests and demo.

### Draw the implemented computation without file-name memorization

Start with `(B,S,D)` floating source vectors and optional `(B,T,D)` target vectors. Identify the query input. Apply separate Q/K/V projections; split D into H heads of width d. Draw score rows as queries and columns as keys, mark allowed pairs, normalize along keys, multiply values, reassemble query-major head features, and apply the output projection.

Then change one condition. If a source pad vector changes, which outputs must remain fixed? If a future target input changes, which causal outputs are protected? If only V changes, should the attention weights move? If all keys are blocked, why can the final output still be the output bias? These changed cases reveal whether the computation is understood more clearly than reciting class names.

The [explanation practice](practice/explain-attention.md), [attention learning points](project/learning-points/01-attention.md) and [comparison record](project/comparisons/01-attention.md) provide worked answers and useful distinctions. No response is recorded as a learner assessment.

### Part 2 will define real text before model expansion

The next implementation works in `src/tokenizer.py`, `src/prepare_data.py` and the package's tokenizer/data modules. Its first contract is one English/French pair transformed into reproducible integer sequences and a batch. Before acquisition, decide the official dataset/subset, license and split authority; pin its revision and record content identities. Fit a newly trained tokenizer only on the allowed training text and save its vocabulary/config identity.

For an illustrative target `[BOS, je, bois, EOS]`, a conventional shifted example uses decoder input `[BOS, je, bois]` and labels `[je, bois, EOS]`. The labels are what each query position should predict. Padding needs both key-validity masks and a separate ignored-label rule. Record special token IDs and truncation/empty-example policy explicitly. These examples define questions for Part 2; they are not acquired token IDs or implemented collation results.

We will compare an actual pair before and after tokenization, print input/label rows beside each other, check masks and show which targets count in the loss. Source/target tokenizers have distinct identities in the planned video-following route. A sentence-length or vocabulary setting in a draft JSON file does not settle the data contract.

### Part 3 will assemble and train a complete model

Only after data and batch contracts are implemented will the later phase add token embeddings, positions, residual paths, normalization, feed-forward blocks, encoder/decoder stacks and a vocabulary prediction head. The attention module becomes one ingredient in that architecture.

Training will need a token-loss denominator, optimizer/schedule, bounded profiles, explicit new/resume modes and persistent full-state checkpoint generations. Inference will need a start token, stopping rule and generation loop; evaluation will need a fixed held-out protocol and clearly configured BLEU. The current config files and Colab instructions describe intended work, not measured training/recovery capability.

### Keep the completion states separate

The [readiness record](project/readiness.md) marks setup and attention software checks as available, with learner review still open. Dataset/tokenizer, full model/training, translation metrics and actual Colab deleted-runtime recovery remain pending. We add those lessons when the corresponding implementation and evidence exist.

This preserves a useful next action: review one complete attention trace now, then begin the data contract when moving to implementation. There is no need to recreate the project or restart a general Python course.

[[practice:handoff-check]]

### Source and review downloads

The [captured source bundle](code/source-bundle.json) contains exact UTF-8 file contents and hashes for the implementation, tests, notebook, lock and honest later placeholders. Its [file hashes](code/source-manifest.json) identify each captured file. Save that JSON file and the [restoration helper](examples/restore_source.py) into the same folder, then run `python restore_source.py /path/to/a/new-directory` there to restore a runnable project. The helper refuses an existing destination and verifies every hash before writing. Do not restore over a working project. The [drawing module](code/src/utils/attention_visuals.py) is available separately.

Review: [setup learning points](project/learning-points/00-setup.md), [attention learning points](project/learning-points/01-attention.md), [setup comparisons](project/comparisons/00-setup.md), [attention comparisons](project/comparisons/01-attention.md), [implementation](project/implementation.md), [reference runs](project/reference-runs.md), [readiness](project/readiness.md), [practice](practice/explain-attention.md), [source research](research.md), [arithmetic probe](examples/arithmetic.py) and [component probe](examples/verify_component.py).
