# Course 1 evidence and the Small GPT handoff

Course 1 builds translation and a RoBERTa encoder for MLM and extractive QA. Reading, running a supplied reference, implementing a model and explaining it are separate accomplishments.

| Gate | Translation evidence | RoBERTa evidence | Current state |
| --- | --- | --- | --- |
| Numbers/gradients | Q/K/V, heads, weighted values, central differences | Head shapes, tied-table gradients, selected loss | Authored checks pass; learner explanation pending |
| Data/objective | Parallel alignment, frozen vocabularies, shifts/masks | Train-only BPE, corruption labels, positions, character/window labels | Fixtures checked; learner implementation beyond existing attention pending |
| Complete model | Conditional forward/backward and efficient parity | MLM/QA forward/backward and compatible transfer | Both references implemented and checked |
| Bounded training | Loss/counts, free generation, held-out examples | Fixed held-out corruption; QA overfit plus separate held-out evaluation | Tiny author results recorded; broader/learner outcomes pending |
| Failure diagnosis | Wrong shift, future leakage, ignored source, EOS | Wrong corruption label, padding, partial answer, invalid span | Examples/checks supplied; learner investigation pending |
| Local recovery | Fresh-process weights/Adam/schedule/data/RNG | Dedicated corruption/batch RNG, MLM and QA continuation | Authored CPU checks pass |
| Colab Free recovery | Verify persistent save, delete runtime, resume and advance | Same drill for MLM and QA | Preparation supplied; hosted unverified |
| Explain/profile | Encoder/decoder/inference trace, measured resources | Corruption/encoder/offset/head/span trace, measured resources | Prompts supplied; learner evidence pending |

[Reference runs](project/reference-runs.md) states exact scope. A lower fixture loss does not settle held-out quality; a local save does not prove Drive recovery after deleting a Colab runtime.

## Explain before moving on

Work one query–key product and value mixture without looking up an API. Draw a decoder with target Q/K/V for self-attention and target Q plus source K/V for cross-attention. Show why teacher forcing needs both a shift and causal visibility.

For RoBERTa, explain right context, unchanged selected targets and vocabulary versus position heads. Map one character answer into overlapping windows; ignore a partial span and decode a valid complete span. Explain what survives interruption: moments, schedule, data state, global and dedicated RNG, and exact identities. Demonstrate corrupt-generation rollback.

## Course two

[Course 2 — build a small GPT language model](/records/tutorials/p01-small-gpt-v1/chapters/next-token/) begins next-token prediction. Its P01 identity and source path are retained. The new model uses a decoder-only pre-norm design, learned positions, GELU and a tied vocabulary head.

Carry axes, tokenization/split discipline, causality, token counts, stopping and recovery. Recheck them under GPT's own configuration. Translation and RoBERTa encoder weights are not automatically a GPT checkpoint. Later conversation/alignment and serving need their own objectives, datasets and evidence.

Record open gates and return to their lessons. The authored references are ready; this page does not claim personal learning or hosted completion.
