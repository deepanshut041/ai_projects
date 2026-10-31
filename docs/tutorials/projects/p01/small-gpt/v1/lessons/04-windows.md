## Lesson 4 — make batches that ask the right question

You now have fixed chapter strings and a fixed token-to-byte mapping. The model cannot consume an entire book at once. Give it bounded windows, and prove which labels each window contributes before building attention.

### Cut targets once, keep document seams intact

For `hi`, form `[BOS,104,105,EOS]`. Take adjacent slices: input drops the last item, labels drop the first. With context four, right-pad both to length four:

```text
x: BOS   h     i    PAD
y:  h    i    EOS   PAD
```

The last row does **not** train the model to output PAD. PAD labels are ignored by the loss. EOS labels count normally. BOS tells the model where a document begins; it is an input here, not a label.

For a longer document and T=3, start the next window at index 3 rather than 4. This shares the boundary token as context and predicts each target exactly once:

```text
stream: BOS a b c d e EOS
x0:     BOS a b       y0: a b c
x1:       c d e       y1: d e EOS
```

The new window cannot see earlier `a,b`; that is a deliberate reset-context approximation. It does not cross into a different document. Evaluating the same way produces a reproducible finite-context metric. A sliding evaluation with more preceding context is a different protocol and may have a different loss.

[[practice:windows]]

Append `windows` to `gpt_data.py`. The returned tensors have shape `[number_of_windows,T]` and int64 IDs. A Python list is flexible while constructing rows; the resulting tensor is rectangular storage for the model.

<!-- file:gpt_data.py mode:append -->
```python


def windows(documents, tok, context):
    if context < 1:
        raise ValueError("Positive context required")
    xs, ys = [], []
    for document in documents:
        ids = [BOS] + tok.encode(document) + [EOS]
        # Each target occurs once. Reset context at each window and document.
        for start in range(0, len(ids)-1, context):
            x = ids[start:start+context]
            y = ids[start+1:start+context+1]
            x = x[:len(y)]
            xs.append(x + [PAD] * (context-len(x)))
            ys.append(y + [PAD] * (context-len(y)))
    if not xs:
        raise ValueError("At least one document required")
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)
```

Run the smallest fixture immediately:

```bash
python -c "from gpt_data import windows; from gpt_tokenizer import ByteBPE; x,y=windows(['hi'],ByteBPE(),4); print(x.tolist()); print(y.tolist())"
```

Expect `[[257,104,105,256]]` and `[[104,105,258,256]]`. If you get identical x and y, you are training identity copying. If EOS disappears from labels, your stopping behavior has no direct target.

### Select training windows without hiding sampler state

A batch of B=8 contains eight windows. This first reference samples window indices uniformly **with replacement**. A window can appear twice, and an “epoch” has no special meaning. We record draws and global token counts instead. This choice makes the data state small and explicit; an epoch-shuffling loader is a later alternative with different state.

Append the remaining sampler and command-line entry to `gpt_data.py`:

<!-- file:gpt_data.py mode:append -->
```python


class Batches:
    """One explicit CPU generator; random windows with replacement, no workers."""
    def __init__(self, x, y, seed):
        self.x, self.y = x, y
        self.rng = torch.Generator().manual_seed(seed)
        self.draws = 0

    def next(self, size, device="cpu"):
        indices = torch.randint(len(self.x), (size,), generator=self.rng)
        self.draws += size
        return self.x[indices].to(device), self.y[indices].to(device)

    def state(self):
        return {"rng": self.rng.get_state(), "draws": self.draws}

    def restore(self, state):
        self.rng.set_state(state["rng"])
        self.draws = state["draws"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--vocab", type=int, default=260)
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.vocab), indent=2))
```

The CPU generator belongs to the data sampler. Model dropout uses Torch's model RNG separately. Saving both is necessary: recovering weights while drawing different next windows changes the optimization path. `indices` picks rows before transferring the batch to the selected device, so the entire text corpus need not occupy VRAM.

[[practice:batch-rng]]

Save [test_gpt_data.py](examples/test_gpt_data.py) as `test_gpt_data.py`. Run `python -m unittest test_gpt_data -v`. Four tests cover the tiny shift/tail, document boundary and one-time target coverage, Unicode/frozen artifact and next-draw restore. Open each fixture and explain what a wrong implementation would return. A test count alone is not the goal.

### Keep all denominators visible

A window may contain fewer than T labels because its document ended. With two windows containing 3 and 1 valid labels, the batch has four supervised tokens, not eight merely because the tensors have eight slots. Store this fact mentally now: the training loss, gradient accumulation and validation metric will all divide by the number of **valid labels**.

Right padding also gives attention a simple contract: the first token is real and real tokens never occur after a PAD. Padded queries may produce unused values, but they can see at least the first real key, so no softmax row is all negative infinity. Valid-token outputs and the loss remain unaffected by appended right padding. We will test that explicitly.

The input/label path is now correct without a model. Next, teach each position how to read its allowed prefix.

[Continue to Lesson 5](lessons/05-causal-attention.md)

### Complete files from this lesson

[gpt_data.py](examples/gpt_data.py)
