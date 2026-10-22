## Lesson 2 — make a project you can return to

You already know the first input and label arrays. Now create a small independent project where each new file has one job. We will build it in order. Do not paste an entire finished repository and then try to discover how it works.

### Start only the selected implementation

The tutorial files live in Projects Docs; your implementation will belong in `ai_projects/projects/gpt-lab`. From workspace home, use `./ai new project gpt-lab --title "Small GPT Language Model"` after the P00 exit review. Register it as a Projects child, not Research. If `gpt-lab` already exists, read its AGENTS/checkpoint and Git status and resume it. Do not overwrite it. Use `./ai task new "P01 Small GPT implementation" --owner gpt-lab --domains project` after registration to create the child-owned work record.

For following the code independently, you can instead make an empty practice directory with `mkdir gpt-practice` and `cd gpt-practice`. That disposable folder is sufficient for the examples; it does not replace the registered child when you begin roadmap work. From this point, **working directory means the learner project root** unless a command explicitly says otherwise.

Create folders with `mkdir -p configs docs notebooks`. The first implementation is intentionally a set of clearly named Python modules in the project root. Python finds sibling imports when you run a script from that directory. You can package them later without changing the math.

```text
gpt-lab/
  gpt_config.py        # dimensions and invariants, today
  gpt_tokenizer.py     # a frozen P00 source snapshot, Lesson 3
  gpt_data.py          # split, encode, window, sample, Lessons 3–4
  gpt_attention.py    # Q/K/V and causal attention, Lesson 5
  gpt_model.py        # blocks, positions, tied vocabulary projection, Lesson 6
  gpt_objective.py    # loss, optimizer, evaluation, Lessons 7–8
  gpt_sample.py       # bounded generation, Lesson 8
  gpt_checkpoint.py   # verified persistent generations, Lesson 9
  gpt_trainer.py      # updates and all resumable state, Lesson 10
  gpt_train.py        # explicit new/resume entry, Lesson 10
  configs/            # your resolved profiles
  docs/               # provenance, measurements, exit record
  notebooks/          # the thin Colab notebook, Lesson 11
```

This tree is a destination map, not a list of files you must create now. Each later lesson supplies its files, instructions and checks when their dependencies exist.

### Give Python an isolated runtime

Use Python 3.12 for the checked CPU reference:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__); print(torch.ones(2,3) @ torch.ones(3,1))"
python -m pip freeze > requirements-cpu-lock.txt
```

The matrix result is a two-row column of 3s. `torch==2.14.1` is the author-tested API version, not a claim that every host has its wheel. Record the Python patch version, package versions, OS and actual CPU in `docs/runtime.md`. Resolve an installation failure before proceeding. The Colab lesson uses a GPU-capable installation separately; do not install this CPU wheel as your GPU training environment.

Create `.gitignore` with `.venv/`, `__pycache__/`, `data/`, `runs/`, `exports/`, `*.pt` and notebook checkpoints. Track source, configs, tests and reports. The raw book and learned weights can be reproduced or restored; they do not belong in Git. This course needs standard library plus Torch, not a tokenizer service, paid API or remote training job.

### Write the model contract first

Create `gpt_config.py` with the following code. A dataclass collects named settings; `frozen=True` prevents accidental changes to this object's fields during a run. `__post_init__` checks invariants immediately after construction.

<!-- file:gpt_config.py mode:write -->
```python
"""The educational P01 model contract; no pretrained compatibility claim."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab: int = 260
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    hidden: int = 256
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.vocab, self.context, self.width, self.heads,
               self.layers, self.hidden) < 1 or self.vocab < 260:
            raise ValueError("Positive dimensions and the structural vocabulary required")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Width must divide into heads; dropout must be in [0,1)")
```

Here B means batch size, T time positions, D model width, H heads, F feed-forward width and V vocabulary size. Defaults are L=2, D=64, H=4, F=256, T≤64, V=260. Each head receives d=D/H=16 features. An invalid division is a configuration error, not something to discover later in a reshape.

Run this now:

```bash
python -c "from gpt_config import Config; c=Config(); print(c); print(c.width//c.heads)"
```

Expect the defaults and 16. Then try `Config(width=63)`: it should fail. Restore the defaults; do not train a different model just to hide a shape error.

[[practice:heads]]

### Preserve the P00 handoff without a hidden dependency

Create `docs/p00-handoff.md`. Record the P00 source revision plus dirty-file hashes, tokenizer schema/hash and tests, and the learner exit evidence. State which code you copy and why. The next lesson includes a self-contained snapshot of the tutorial's P00 tokenizer for study. If using your own completed tokenizer, check that artifact/API compatibility first. Never import through `../../transformer-foundations-lab` or load P00 encoder–decoder weights as a GPT checkpoint.

Create `docs/p01-exit.md` from the [readiness rows](project/readiness.md). Keep author checks and your own checks separate. There is no need to finish the record today; its first entry is the actual environment and first passing configuration check.

You now have a place to save results and a model size that is small enough to measure. The next unresolved problem is what text the model is allowed to learn from, and how to preserve that decision across runs.

[Continue to Lesson 3](lessons/03-text-and-tokenizer.md) · [Project implementation](project/implementation.md)

### Complete files from this lesson

[gpt_config.py](examples/gpt_config.py)
