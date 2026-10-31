# Programs and checks behind the rewritten explanation

The text revision uses one four-position fixture throughout. Its ten visible Python programs are saved under examples/walkthrough/ in teaching order. Each includes its imports, data and called local helpers and can run in a fresh process. The [setup reference](lessons/02-project-setup.md) gives the environment and commands.

## Numerical checks for this explanation

[arithmetic.py](examples/arithmetic.py) independently calculates the supplied context `[1.4,1]`, query-dependent context `[1.339523,1.000000]`, query gradient `[-0.206595,0.353553]` and post-step loss `0.701457` with Python's standard library. Its derivative is checked by central finite differences. The publisher records its output in example-checks.json.

[verify_component.py](examples/verify_component.py) compares the same fixture with the captured actual component, including explicit/optimized agreement, padding and causal perturbations, cross-attention output length, all-blocked output bias and the same query gradient. It is an author check, not a second required reading route.

All ten displayed Python blocks are separately executed and checked by the teaching-revision audit. The complete final class is compared with the actual source by Python syntax tree, including every helper in its dependency closure. These checks can catch missing definitions, wrong values and code drift. They cannot approve the teaching flow on the learner's behalf.

## Existing component and Colab boundary

The existing project has 35 source/visualization checks and a standalone demo. In the project directory, run:

```bash
uv run --frozen --extra notebook pytest -q
uv run --frozen python src/attention.py
```

The Colab runner contains six operational code cells and saved outputs are cleared. Its setup/demo/check workflow has prior local verification; actual hosted execution and persistent hosted recovery remain unverified. This text revision does not change its behavior or claim a translation result.

Earlier course and media checks remain historical evidence in the authoring records and local recovery archive. The old media has been detached from these reading pages. The 26 October 2026 deliverable is the complete revised written route. No replacement videos are produced or attached.
