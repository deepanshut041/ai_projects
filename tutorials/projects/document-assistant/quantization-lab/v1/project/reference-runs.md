# Author/reference execution record

All results here concern authored teaching components and the existing read-only scaffold, not Gemma inference, trained models, acquired QA data or learner answers.

## Executed checks

- 24 complete standalone reference files were run from a disposable directory in teaching order; the C++ file compiled under C++20 with -O2 -Wall -Wextra -Werror -pedantic. Each displayed listing matched the downloadable file exactly.
- 66 separate audit assertions passed: exact rational layer products and changed input, affine endpoints, independent packing arithmetic, ternary losses, constrained quadratic loss, cache/residual counts, numerical softmax derivatives, empty/ragged calibration rejection, denominator invariance, invalid quantizer ranges, no-feasible-candidate behavior, 24 listing matches, 24 extracted executions, fresh-process toy reload and actual scaffold action checks.
- The actual project interpreter ran status/check/show-config successfully from /tmp; answer returned the expected not_implemented status and exit 2. No model/output was created.

`evidence/reference-outputs.json` retains exact illustrative JSON. `evidence/numerical-audit.json` retains every scoped audit assertion and extracted-file digest. The schema runner's `example-checks.json` records the current final example bytes and outputs after declared-example execution. Reader native tests passed 21/21. The complete browser suite passed for 208 records, existing project/media flows and code rendering. Focused root and /project_docs/ checks each passed all 34 course pages, 24 lesson practice controls, 195 internal link checks, equation rendering, 320/390/1440-pixel containment, ten figure label-bound checks and print generation. All ten figures were visually inspected. These checks do not change model/learner states. Detailed receipts remain in the owning work package.

## Repeat commands

From /mnt/projects/ai:

```bash
./ai learn examples --lesson projects/document-assistant/quantization-lab/v1
./ai learn validate --lesson projects/document-assistant/quantization-lab/v1
./ai learn render --lesson projects/document-assistant/quantization-lab/v1
```

The examples command uses a frozen controller environment and runs declared files. For a downloaded individual file, use its lesson's scratch-directory command. The optional live scaffold trace additionally needs the actual project environment/root; its portable no-argument run does not claim a live checkout check.

## What has not run

No Gemma source weights, converter, imatrix collector, AWQ/GPTQ exporter, cache quantizer, task benchmark or model-backed application was executed. Conditional future commands are labeled in the lessons. The HTTP and answer adapters identify themselves as fixtures; their successful response establishes only their stated boundary behavior. Model compatibility, quality/memory/latency results, application/restart replay and hosted execution remain pending.
