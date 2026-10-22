# Second pass: maths and the P01 handoff

Date: 2026-10-09. Owner: Projects Docs. Status: completed local instructional audit and verification.

## Scope and evidence

Read the actual roadmap P00 M1–M6/P01 required scope, Colab contract, eleven integrated lessons, project implementation checklist and historical reference evidence. Preserve the first-lesson entry and all implementation material. Source and worked-code baseline is saved in ignored `.ai/tutorial-runs/p00-second-pass/before/` at workspace home.

This is instructional preparation. The independent learner repository does not exist; P00-M1 is not-started. The historical author run failed the proposed 90% exact copy gate (10.94%); local full-state recovery is not Colab deleted-runtime proof.

## Work plan

1. Trace each required mathematical idea from background to numbers, code and a learner check. Introduce missing background within the lesson that uses it.
2. Explain scalar-to-vector gradients, loss reduction, probability/logarithms, dot products, variance/normalization, Adam/warmup/clipping, dropout and sequence likelihood. Add a bounded P01 pre-norm/GELU/positions/objective bridge.
3. Give remaining milestone tasks exact file destinations, concrete tests and a project readiness record. Separate mandatory P00 evidence, P01 implementation work and optional historical study.
4. Run an independent maths check, the extracted worked project and source/example validation. Build and exercise the website at root and prefixed URLs, inspect representative desktop/mobile maths.
5. Record observed results and refresh canonical preparation snapshots. Do not claim learner mastery, successful task training or hosted recovery.

## Verification

- Twenty standard-library arithmetic checks pass, including finite differences, target gradients and exhaustive independent-coordinate variance enumeration.
- Eight PyTorch comparison groups pass on 2.14.1+cpu: CE gradients, Adam/state, global clipping, actual warmup indexing, LayerNorm, GELU, dropout and ignored-label reduction. All five new standalone inline Python blocks produce their stated outputs.
- Twenty-three cumulative/combined worked-project tests pass; the generated eight-cell notebook is syntax-checked and output-cleared. Hosted cells were not executed.
- Five declared C++/Python examples pass; source validation/render succeeds. Twenty-six existing worked files retain identical parsed code; nineteen existing examples/diagram assets retain identical bytes. Eleven lessons, eleven diagrams, ten tables and four scenarios remain. Practice grows from 25 to 31; source records from 14 to 22.
- Eight native site tests pass. Root and `/project_docs/` builds/browser suites pass across 37 records and 1440/768/390/320 widths. Ten targeted maths/readiness views and the lesson-to-readiness link pass. Representative desktop/mobile screenshots were visually read; no equation errors or page overflow were found.
- The final matrix-gradient addition and code-power notation correction received another snippet run, render and targeted page check; final build fingerprints are recorded in second-pass-checks.json.

Commands from workspace home: `./ai learn examples --lesson <absolute lesson path>`, `./ai learn render --lesson <absolute lesson path>`, and `python3 .../verify-build-guide.py --lesson ... --output .ai/tutorial-runs/p00-second-pass/worked-project --python .ai/envs/p00-tutorial/bin/python`. Site checks: `npm test`, `npm run build`, `npm run test:browser` from the website; root hosting uses an ignored independent docs copy and `SITE_BASE=/`. Runtime comparison script and disposable outputs are in `.ai/tutorial-runs/p00-second-pass/`; selected safe evidence is in implementation/evidence/2026-10-09-p00-second-pass/.

## Coverage decisions

| Gap found | Change and practical application |
| --- | --- |
| Symbols and calculus assumed before use | Lesson 3 notation/limit/partial/log background; Lesson 4 multi-parameter and matrix-layer backward trace |
| Loss formulas disconnected from data | Lesson 6 conditional sequence likelihood, valid-token denominator and data-space counting; Lesson 9 perplexity and metric distinctions |
| Attention/norm maths too compressed | Lesson 7 radians, dot-product magnitude, expectation/variance and scaling derivation; Lesson 8 stepwise LayerNorm |
| Optimizer controls named without enough arithmetic | Lessons 4/9 Adam moments and bias correction, clipping, actual warmup update rates and dropout expectation |
| P01 architectural names unexplained | Lesson 11 next-token window, pre-norm versus post-norm zero-sublayer example, learned positions, Gaussian/GELU background, initialization, AdamW and temperature |
| Remaining learner tasks vague | Exact character-baseline file/functions/tests; model-boundary fixtures; ordered retention/storage/failure-injection implementation guidance |
| No single project exit review | project/p01-readiness.md maps every P00 milestone and required maths to evidence, including failed/unknown states and the required versus optional P01 scope |

## Limits and next action

No unexplained prerequisite gap was found in this bounded pass against the current P00/P01 roadmap after these additions. That is an editorial coverage judgment, not a guarantee of learner understanding. Existing exercises still require implementation and explanation. The historical copy gate failed and hosted recovery is unverified. Begin with Lesson 1, continue through the integrated route, and use the exit record to decide when actual P00 evidence supports P01 kickoff.
