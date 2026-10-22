# Tracker — Waste Material Classifier vision README

Date: 2026-10-15 (America/Toronto)

| Task | Status | Evidence |
| --- | --- | --- |
| V01 Context/source review | Done | UCI metadata; PyTorch operators/loss/transfer references; residual paper; local ownership and existing ML records |
| V02 README and overview | Done | Separate 388-line project README and updated collection route, table, tree, boundaries and completion criteria |
| V03 Canonical summaries | Done | Resume/progress/roadmap, D059 and six compact entry-point summaries |
| V04 Documentation checks | Done | 17 relative README links; snapshot links; independent patch/gradient/count/shape/CE checks; fences/whitespace; scoped git diff --check |

No implementation, data/weights, environments, model training, app, launcher/TUI or publication authorized by this README request. Preserve unrelated edits and the two classical ML READMEs.

## Verification and limitations

- The 3x3/2x2 cross-correlation returns [[6,8],[12,14]], sum 40; central differences agree with filter gradients [[12,16],[24,28]]. Changing the first weight by 0.1 changes the objective by 1.2.
- Proposed MLP parameter count: 1,574,153. Proposed CNN: 448 + 4,640 + 18,496 + 585 = 24,169. Spatial shape trace checked independently.
- Published class counts sum to 4,752. Uniform nine-class CE is approximately 2.197225, with target gradient -8/9 and others 1/9.
- Main README explicitly distinguishes the third application from the user's second learning stage. Sequence/embedding topics remain separate; no test/model/hosted claims were advanced.
- Project folder contains only README.md and no nested repository. Existing classical ML project READMEs were not edited.
- Source references identify external design documentation, not installed package versions or measured implementation results.
- No website code/reader route changed; no runtime/site/training tests were run for this documentation-only delivery. Arithmetic and link checks validate authored material, not the unimplemented network.
