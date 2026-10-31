# Attention Foundations project preparation

Prepared 30 October 2026. The user requested a basic-attention implementation project goal and boilerplate that works with Google Colab. Extend the existing [Attention Foundations project](../../../../projects/attention-labs/attention-foundations/README.md) in its independent collection; keep MHA/GQA in the sibling attention-heads project.

The goal covers additive, dot-product, scaled-dot-product, raw self-attention, projected self-attention and causal single-head attention, with explicit forward/gradient/mask/dropout evidence. Six phase overviews and six current setup/learning records live in ignored project docs. The shared subtitle/slide-backed tutorials stay under tutorials/projects/attention-labs.

Delivered source includes responsibility folders, direct/module/console entry points, installed TOML configuration, a synthetic input fixture, a pinned independent Python environment, setup tests, package builds and four workspace runners. Mechanism commands fail explicitly as not_implemented without producing a result. The setup autograd probe is L = 1² + (−2)² = 5 with gradient [2, −4]; it contains no attention algorithm.

The [Colab instructions](../../../../projects/attention-labs/attention-foundations/COLAB.md) and output-cleared notebook use a public-source ZIP plus matching archive/member hashes. The notebook verifies all members before extraction, creates an isolated locked environment, runs setup and pytest, and downloads logs/source identity and a scoped receipt. It requires no repository publication or Drive mount.

| Verification | Result |
| --- | --- |
| Frozen environment | Python 3.12.13, torch 2.14.1+cpu, numpy 2.5.4 |
| Scaffold suite | 18 passed; no skipped tests |
| Source/wheel builds | Passed; no private docs or bytecode packaged |
| Isolated installed wheel | Module check, console status and installed configuration pass from an empty working directory |
| Real workspace launches | status, check, config and colab-export exit 0 |
| Colab operational cells | All six execute with actual locked setup and project tests in a fresh local source/runtime; google.colab.files simulated |
| Docs native checks | 24 passed |
| Reader | Root/prefixed desktop/mobile project navigation, six phases, six records, notebook download, tutorial link, Mermaid and private-export boundary pass |
| Actual Google-hosted run | Pending |
| Attention algorithms/correctness | Pending |
| Learner explanation | Unassessed |

Collection source: `ccab1d150ed1a3c57c2aa6855d87fa64b005a875`. The final clean export identifies this commit and hashes 23 public files. Exported pairs live in the ignored project artifacts/colab folder. Author logs and screenshots remain in workspace .ai/attention-foundations-setup/2026-10-30. A source-manifest hash proves consistency of the exported pair, not trust in an unknown sender.

The package review found two data-module cache files had been included by an overly broad source-directory ignore exception. A follow-up collection commit removes the tracked caches and narrows the exception; final source/wheel/archive checks exclude bytecode and preserve the real data fixture source.

Next implementation action: freeze one additive query/key/value retrieval and visibility contract, explain the worked expectation, then implement and independently test it. Actual hosted Colab acceptance, six attention mechanisms and learner review remain pending. Existing P00, Attention Is All You Need, model and hosted checkpoints are unchanged.
