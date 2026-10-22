# Plan — Air Quality Forecaster recurrent sequence project README

Date: 2026-10-15 (America/Toronto)
Lead: ai-project-workflow through workspace planning. Supporting workspace tutorial/whiteboard guidance is reused from the prior README preparation; this is reference-design planning, not live learner rehearsal.

## Scope

The user requested the last foundations project covering RNNs/LSTMs and related models, following the prior folder/README creation pattern. Choose Air Quality Forecaster as a concrete non-text sequence application; create its separate in-depth README, update the collection and vision handoff, and reconcile canonical preparation records/six snapshots. No data download, environment, source/model implementation, training, hosted action, CLI/TUI integration, commit or publication.

## Sources and design

- UCI Beijing Multi-Site Air Quality, dataset 501: hourly multi-station real measurements, 2013-03-01 through 2017-02-28, missing observations and CC BY 4.0 provenance. Inspect official archive/schema at implementation kickoff; start one station before a pooled extension.
- PyTorch 2.14 RNN/LSTM/GRU references inspected on 2026-10-15; exact library gate/bias/state conventions will be pinned at implementation, not assumed from a paper formula.
- Forecast PM2.5 from history known at issue time; no live health recommendation/AQI claim, future measured weather or random-window splitting. Missing inputs and missing targets have different handling.
- Baselines -> explicit RNN cell/BPTT -> LSTM/GRU -> controlled context/optimizer/loss/state/TBPTT studies -> direct/recursive/seq2seq horizons -> historical replay/recovery. No attention, text or embeddings before the later router.

## Tasks

| ID | Task | Dependencies | Acceptance |
| --- | --- | --- | --- |
| Q01 | Inspect context and sources | None | Existing sequence and constraints reconciled |
| Q02 | Create deep project README | Q01 | Task/data/window/state/gradient/architecture/experiment/evaluation/app/recovery contracts |
| Q03 | Update overview and continuation records | Q02 | Fourth application/third foundations stage selected; Bike remains immediate next implementation |
| Q04 | Verify and deliver | Q02,Q03 | Local links, examples/gradients/counts/window boundaries, Markdown/whitespace |

## Teaching route

Linear/MLP baselines establish the value of historical lags. A two-step scalar recurrence exposes shared-state gradients before optimized library recurrence. Gated cells introduce retention/update controls with exact library parity. Multi-step forecasts expose target alignment and teacher-forcing/evaluation differences. Stateful/TBPTT experiments follow safe independent-window training, with explicit resets and recovery. Full practical evidence remains pending.
