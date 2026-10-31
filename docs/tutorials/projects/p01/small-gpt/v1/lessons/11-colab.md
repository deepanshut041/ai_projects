## Lesson 11 — continue the same trainer on Colab Free

A terminal resume proves local persistence. Colab adds a disposable runtime and mounted persistent storage. The crucial question is whether the run can continue after the runtime itself has been deleted, using only recorded source, data and persistent state.

### Keep the notebook thin

Free resources vary; there is no promised GPU model or session duration. Use the small profile first, CPU correctness before GPU training, and measured budgets. The [Colab FAQ](source:colab) also advises reducing many small Drive operations. Stage the dataset locally under `/content`; use Drive for infrequent complete recovery generations rather than every batch read.

The notebook will perform setup, call the common trainer, and show its results. It does not implement attention or duplicate the optimizer loop. The current workspace remotes are local snapshots, so the source delivery route is a verified ZIP upload; no GitHub publication is required.

### Export only the files needed to reproduce the run

Save [export_source.py](examples/export_source.py) as `export_source.py` in your learner root. It contains an explicit allowlist of source/test/config files. Read the list first: no raw data, environment, checkpoints, credentials or notebook outputs enter the ZIP. It records the Git revision when available and hashes actual file bytes, so dirty changes remain identifiable.

Run `python export_source.py exports/p01-v1`. The new directory contains `p01-source.zip`, `source_manifest.json` and an output-cleared `colab.ipynb`. Copy the notebook to your project's `notebooks/colab.ipynb` for tracking; keep generated ZIPs ignored. Re-export to a new directory after changing source and record the new archive hash; a changed source identity cannot strictly resume an old run without an explicit reviewed migration.

The generated notebook checks its expected ZIP hash, rejects duplicate/path-escaping/link entries, caps expanded size and verifies each allowlisted file hash before writing source. This prevents an upload with the wrong files from masquerading as the recorded experiment. Notebook source is still inspectable before any execution.

### Walk through the notebook cells

| Cell | Concrete action and result |
| --- | --- |
| Upload | Choose the source ZIP that belongs to this notebook |
| Verify/extract | Check archive/member hashes; create a fresh `/content/p01-source` |
| Install/test | Install the declared Torch version; print actual Python/Torch/CUDA/device; run the twenty-one local correctness tests |
| Mount/acquire | Mount Drive, test read/write, reacquire pinned data on local disk, select a persistent run directory |
| Train/resume | Choose mode explicitly and call `gpt_train.py` with argument arrays, limits and the same profile |
| Sample/report | Call the same sampling program and print the recovery location |

A fresh runtime should execute these cells top to bottom. Re-running extraction/acquisition in the same runtime intentionally refuses existing destinations; resume after deletion begins in a fresh filesystem. If you need to repeat only training in an existing runtime, skip setup cells whose outputs you have already verified rather than erasing a directory blindly.

The source requires the inspected Torch 2.14.1 API. Installation/notebook syntax were checked locally; the actual Free Python/CUDA environment and wheel compatibility must be verified in your hosted session. If the runtime version differs, record and reconcile the difference before creating a run. Strict identity rejection on resume is expected when environment/device changes; do not bypass it and then claim deterministic continuation.

### Choose a real persistent run identity

Use a stable path such as `MyDrive/ai-runs/p01/first-text-run`. The run ID names the experiment across sessions; it is not a disposable timestamp generated every time the notebook opens. The generated notebook defaults to `mode="new"`; after a deleted-runtime restart you must set `mode="resume"` explicitly. A CPU fallback can run a new smoke when no GPU is available. It does not silently convert a saved CUDA run into a different training trajectory.

Keep the declared experiment at 300 total updates for the first example, with a 100-update session slice. The workspace's upper initial session bounds are 60 minutes or 1,000 additional updates, with a save requested after five minutes or 100 updates and on normal stop. A smaller recovery drill can save every 25. Record save duration: copying a large state may dominate a tiny model's work.

### Perform the deleted-runtime drill

1. Run tests and a new bounded session, then inspect a verified Drive generation and its actual update/token counts.
2. Record source archive, data/tokenizer/config hashes, runtime/device, checkpoint bytes and path. Download a compact report, not the model into Git.
3. Use **Disconnect and delete runtime**. Merely rerunning a cell or restarting Python does not remove all runtime-local files.
4. Open a fresh runtime, upload the same source, rerun setup/acquisition, mount the same Drive and choose resume with the same run ID.
5. Confirm the printed restored generation/update, then advance at least one update and verify another persistent generation. Record any rollback or rejected generation.
6. Compare with the appropriate local parity checks. Do not promise bitwise CPU/GPU equality; changes need a separately declared continuation protocol.

[[practice:hosted-proof]]

### Record resource and failure evidence

In `docs/colab-recovery.md`, capture elapsed session time, actual device/RAM/VRAM, peak allocated/reserved GPU memory when available, updates/tokens, checkpoint footprint, save latency, rollback and the recovery command/result. CUDA timings require synchronization around the measured region; an asynchronous launch time is not completed kernel time.

Inject a failed persistence operation in a disposable test run and confirm training stops while an older valid state remains. Keep two recent verified full states plus best within the measured storage budget. Do not mark Colab-ready from the source ZIP, a mounted Drive screenshot or a local-directory round trip. The author has not executed a hosted session for this tutorial.

The final lesson maps these files and measurements back to the project milestones, and gives you a precise boundary for GPT-2.

[Continue to Lesson 12](lessons/12-profile-and-handoff.md) · [Readiness record](project/readiness.md)
