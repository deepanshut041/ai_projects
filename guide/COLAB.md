# Colab training and recovery contract

Updated: 2026-10-09
Status: requirements recorded; model notebooks, training and hosted recovery checks are pending. Current project: P00 Tokenization and Transformer Foundations / P00-M1 in-progress (Lesson 2 local setup complete); P01 is the later Small GPT Language Model. Read [resume.md](resume.md), [roadmap.md](roadmap.md) and [START_HERE.md](getting-started.md#first-project) before implementation.

## Free first, Pro later

Develop and check models locally; run bounded training through Google Colab. Initially **P00–P04** (tokenizer/Transformer foundations, small GPT, GPT-2, Llama 2/3, Qwen Dense) must supply a Colab Free learning profile and recovery workflow. These are complete miniature architecture implementations, not full-size pretrained models. Each named variant needs coverage. P00 joins the earlier group because it ends in a complete trained conditional model. The tier choice can change with measured evidence. P05 onward gets a compute decision at kickoff; Free feasibility is not assumed for the whole roadmap.

Colab resources and timeouts vary; GPUs are not guaranteed. VMs are temporary. Free notebooks may run for at most 12 hours depending on availability and usage; design much shorter sessions. Paid plans also have resource constraints. These are service facts from the [Google Colab FAQ](https://research.google.com/colaboratory/faq.html), checked 2026-10-09; the budgets and recovery protocol below are workspace design requirements.

### Configuration profiles

| Profile | Purpose | Required behavior |
| --- | --- | --- |
| Local correctness | Tiny FP32 CPU model and independent numerical checks | Runs without Colab, Drive or CUDA; remains the first correctness gate. |
| Colab Free | Complete small model, bounded data/token budget, CPU smoke and optional single-GPU training | Same model/trainer as local; ordinary PyTorch operations, modest batches/context and no mandatory custom CUDA kernels, paid APIs or distributed setup. |
| Architecture reference | Published dimensions and deliberate departures | Metadata only; never selected or allocated implicitly. |
| Colab Pro, later | Larger measured training budget | Same trainer, checkpoint schema and resume flow; add only when requested. |

Keep C++/Python learning examples and native correctness paths. Python/PyTorch supplies the first Colab training entry point. Optional C++ extensions need a reproducible Colab build before they can become required; they must not block the reference training path.

Start P00's model smoke with 1 encoder + 1 decoder layer, width 32, 4 heads, FFN 128, source/target limit 32 and V260; its proposed Free profile is up to 2+2 layers/width 64/limit 64/V≤512. P01's correctness model remains 2 decoder layers/width 64/context 64/V260, with the actual frozen vocabulary in its BPE profile. Scale toward the saved small CPU model only after measuring. Keep architecture mechanisms intact when sizing a Free profile. At each project kickoff record actual parameters, system RAM, GPU/VRAM if available, peak memory, step time, dataset/token budget, checkpoint bytes and persistence time. Proposed budgets are not hardware-fit evidence.

## Child-owned artifacts and notebook flow

Each early independent model repository must include:

- `notebooks/colab.ipynb`: a thin, output-cleared setup/train/resume/evaluate notebook.
- `configs/`: local correctness, Colab Free and architecture-reference profiles; Pro later.
- One importable trainer and explicit new-run/resume options shared by local and Colab entry points.
- A documented, tested Colab dependency setup alongside the local lock, plus a captured runtime environment manifest. Check the actual Python/PyTorch/CUDA environment; a local CPU wheel lock is not Colab GPU compatibility proof.
- A README quickstart, data acquisition/provenance instructions, checkpoint/recovery tests and a small authored verification report.

Create these with P00-M5 model training/recovery and again for P01. The P00 child now exists with Lesson 2 local setup only; both notebooks, the complete P00 model and the P01 repository remain pending. Tokenizer-only exercises need artifact determinism/save-load checks; hosted training/recovery gates apply to the complete P00 model, not every tokenization exercise. The notebook must run top-to-bottom on a fresh runtime and also recover an existing run without depending on hidden cell state:

1. Obtain the exact child source, install its declared dependencies and print source/config/environment identity. Call commands with argument arrays, explicit working directories and checked exit status.
2. Let the user mount Drive and select a persistent run root and stable run ID. Verify read/write access, storage space and the intended run's manifest before training.
3. Detect resources. Run a CPU smoke check when no GPU exists; stop clearly if the requested training budget requires unavailable hardware. Do not silently enlarge or replace a config.
4. Acquire/version data and tokenizer with license, splits and hashes; stage bounded training data on runtime-local disk. Use Drive for recovery artifacts and infrequent transfers. Google recommends reducing small Drive I/O operations in its [FAQ](https://research.google.com/colaboratory/faq.html).
5. Choose **new** or **resume** explicitly. An existing run directory must not be overwritten by a new run; resume must not silently become a new run if no valid checkpoint exists. Print the restored update/token count and last verified persistent checkpoint.
6. Train for one bounded session through the common trainer, with live metrics and periodic persistent saves. Evaluate/sample through the same code.
7. On a normal stop, save and verify once more. Display the recovery location, run ID, update count and restart instructions. Periodic saves remain essential when a termination gives no shutdown opportunity.

Projects implementation folders are ordinary parts of ai_projects; no hosted source publication is assumed. Documentation and Research submodules retain local snapshot remotes. Before remote publication is separately requested, provide a reproducible child-source ZIP upload route: safe extraction, recorded revision/dirty changes and archive hash, with dependencies/data/weights/credentials excluded. Pin a hosted commit when an accessible remote is later authorized. Do not require the entire workspace, its Linux supervisor or developer-specific paths in Colab. Track only source/notebooks/configs/provenance; keep datasets, checkpoints, weights and generated runs out of Git and notebook outputs.

## Bounded sessions and full training state

Proposed initial defaults, finalized from each child pilot: stop a session at **60 minutes or 1,000 additional optimizer updates**, whichever comes first; request a checkpoint after **5 minutes or 100 completed updates**, whichever comes first, and at normal end. Save more often in short recovery drills. Record save/upload latency and actual maximum rollback; an unusually long update or failed transfer can exceed the interval. Stop if persistence fails rather than continuing a long run protected only by runtime-local files. Use no keepalive or timeout-bypass mechanism.

Keep the experiment's total-update/token goal and LR schedule fixed across session slices. Session wall-time/additional-update limits control when to pause, not a fresh optimizer or restarted schedule. Log global update, tokens/examples consumed, loss, LR, elapsed time, memory, session ID and persistent-save status. Make resumed metrics identifiable by run/global-update/session so replay after rollback is not mistaken for new progress.

A resume checkpoint is versioned and includes:

| State | Recovery requirement |
| --- | --- |
| Model and optimizer | All weights/buffers and optimizer moments/parameter mapping. Weight-only exports are inference artifacts. |
| Schedule and precision | LR scheduler, any AMP gradient scaler, precision mode and effective-batch/accumulation settings. |
| Progress and data | Global optimizer updates, tokens/examples, epoch, sampler/shuffle state, next batch position and any streaming cursor. |
| Randomness | Python, NumPy when used, PyTorch CPU and applicable CUDA RNG states, including any explicit data/model generators. |
| Identity | Schema, resolved config, tokenizer/data/split hashes, source revision plus dirty-source identity, dependency versions, hardware and deterministic settings. |
| Monitoring | Best validation metric/checkpoint identity, early-stop state if used, and metric/event position. |

[PyTorch's checkpoint tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) establishes the need to save optimizer state as well as weights. The additional fields above are this workspace's recovery contract. Separate `latest` resumable state from `best` evaluation weights; label a best checkpoint explicitly if it is resumable.

Initially save only after a completed optimizer update, with no pending accumulated gradients and a defined next batch. Use a resumable deterministic data order with `num_workers=0` for the first parity drill; adding workers/prefetch requires accounting for consumed versus queued data. Interruption inside an accumulation window rolls back to the last completed persistent checkpoint. Mid-window saving is optional only with gradients, accumulation position and data/RNG state tested together.

## Persistent publication and restart

Runtime-local `/content` is scratch space. The recovery copy must live in the configured Drive run directory, such as `MyDrive/ai-runs/<project>/<run-id>/`. Data caches can be recreated; optimizer progress cannot. Keep source/config/environment and tokenizer identity with the run, plus compact metrics and evaluation samples.

1. Serialize a stable snapshot locally to a temporary file; close and validate it, then finalize locally. Copy it to a **new immutable checkpoint generation** in persistent storage; never move away or overwrite the only good checkpoint.
2. Write its manifest with expected byte sizes, SHA-256 hashes and progress. Re-read the persisted bytes and validate checksums/loadability. Write a completion record last; partial generations are not recovery candidates. The record is useful only when its manifest and every required file validate.
3. Treat a latest pointer as advisory. Recovery scans completed generations, verifies them and selects the newest compatible valid state; a corrupt/missing newest generation must fall back to an older verified one with an explicit rollback report. If none validate, stop. Do not rely on atomic rename semantics for mounted Drive or claim a checksum proves storage survival across a deleted runtime.
4. Retain at least two recent verified full checkpoints and a best artifact where space permits. Rotate only after a new generation verifies, protecting the last good copies. Measure the storage budget and stop with an actionable message if required retention cannot fit.
5. In a fresh runtime reacquire exact source/dependencies/data, mount storage, validate identity and restore all state on the selected device. Reject incompatible schema/config/tokenizer/data; migrations or deliberate warm starts are separate operations. Restore optimizer tensors to the active device and RNG after setup, before fetching the next batch.

CPU/GPU loading must be device-aware. Same-environment deterministic parity is separate from cross-device continuation: [PyTorch's reproducibility guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md) does not guarantee identical results across platforms/releases or CPU/GPU. Record any device/precision change and its numerical check; unsupported RNG-state mappings must fail clearly or enter a documented, explicitly selected non-identical continuation mode.

## Acceptance before an early project is called Colab-ready

| Check | Evidence required in the child package |
| --- | --- |
| Local correctness | Model/numerical checks plus checkpoint reload in a new process. |
| Resume parity | Uninterrupted N updates versus K + reload + N−K: batch order, LR, optimizer/progress and weights agree under declared same-environment deterministic settings and tolerances. |
| Interrupted write | Kill during save/copy; incomplete generation is ignored and the older verified checkpoint still loads. Test corrupted newest files and failed persistence too. |
| Interrupted training | Abruptly stop between saves or inside accumulation; resume from the last verified update with a measured, reported rollback. |
| Fresh Colab Free runtime | Clear notebook execution from source/data/setup to training, verified Drive save, then **Disconnect and delete runtime**, reconnect and resume the same run. Record actual resource/version/config identity and restored/advanced updates. |
| Portability and guardrails | CPU fallback smoke, device-aware load when available, incompatible identity rejection, explicit new/resume behavior and retention/storage failure handling. |
| Budget | Actual session duration, updates/tokens, peak RAM/VRAM, checkpoint size/persistence time and recovery loss; distinguish fitting from held-out quality. |

Only successful hosted evidence establishes Colab readiness. Local tests/builds establish local behavior; saving to a temporary local directory does not prove Drive durability. Learner explanation, model training, Colab recovery and publication remain separate states in [progress.md](progress.md).

Move to Pro when measured memory, throughput or Free availability prevents the next agreed learning exercise. Record the need and proposed budget, then wait for the user's instruction to use paid compute. Preserve the same checkpoints, thin notebook and bounded resume protocol. Neither an upgrade nor a model name establishes full-scale pretraining feasibility.
