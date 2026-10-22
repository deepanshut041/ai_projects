# Tutorial implementation and validation contract

Updated: 2026-10-09. Lead: workspace developer-learning-tutorials; supporting planning/project workflow and feature-whiteboard-defense. Owner: Projects Docs. Artifact work, no live assessment/delegation.

## Delivered structure

Versioned source: tutorials/projects/p00/transformer-foundations/v1/. Twelve chapters cover M1–M6; lesson.json owns eight objectives, eleven semantic SVG boards, ten tables, four exhaustive declarative scenarios and sixteen practice reveals. Workbook supplies concrete learner deliverables/checks; whiteboard dossier and thirty questions connect mechanisms/evidence. Preserve historical P01 v1.

Six downloadable C++/Python programs: standard-library numeric traces in both languages; deterministic byte-BPE; tensor/autograd/MLP; complete PyTorch encoder–decoder/trainer; fresh-process full-state resume checker. The three dependency-free examples use the existing learn examples runner; PyTorch programs execute separately in ignored CPU environment. Linked source files participate in fingerprint/export. No new reader/schema/language.

## Numerical/model contracts

FP32 tiny model: V260, D32, H4, F128, one encoder and decoder, context32, post-norm, sinusoidal positions, ReLU, scaled embeddings, untied tables/head. Manual attention/LayerNorm; independent SDPA/LayerNorm references and FP64 gradcheck. Byte IDs 0–255; structural 256–259; BPE 260+. Exact UTF-8, document boundaries, rank encoding, lexicographic pair ties, no normalization. Unique records split before tokenization/batching. Shift BOS/content input versus content/EOS labels; ignore PAD target denominator. Allow True masks over key axes; nonempty rows; source/full visibility, target causal visibility; padded-query output policy.

Fresh-request greedy generation with source encoded once, no KV cache, structural exclusions, EOS per row and bounded context. Training state contains model/optimizer/scheduler/sampler/RNG/config/source/runtime identity at complete update boundary. Immutable verified generations, completion last, fallback/error. Educational reference retains generations; storage retention/failed-copy/hosted deletion gates remain child work.

## Acceptance and scope

Analytic numeric/C++ checks; Unicode/BPE rank/boundary/determinism/artifact checks; tensor and tinyMLP fit; independent attention forward/backward, LayerNorm gradcheck, model causality/PAD/source/gradient/EOS/reset checks; dropout CPU resume parity plus fresh-process full-state parity and corrupt/incomplete/config failure checks. 600-update author copy demonstration is preserved as a failed task gate, not a training success claim.

Structural validate/examples/render; site npm test/build/browser. Generalize prior tests to every authored lesson and match scenario IDs to DOM order. Test every declared case, all reveals, print/download/asset links, desktop/mobile containment. Capture standalone SVGs for full-board visual review, not only their clipped scrolling containers. Inspect arrow meaning and readable labels.

## Remaining learning work

Independent child setup/lock, learner implementation/explanation, successful declared task gate, common hosted notebook, storage retention/failed persistent saves and Drive recovery after runtime deletion. No commit, push, publication, paid service or hosted run authorized or performed.
