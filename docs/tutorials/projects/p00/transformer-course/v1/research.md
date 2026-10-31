# Sources and implementation choices

Reviewed: 2026-10-12. This course is authored around five Priyam Mazumdar videos and the accompanying MIT PyTorch-Adventures source at revision `2f18a5470c4ef0a3d413dd7b18d07941f45779e9`. Three describe an English-to-French encoder–decoder; two describe RoBERTa masked-language pretraining and extractive question answering. Lessons follow dependencies and topics rather than video boundaries.

## Primary material

- [Attention](https://www.youtube.com/watch?v=JXY5CmiK3LI): weighted values, learned projections, heads, masks, cross-attention and efficient SDPA.
- [Translation data](https://www.youtube.com/watch?v=UkBggwQsbF8): separate English/French tokenizers, parallel text, shifted targets and collation.
- [Translation model and training](https://www.youtube.com/watch?v=DOqg_QxInd8): positions, encoder/decoder blocks, training and inference.
- [RoBERTa pretraining](https://www.youtube.com/watch?v=Fzv2lSt0Ing): corpus preparation, dynamic MLM, encoder and training.
- [RoBERTa extractive QA](https://www.youtube.com/watch?v=_FVwYFICYuE): span alignment, backbone transfer, QA loss, training and decoding.
- [Pinned reference repository](https://github.com/priyammaz/PyTorch-Adventures/tree/2f18a5470c4ef0a3d413dd7b18d07941f45779e9).
- [Transformer paper](https://arxiv.org/abs/1706.03762), [BERT paper](https://arxiv.org/abs/1810.04805), [RoBERTa paper](https://arxiv.org/abs/1907.11692).

## Current source findings

The translation source uses separate embeddings, sinusoidal positions, GELU FFNs and post-norm residual blocks. The original Transformer paper uses ReLU and embedding scaling; differences are explained where used rather than silently mixed. Its supplied training recipe uses AdamW, cosine scheduling and warmup. Small local profiles are reference correctness exercises, not WMT-scale reproduction.

The inspected RoBERTa source has consequential differences from standard tokenizer/backbone conventions: inconsistent PAD defaults, padding masks formed after padding, a batchwide random-token sampling restriction, plain sequential positions and an untied MLM decoder. Its QA processor/decoder needs explicit half-open span boundaries, window handling and constrained start/end selection. The lessons explain the original idea and use corrected, tested contracts. Exact pretrained-library checkpoint interchangeability is not implied by architectural similarity.

## Evidence boundary

Raw captions/metadata/repository copies are kept only in the ignored workspace source cache. The published course contains authored explanations, original arithmetic/diagrams, source links and digest/provenance notes. Local reference tests and bounded toy runs are author verification; learner implementations/explanations, held-out task quality and Colab deleted-runtime recovery remain separate gates.

## Teaching and verification choices

The shared prerequisites have four detailed lessons on tensors, stable probabilities/gradients/Adam, tokenization and objective/data boundaries. Eight lessons develop attention and complete translation; eight develop RoBERTa and QA. Recovery and the GPT handoff join those tracks into one 24-lesson route. Numerical matrix equations and original semantic diagrams expose each operation's axes. Native retrieval exercises keep questions and worked answers separate; interactive cases change visibility, corruption and valid spans.

The offline translation fixture uses small word vocabularies; the source-style WordPiece adapter is shown and separately checked. RoBERTa uses an own-format byte-BPE implementation with character offsets. It preserves the relevant mechanisms, but has different pretokenization, vocabulary and parameter names from official RoBERTa. No pretrained checkpoint interchangeability is claimed.

Both models train locally using their ordinary Python modules. Shared immutable checkpoint generations verify completion/hashes and restore exact compatible CPU state. The export generates a thin Colab notebook which calls those same modules; hosted resources and deleted-runtime persistence still require a real run. Read [reference results](project/reference-runs.md), [source map](project/source-map.md), and [readiness](project/readiness.md) together. [Acquisition metadata](source-provenance.json) retains safe source/chapter/digest evidence without publishing raw captions.

Source code adapted from the pinned MIT reference retains [its license](examples/LICENSE.txt). New math, diagrams, checks, tiny corpora, corrections and recovery code are authored for this course. Current source inspection and small fixture results justify local teaching examples; they do not establish large-scale reproduction, GPU performance, learner mastery or hosted completion.
