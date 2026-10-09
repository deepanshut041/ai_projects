---
title: "Retrieval & Evaluation Lab"
description: "A deterministic lexical retrieval baseline with explicit binary-relevance metrics."
updated: "2026-10-09"
status: "implemented example"
tags: ["implementations", "retrieval-lab"]
sources: [{"title": "scikit-learn: Text feature extraction", "url": "https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction"}, {"title": "scikit-learn: Model evaluation", "url": "https://scikit-learn.org/stable/modules/model_evaluation.html#ranking-metrics"}]
---

## Working contract

The retriever fits smoothed IDF on the indexed documents, weights word counts, normalizes vectors and scores cosine similarity. Query-time vocabulary does not modify the index. Zero-score documents are omitted; ties use document IDs for deterministic ordering.

## Run and inspect

```bash
./ai test retrieval-lab
./ai run retrieval-lab
```

The corpus has four original toy documents and three original queries. Metrics are recall at k, reciprocal rank at k and binary NDCG at k. Recall divides retrieved relevant documents by the number of relevant documents. Queries without positive relevance judgments are rejected instead of silently changing the denominator.

## Evidence and limits

Tests use hand-calculated IDF, cosine and ranking metrics, and cover empty queries, out-of-vocabulary terms and deterministic ties. This baseline does not generate answers. Toy-query success does not establish retrieval quality on a real corpus or fairness across query groups.
