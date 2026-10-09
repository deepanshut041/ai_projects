---
title: "Linear Model Lab"
description: "Train a linear regressor with an intercept and verify its gradient and fitted weights."
updated: "2026-10-09"
status: "implemented example"
tags: ["implementations", "linear-model-lab"]
sources: [{"title": "scikit-learn: Linear models", "url": "https://scikit-learn.org/stable/modules/linear_model.html"}, {"title": "NumPy: numpy.linalg.lstsq", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html"}]
---

## Working contract

Input is a finite matrix of samples by features plus one finite target per sample. Gradient descent minimizes mean squared error, checks a conservative design-dependent step-size bound and returns weights and training-loss history. Prediction rejects a different feature count.

## Run and inspect

```bash
./ai setup linear-model-lab
./ai test linear-model-lab
./ai run linear-model-lab
./ai build linear-model-lab
```

The demo creates 160 original synthetic samples with seed 42, trains on 120, and evaluates on the held-out 40. It reports a constant-predictor baseline and the difference from NumPy's least-squares reference. Tests compare fitted weights with the independent reference and the analytic gradient with finite differences.

## Limits and next extension

This is a dense CPU example with fixed synthetic data and no preprocessing or tuning. Add training-only preprocessing, validation splits and regularization before studying real datasets. Training loss alone does not establish generalization.
