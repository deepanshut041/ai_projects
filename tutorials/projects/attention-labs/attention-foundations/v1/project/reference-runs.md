# Author numerical references

These are computed toy examples, not model results or learner answers. Run from this package directory. PyTorch 2.14.1+cpu was already installed in the existing Attention project environment; no dependencies or model data were downloaded for author verification.

```json
[
  {
    "id": "example-02-fixed-summary",
    "path": "examples/02_fixed_summary.py",
    "output": {
      "states": [
        1.0,
        0.5,
        2.25
      ],
      "changed_final": 2.5,
      "first_input_sensitivity": 0.25
    }
  },
  {
    "id": "example-03-additive",
    "path": "examples/03_additive.py",
    "output": {
      "scores": [
        0.761594,
        -0.761594,
        0.0
      ],
      "weights": [
        0.593494,
        0.129391,
        0.277115
      ],
      "context": [
        0.870609,
        0.406506
      ],
      "new_query_weights": [
        0.541045,
        0.20633,
        0.252626
      ]
    }
  },
  {
    "id": "example-04-scoring",
    "path": "examples/04_scoring.py",
    "output": {
      "scores": [
        1.0,
        1.0,
        2.0
      ],
      "unscaled": [
        0.211942,
        0.211942,
        0.576117
      ],
      "scaled": [
        0.248255,
        0.248255,
        0.50349
      ],
      "mild_diagonal_derivative": 0.104993585,
      "sharp_diagonal_derivative": 2.061e-09
    }
  },
  {
    "id": "example-05-self-attention",
    "path": "examples/05_self_attention.py",
    "output": {
      "scores": [
        [
          1.0,
          0.0,
          1.0
        ],
        [
          0.0,
          1.0,
          1.0
        ],
        [
          1.0,
          1.0,
          2.0
        ]
      ],
      "weights": [
        [
          0.422319,
          0.155362,
          0.422319
        ],
        [
          0.155362,
          0.422319,
          0.422319
        ],
        [
          0.211942,
          0.211942,
          0.576117
        ]
      ],
      "output": [
        [
          0.844638,
          0.577681
        ],
        [
          0.577681,
          0.844638
        ],
        [
          0.788058,
          0.788058
        ]
      ],
      "permutation_equivariant": true,
      "zero_query_weights": [
        0.333333,
        0.333333,
        0.333333
      ]
    }
  },
  {
    "id": "example-06-projections",
    "path": "examples/06_projections.py",
    "output": {
      "q": [
        [
          1.0,
          0.0
        ],
        [
          0.0,
          1.0
        ],
        [
          1.0,
          1.0
        ]
      ],
      "k": [
        [
          1.0,
          1.0
        ],
        [
          0.0,
          1.0
        ],
        [
          1.0,
          2.0
        ]
      ],
      "v": [
        [
          2.0,
          0.0
        ],
        [
          0.0,
          -1.0
        ],
        [
          2.0,
          -1.0
        ]
      ],
      "last_weights": [
        0.283995,
        0.140029,
        0.575975
      ],
      "last_output": [
        1.719942,
        -0.716005
      ],
      "loss": 0.039216379,
      "gradient": -0.047694266,
      "gradient_verified": true,
      "updated_parameter": 1.009538853,
      "updated_loss": 0.038763847
    }
  },
  {
    "id": "example-07-causal",
    "path": "examples/07_causal.py",
    "output": {
      "weights": [
        [
          1.0,
          0.0,
          0.0
        ],
        [
          0.5,
          0.5,
          0.0
        ],
        [
          0.283995,
          0.140029,
          0.575975
        ]
      ],
      "output": [
        [
          2.0,
          0.0
        ],
        [
          1.0,
          -0.5
        ],
        [
          1.719942,
          -0.716005
        ]
      ],
      "future_invariant": true,
      "dropout_sample": [
        0.567991,
        0.0,
        1.151951
      ],
      "dropout_sample_sum": 1.719942,
      "dropout_expectation": [
        0.283995,
        0.140029,
        0.575975
      ],
      "all_masked_rejected": true
    }
  },
  {
    "id": "example-attention",
    "path": "examples/attention.cpp",
    "output": {
      "output": [
        [
          2.0,
          0.0
        ],
        [
          1.0,
          -0.5
        ],
        [
          1.719942,
          -0.716005
        ]
      ]
    }
  }
]
```

The separate packed PyTorch program was run on CPU float64 and checked output and query-gradient agreement with an explicit loop, plus invariance of earlier outputs to a changed future token. Detailed independent verification is saved in the owning implementation record.
