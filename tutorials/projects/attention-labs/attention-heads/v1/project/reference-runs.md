# Author numerical references

These are computed toy examples, not model results or learner answers. Run from this package directory. PyTorch 2.14.1+cpu was already installed in the existing Attention project environment; no dependencies or model data were downloaded for author verification.

```json
[
  {
    "id": "example-multihead",
    "path": "examples/multihead.py",
    "output": {
      "head1_weights": [
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
          0.422319,
          0.155362,
          0.422319
        ]
      ],
      "head2_weights": [
        [
          1.0,
          0.0,
          0.0
        ],
        [
          0.268941,
          0.731059,
          0.0
        ],
        [
          0.155362,
          0.422319,
          0.422319
        ]
      ],
      "concatenated": [
        [
          1.0,
          1.0
        ],
        [
          1.0,
          -0.462117
        ],
        [
          1.422319,
          -0.266956
        ]
      ],
      "output": [
        [
          0.5,
          1.5
        ],
        [
          1.231059,
          0.037883
        ],
        [
          1.555797,
          0.444203
        ]
      ]
    }
  }
]
```

The separate packed PyTorch program was run on CPU float64 and checked output and query-gradient agreement with an explicit loop, plus invariance of earlier outputs to a changed future token. Detailed independent verification is saved in the owning implementation record.
