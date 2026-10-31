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

Sequential revision: head_loop.py registers two full causal heads and output mixing on [2,6,6], with 144 parameters. lecture8_equivalence.py copies every Q/K/V/output coefficient into the packed layer, reproduces all tensor shapes, compares outputs plus input/all-parameter gradients, and intervenes on future inputs. CPU float64 maximum forward difference is 2.22e-16 in the declared run. Browser timing screenshots are illustrative measurements, not a portable performance claim. Actual hosted Colab execution is pending.
