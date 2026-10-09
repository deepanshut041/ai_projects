# Plan — Colab Free training and recovery requirements

Develop models locally, then run the same training code using a small child-owned Colab notebook. Initially require Free support for P01–P04; choose Pro later from measured needs. Local correctness remains the first gate.

Make interrupted-session recovery part of training: bounded runs, complete training state, verified persistent checkpoints and explicit resume in a fresh runtime. Save during training because termination may never run shutdown code.

Deliver shared guidance and integrate roadmap, kickoff, setup, canonical status and entry instructions. Verify the documentation website locally. No model exists yet; notebook/code/recovery validation belongs in P01's future child-owned package.
