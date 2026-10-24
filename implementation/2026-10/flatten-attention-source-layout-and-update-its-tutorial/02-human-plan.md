# Requested organization and completion criteria

Group code by responsibility under src: core for attention/model/config, data for tokenizer/dataset logic, training for trainer/checkpoints, cli for runnable commands and utils for graphs/controls. Keep notebooks outside src and import the same implementation. Preserve the attention calculation while replacing all active old paths and package imports.

The migration is complete only when source, installed wheel, workspace launcher, notebooks, source restoration, tutorials and reader links work with the new structure. Verify known numerical behavior, full notebook execution, package contents, ignored runtime boundaries and desktop/mobile docs. Keep the existing detailed tutorial depth and preserve unrelated edits/results. No future model training is part of this request.
