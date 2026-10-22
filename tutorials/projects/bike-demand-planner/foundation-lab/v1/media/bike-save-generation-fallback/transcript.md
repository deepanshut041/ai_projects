# Recover from a damaged save without hiding the fallback

## A readable file can still be the wrong state

Saving every field is not enough if the loader can combine fields from different moments. New weights with old optimizer moments and another cursor describe a state that never existed. Overwriting one live file also risks damaging the only copy when a write is interrupted. The store therefore publishes complete numbered generations, retaining an older valid generation. Each generation contains state, context and a receipt tying their bytes and update index together. The recovery question is which complete compatible state can be trusted now, rather than whether some JSON happened to parse.

## Publish the completed directory together

The writer prepares a unique pending directory next to the intended published directory. Each file is written, flushed and synchronized before publication. It synchronizes the staging directory, then renames that complete directory to its numbered name on the same filesystem. The parent directory is synchronized afterward on POSIX. The loader never chooses a pending directory. Existing destinations are not overwritten, and the checkpoint store assumes a single writer. These local primitives and fault tests do not prove arbitrary storage or power loss behavior. A stale writer lock after a real process failure needs inspection before saving again.

## Make the fault injection inspectable

The demonstration first saves update two hundred and sixty nine, then a second complete generation at two hundred and seventy three. It saves update two hundred and seventy four as generation three, then deliberately appends corrupt bytes to its state file. A separate pending directory contains a truncated state with no completed publication. Those deliberate faults are retained for inspection, rather than hidden by deleting or replacing the bad files. This is an injected damaged and partial save scenario. It is not an observed machine crash or proof of recovery after deleting a hosted runtime.

## Verify each candidate before accepting it

The loader considers published generations from newest to oldest. It reads the receipt and verifies member hashes before accepting their state. Generation three fails because its state bytes no longer match the receipt. The rejection reason is kept. Generation two has matching bytes, source and data context, schema, transform, configuration and consistent numerical state. It is accepted as a whole. The pending directory is listed as incomplete rather than considered for loading. The returned receipt names the loaded generation and the rejected and ignored items, making fallback visible to the caller.

## Rollback the whole state, then recompute the next move

Recovery therefore resumes from update two hundred and seventy three, cursor sixty four, rather than borrowing the corrupted newer weights. The next batch and rate come from that same older state, so update two hundred and seventy four is recomputed consistently. The final numerical state still matches the uninterrupted path at update eight hundred and thirteen. Rolling back one update loses a little work but preserves the computation. Mixing a newer parameter file with an older order or moment file would preserve neither. The receipt lets us distinguish genuine rollback from an unexplained successful looking restart.

## An incompatible old generation is not a valid fallback

Change the expected training hash or feature order instead. An older generation from another context must not be treated as recovery of this experiment. The loader keeps looking for a compatible generation and raises an explicit error if none remains. Unit checks exercise source, data and schema mismatches, inconsistent cursor or schedule, failed state and failure before publication. Passing those checks establishes the implemented local behavior, while arbitrary device failure and hosted recovery remain separate evidence. The useful habit is to keep the loaded identity, fallback reason and scope visible instead of compressing every restart into one success flag.
