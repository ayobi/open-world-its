## 2026-09-22  Decided before any corrected CAL/TEST metric was computed
1. Embeddings recomputed with exact-length batching (no padding) for primary M4 and the
   leakage-safe retrain; same checkpoints, splits and calibration. Reason: GroupNorm
   statistics and dilated convolutions include padded positions (TEST_KNOWN core,
   bs=1 vs bs=64: mean |d max-cos| 0.022, max 0.146).
2. Historical identity arm rerun with --maxaccepts 0 --maxrejects 0 (true best hit).
Sealed results are kept and reported beside the corrected ones, whichever way they move.

2026-09-23  Decided before any exact-length DEV metric was computed
    The development ladder (M0-M4) and the M4 3x3 matrix will be rescored with
    exact-length inference from the saved checkpoints. No retraining, no reselection.
    M4 was selected against the integration targets on padded DEV values. If under
    exact-length inference M4 misses one or more targets, that will be reported as
    such; the frozen checkpoint and its TEST evaluation are unchanged, because
    selection cannot be redone after TEST was opened.
