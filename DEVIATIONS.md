## 2026-09-22  Decided before any corrected CAL/TEST metric was computed
1. Embeddings recomputed with exact-length batching (no padding) for primary M4 and the
   leakage-safe retrain; same checkpoints, splits and calibration. Reason: GroupNorm
   statistics and dilated convolutions include padded positions (TEST_KNOWN core,
   bs=1 vs bs=64: mean |d max-cos| 0.022, max 0.146).
2. Historical identity arm rerun with --maxaccepts 0 --maxrejects 0 (true best hit).
Sealed results are kept and reported beside the corrected ones, whichever way they move.
