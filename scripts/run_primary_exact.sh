#!/usr/bin/env bash
# Run from the original itsnet project root, with its existing environment active.
set -euo pipefail
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
result_dir="${1:-runs/m4_final_exact_v1}"
if [ -e "$result_dir" ]; then
  echo "Choose a new output directory; already exists: $result_dir" >&2
  exit 1
fi
python3 scripts/audit_inference_exact.py \
  --checkpoint runs/frozen_m4/frozen_m4_checkpoint.pt \
  --manifest runs/frozen_m4/freeze_manifest.json \
  --out "${result_dir}_audit.json"
python3 scripts/eval_openworld_final_exact.py \
  --input data/openworld/unite2025_splits.tsv \
  --frozen-dir runs/frozen_m4 \
  --eval-batch-size 256 \
  --outdir "$result_dir"
python3 - "$result_dir" <<'PY'
import csv
from pathlib import Path
import sys
outdir = Path(sys.argv[1])
with (outdir / "final_predictions.tsv").open() as inp, (outdir / "m4_test_cosine.csv").open("x", newline="") as out:
    reader = csv.DictReader(inp, delimiter="\t")
    writer = csv.writer(out)
    writer.writerow(["query_id", "view", "cosine"])
    seen = set()
    n = 0
    for row in reader:
        key = (row["query_id"], row["view"])
        if key in seen:
            raise RuntimeError(f"Duplicate query/view in corrected predictions: {key}")
        seen.add(key)
        writer.writerow([*key, row["max_cosine"]])
        n += 1
print(f"Exported {n} corrected rows to {outdir / 'm4_test_cosine.csv'}")
PY
