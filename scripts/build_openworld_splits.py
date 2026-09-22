#!/usr/bin/env python3
"""Build leakage-resistant open-world splits from a taxonomy TSV.

Required columns: id, family, genus, species
Other columns (sequence, ITS1/ITS2/core views, higher taxonomy) are preserved.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter

from itsnet.openworld.split import (
    OpenWorldSplitConfig, TaxonRecord, build_openworld_splits, validate_split_invariants,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="taxonomy/view TSV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    with open(args.input, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    required = {"id", "family", "genus", "species"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"input must contain columns: {', '.join(sorted(required))}")

    records = [TaxonRecord(r["id"], r["family"], r["genus"], r["species"]) for r in rows]
    cfg = OpenWorldSplitConfig(seed=args.seed)
    split = build_openworld_splits(records, cfg)
    errors = validate_split_invariants(records, split)
    if errors:
        raise SystemExit("split invariant failure:\n  " + "\n  ".join(errors[:20]))

    fields = list(rows[0]) + ["split"]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            if r["id"] in split:
                w.writerow({**r, "split": split[r["id"]]})

    counts = Counter(split.values())
    for name in ("TRAIN_REF", "DEV_KNOWN", "DEV_NOVEL", "CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"):
        print(f"{name:12s} {counts[name]:8d}")
    print(f"wrote {sum(counts.values())} records -> {args.output}")


if __name__ == "__main__":
    main()
