#!/usr/bin/env python3
"""Materialize per-partition, per-view FASTAs from the open-world split table.

The sequences live in data/openworld/unite2025_splits.tsv, in the `core`, `its1`
and `its2` columns, and the partition lives in `split`. No per-partition FASTA
exists on disk, which is why nothing downstream can run yet. This writes them.

A view is emitted for a record only when that view has a sequence AND the
matching *_conflict flag is 0, which is how the manuscript's per-view
availability arises (a record can contribute ITS-core but not ITS2 if its ITS2
sequence was masked as a cross-genus exact duplicate).

The script then checks every partition-by-view count against the numbers already
published in data/m4_dev_retrieval_matrix.csv, data/frozen_test_metrics.csv and
Table 1. If all 18 cells match, the FASTAs are provably the same records the
training runner and the frozen evaluator used, and the hours of VSEARCH that
follow are not being spent on the wrong files. If a cell disagrees, stop and
find out why before running anything.

Usage
-----
    python3 analysis/materialize_fastas.py
    python3 analysis/materialize_fastas.py \
        --splits data/openworld/unite2025_splits.tsv \
        --outdir data/openworld/fasta

Writes
------
    <outdir>/{train_ref,dev_known,dev_novel,cal_known,test_known,test_novel}_{core,its1,its2}.fasta
    <outdir>/../test_query_manifest.csv   query_id, partition, view
    <outdir>/../ref_taxonomy.csv          ref_id, genus, family, order, class
    <outdir>/../query_taxonomy.csv        query_id, genus, family, order, class
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

VIEWS = ["core", "its1", "its2"]
PARTITIONS = ["TRAIN_REF", "DEV_KNOWN", "DEV_NOVEL",
              "CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"]

# Counts already published, used as a self-check. Per-view figures come from the
# deposited metrics files; totals from Table 1.
EXPECTED = {
    "TRAIN_REF":  {"total": 46453, "core": 46151, "its1": 46342, "its2": 46076},
    "DEV_KNOWN":  {"total":  1528, "core":  1515, "its1":  1527, "its2":  1515},
    "DEV_NOVEL":  {"total":  8698, "core":  8659, "its1":  8682, "its2":  8642},
    "CAL_KNOWN":  {"total":  1505, "core":  1494, "its1":  1501, "its2":  1493},
    "TEST_KNOWN": {"total":  1484, "core":  1467, "its1":  1477, "its2":  1463},
    "TEST_NOVEL": {"total":  7738, "core":  7718, "its1":  7702, "its2":  7671},
}
QUERY_PARTITIONS = ["CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"]


def wrap(seq: str, width: int = 80):
    for i in range(0, len(seq), width):
        yield seq[i:i + width]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path,
                    default=Path("data/openworld/unite2025_splits.tsv"))
    ap.add_argument("--outdir", type=Path, default=Path("data/openworld/fasta"))
    ap.add_argument("--no-check", action="store_true",
                    help="write the files even if the count check fails")
    args = ap.parse_args()

    if not args.splits.exists():
        sys.exit(f"split table not found: {args.splits}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    meta_dir = args.outdir.parent

    handles = {}
    for p in PARTITIONS:
        for v in VIEWS:
            handles[(p, v)] = open(args.outdir / f"{p.lower()}_{v}.fasta", "w")

    counts = {p: {"total": 0, **{v: 0 for v in VIEWS}} for p in PARTITIONS}
    other_splits: dict[str, int] = {}
    manifest = open(meta_dir / "test_query_manifest.csv", "w", newline="")
    mw = csv.writer(manifest)
    mw.writerow(["query_id", "partition", "view"])
    reft = open(meta_dir / "ref_taxonomy.csv", "w", newline="")
    rw = csv.writer(reft)
    rw.writerow(["ref_id", "genus", "family", "order", "class"])
    qryt = open(meta_dir / "query_taxonomy.csv", "w", newline="")
    qw = csv.writer(qryt)
    qw.writerow(["query_id", "genus", "family", "order", "class"])

    csv.field_size_limit(10 ** 7)
    with args.splits.open(newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        required = {"id", "split", "genus", "family", "order", "class"} | set(VIEWS)
        missing = required - set(rd.fieldnames or [])
        if missing:
            sys.exit("split table is missing column(s): " + ", ".join(sorted(missing)))
        has_conflict = {v: f"{v}_conflict" in (rd.fieldnames or []) for v in VIEWS}
        if not all(has_conflict.values()):
            print("warning: no *_conflict columns for "
                  + ", ".join(v for v in VIEWS if not has_conflict[v])
                  + "; masking cannot be applied for those views")

        n_spaces = 0
        for row in rd:
            split = (row["split"] or "").strip()
            if split not in counts:
                if split:
                    other_splits[split] = other_splits.get(split, 0) + 1
                continue
            rid = (row["id"] or "").strip()
            if not rid:
                continue
            if " " in rid or "\t" in rid:
                n_spaces += 1

            counts[split]["total"] += 1
            lineage = [row.get("genus", ""), row.get("family", ""),
                       row.get("order", ""), row.get("class", "")]
            if split == "TRAIN_REF":
                rw.writerow([rid, *lineage])
            else:
                qw.writerow([rid, *lineage])

            for v in VIEWS:
                seq = (row.get(v) or "").strip().replace(" ", "")
                if not seq:
                    continue
                if has_conflict[v] and (row.get(f"{v}_conflict") or "0").strip() == "1":
                    continue
                counts[split][v] += 1
                fh_out = handles[(split, v)]
                fh_out.write(f">{rid}\n")
                for line in wrap(seq):
                    fh_out.write(line + "\n")
                if split in QUERY_PARTITIONS:
                    mw.writerow([rid, split.lower(), v])

    for h in handles.values():
        h.close()
    for h in (manifest, reft, qryt):
        h.close()

    # ---- report and self-check -------------------------------------------
    print(f"\nwrote {len(handles)} FASTA files to {args.outdir}\n")
    hdr = f"{'partition':<12}{'view':>6}{'written':>10}{'expected':>10}   status"
    print(hdr)
    print("-" * len(hdr))
    ok = True
    for p in PARTITIONS:
        for key in ["total"] + VIEWS:
            got = counts[p][key]
            exp = EXPECTED[p][key]
            good = got == exp
            ok &= good
            label = p if key == "total" else ""
            print(f"{label:<12}{key:>6}{got:>10,}{exp:>10,}   "
                  f"{'ok' if good else 'MISMATCH ' + f'{got - exp:+d}'}")
        print()

    if other_splits:
        print("records carrying some other split label (not written):")
        for k, n in sorted(other_splits.items(), key=lambda kv: -kv[1]):
            print(f"  {k or '<blank>'}: {n:,}")
        print()
    if n_spaces:
        print(f"WARNING: {n_spaces:,} record ids contain whitespace. VSEARCH "
              "truncates labels at the first space unless --notrunclabels is "
              "given, which would break the join with the manifest. Set "
              "NOTRUNC=--notrunclabels when running the searches.\n")

    print(f"wrote {meta_dir/'test_query_manifest.csv'}, "
          f"{meta_dir/'ref_taxonomy.csv'}, {meta_dir/'query_taxonomy.csv'}")

    if ok:
        print("\nAll 24 counts match the published figures. These FASTAs contain "
              "the same records the training runner and the frozen evaluator "
              "used, so the searches can proceed.")
        return 0
    print("\nAt least one count disagrees with the published figures. Do not run "
          "the searches yet: either the split table on disk is not the one the "
          "paper was built from, or the conflict masking is being applied "
          "differently here than in the training runner. Resolve that first "
          "(--no-check writes the files anyway, for inspection).")
    return 0 if args.no_check else 1


if __name__ == "__main__":
    raise SystemExit(main())
