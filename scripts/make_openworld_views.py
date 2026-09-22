#!/usr/bin/env python3
"""Create canonical ITS-core / ITS1 / ITS2 views from FASTA + ITSx positions.

The output is a TSV suitable for scripts/build_openworld_splits.py.
Only coordinates explicitly present in positions.txt are sliced; missing regions
remain empty rather than being inferred from sequence ends.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from collections import defaultdict

from itsnet.data import read_fasta, parse_taxonomy, seq_id
from itsnet.labels import read_positions


def slice_region(seq, span):
    if span is None:
        return ""
    s, e = span
    if s < 0 or e > len(seq) or e <= s:
        return ""
    return seq[s:e].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--conflicts", default=None,
                    help="optional TSV for exact-view sequences assigned to >1 genus")
    args = ap.parse_args()

    pos = read_positions(args.positions)
    fields = [
        "id", "source_id", "kingdom", "phylum", "class", "order", "family", "genus", "species",
        "core", "its1", "its2", "extraction_method",
    ]
    rows = []
    seen_ids = set()
    view_taxa = defaultdict(set)
    view_ids = defaultdict(list)

    for header, seq in read_fasta(args.fasta):
        sid = seq_id(header)
        if sid in seen_ids:
            raise SystemExit(f"duplicate FASTA identifier after normalization: {sid}")
        seen_ids.add(sid)
        rec = pos.get(sid)
        if rec is None:
            continue
        tax = parse_taxonomy(header)
        its1 = slice_region(seq, rec.regions.get("ITS1"))
        its2 = slice_region(seq, rec.regions.get("ITS2"))
        core = ""
        if all(k in rec.regions for k in ("ITS1", "S58", "ITS2")):
            s = rec.regions["ITS1"][0]
            e = rec.regions["ITS2"][1]
            if 0 <= s < e <= len(seq):
                core = seq[s:e].upper()
        row = {
            "id": sid,
            "source_id": sid,
            "kingdom": tax.get("k", ""),
            "phylum": tax.get("p", ""),
            "class": tax.get("c", ""),
            "order": tax.get("o", ""),
            "family": tax.get("f", ""),
            "genus": tax.get("g", ""),
            "species": tax.get("s", ""),
            "core": core,
            "its1": its1,
            "its2": its2,
            "extraction_method": "ITSx_positions",
        }
        rows.append(row)
        for name, view in (("core", core), ("its1", its1), ("its2", its2)):
            if view and row["genus"]:
                key = (name, hashlib.sha256(view.encode()).hexdigest())
                view_taxa[key].add((row["family"], row["genus"]))
                view_ids[key].append(sid)

    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    conflicts = [(k, taxa, view_ids[k]) for k, taxa in view_taxa.items() if len(taxa) > 1]
    if args.conflicts:
        with open(args.conflicts, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["view", "sha256", "taxa", "ids"])
            for (view, sha), taxa, ids in conflicts:
                w.writerow([view, sha, ";".join(f"{f}|{g}" for f, g in sorted(taxa)), ";".join(ids)])

    n_core = sum(bool(r["core"]) for r in rows)
    n_i1 = sum(bool(r["its1"]) for r in rows)
    n_i2 = sum(bool(r["its2"]) for r in rows)
    print(f"records : {len(rows)}")
    print(f"core    : {n_core}")
    print(f"ITS1    : {n_i1}")
    print(f"ITS2    : {n_i2}")
    print(f"cross-genus exact-view conflicts: {len(conflicts)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
