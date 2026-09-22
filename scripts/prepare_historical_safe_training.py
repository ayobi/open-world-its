#!/usr/bin/env python3
"""Mask historical benchmark query taxa out of the already-selected M4 TRAIN_REF.

This is a post-development leakage-control step. It does NOT rebuild/tune the
six-way split. It only changes TRAIN_REF membership for a frozen-architecture
retraining used to compare against the historical LSO/LGO benchmark.

Exclusions:
  * every genus queried by historical LGO (deep novelty),
  * every (genus, species) queried by historical LSO (shallow novelty),
  * exact historical query sequence IDs as a final safety net.

All DEV/CAL/TEST rows are preserved byte-for-field and never promoted to training.
Excluded TRAIN_REF rows are retained with split=BENCHMARK_HOLDOUT for auditability.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ACC_RE = re.compile(r"^(UDB\d+|[A-Z]{1,3}\d{5,8}(\.\d+)?)$")
SH_RE = re.compile(r"^SH\d+\.\d+\w*$")


def read_tsv(path: Path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_tax(path: Path):
    rows = read_tsv(path)
    required = {"seq_id", "g", "s"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"taxonomy missing columns {sorted(required)}: {path}")
    return {r["seq_id"]: r for r in rows}


def query_ids(split_path: Path):
    rows = read_tsv(split_path)
    if not rows or not {"seq_id", "role"}.issubset(rows[0]):
        raise SystemExit(f"split missing seq_id/role: {split_path}")
    return {r["seq_id"] for r in rows if r["role"] == "query"}


def unite_seq_id(header: str):
    parts = [p.strip() for p in (header or "").split("|")]
    sid = next((p for p in parts if ACC_RE.match(p)), None)
    if sid is None:
        sid = next((p for p in parts if SH_RE.match(p)), None)
    return sid


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="selected six-way open-world TSV")
    ap.add_argument("--lgo-split", required=True)
    ap.add_argument("--lgo-taxonomy", required=True)
    ap.add_argument("--lso-split", required=True)
    ap.add_argument("--lso-taxonomy", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    args = ap.parse_args()

    input_path = Path(args.input)
    rows = read_tsv(input_path)
    required = {"id", "genus", "species", "split"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit("input missing columns: " + ", ".join(sorted(required)))

    lgo_q = query_ids(Path(args.lgo_split))
    lso_q = query_ids(Path(args.lso_split))
    lgo_tax = load_tax(Path(args.lgo_taxonomy))
    lso_tax = load_tax(Path(args.lso_taxonomy))

    missing_lgo = sorted(lgo_q - set(lgo_tax))
    missing_lso = sorted(lso_q - set(lso_tax))
    if missing_lgo or missing_lso:
        raise SystemExit(
            f"benchmark taxonomy missing query IDs: LGO={len(missing_lgo)} LSO={len(missing_lso)}"
        )

    lgo_genera = {lgo_tax[x]["g"] for x in lgo_q if lgo_tax[x]["g"]}
    lso_species = {
        (lso_tax[x]["g"], lso_tax[x]["s"])
        for x in lso_q if lso_tax[x]["g"] and lso_tax[x]["s"]
    }
    lso_genera = {g for g, _ in lso_species}
    exact_q = lgo_q | lso_q

    before_train = [r for r in rows if r["split"] == "TRAIN_REF"]
    out_rows = []
    reasons = Counter()
    excluded_ids = set()
    for r in rows:
        rr = dict(r)
        if r["split"] == "TRAIN_REF":
            sid = unite_seq_id(r["id"])
            why = []
            if r["genus"] in lgo_genera:
                why.append("LGO_GENUS")
            if (r["genus"], r["species"]) in lso_species:
                why.append("LSO_SPECIES")
            if sid and sid in exact_q:
                why.append("EXACT_QUERY_ID")
            if why:
                rr["split"] = "BENCHMARK_HOLDOUT"
                excluded_ids.add(r["id"])
                for x in set(why):
                    reasons[x] += 1
        out_rows.append(rr)

    after_train = [r for r in out_rows if r["split"] == "TRAIN_REF"]
    remaining_ids = {unite_seq_id(r["id"]) for r in after_train}
    remaining_ids.discard(None)
    remaining_genera = {r["genus"] for r in after_train if r["genus"]}
    remaining_species = {
        (r["genus"], r["species"])
        for r in after_train if r["genus"] and r["species"]
    }

    leak_lgo_ids = lgo_q & remaining_ids
    leak_lgo_genera = lgo_genera & remaining_genera
    leak_lso_ids = lso_q & remaining_ids
    leak_lso_species = lso_species & remaining_species
    if leak_lgo_ids or leak_lgo_genera or leak_lso_ids or leak_lso_species:
        raise SystemExit(
            "post-filter leakage remains: "
            f"LGO ids={len(leak_lgo_ids)} genera={len(leak_lgo_genera)} "
            f"LSO ids={len(leak_lso_ids)} species={len(leak_lso_species)}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(out_rows)

    split_counts = Counter(r["split"] for r in out_rows)
    audit = {
        "policy": "original TRAIN_REF minus historical LGO query genera minus historical LSO query species; no other split promoted",
        "historical_lgo_queries": len(lgo_q),
        "historical_lgo_query_genera": len(lgo_genera),
        "historical_lso_queries": len(lso_q),
        "historical_lso_query_species": len(lso_species),
        "historical_lso_query_genera": len(lso_genera),
        "train_ref_before": len(before_train),
        "train_ref_after": len(after_train),
        "train_ref_excluded": len(before_train) - len(after_train),
        "exclusion_reason_counts_nonexclusive": dict(reasons),
        "remaining_train_genera": len(remaining_genera),
        "remaining_train_families": len({r.get("family", "") for r in after_train if r.get("family", "")}),
        "lso_query_genera_still_represented_in_train": len(lso_genera & remaining_genera),
        "post_filter_leakage": {
            "lgo_exact_query_ids": 0,
            "lgo_query_genera": 0,
            "lso_exact_query_ids": 0,
            "lso_query_species": 0,
        },
        "output_split_counts": dict(split_counts),
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    print("HISTORICAL BENCHMARK-SAFE TRAINING MASK")
    print(f"TRAIN_REF before              : {len(before_train):,}")
    print(f"TRAIN_REF after               : {len(after_train):,}")
    print(f"excluded                      : {len(before_train)-len(after_train):,}")
    print(f"remaining genera              : {len(remaining_genera):,}")
    print(f"LGO query genera remaining    : {len(leak_lgo_genera)}")
    print(f"LGO exact query IDs remaining : {len(leak_lgo_ids)}")
    print(f"LSO query species remaining   : {len(leak_lso_species)}")
    print(f"LSO exact query IDs remaining : {len(leak_lso_ids)}")
    print(f"LSO query genera still known  : {len(lso_genera & remaining_genera):,} / {len(lso_genera):,}")
    print(f"wrote {out_path}")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
