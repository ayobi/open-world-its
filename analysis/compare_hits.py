#!/usr/bin/env python3
"""Quantify what the corrected VSEARCH configuration changed.

The deposited searches used --maxaccepts 1 --maxrejects 32, which stops at the
first target passing --id and gives up after 32 consecutive failures. This
compares those hits against an exhaustive --maxaccepts 0 --maxrejects 0 rerun
and reports how often the default settings returned a non-best hit.

That number belongs in Methods. It is the difference between saying "the
identity baseline is a default-settings baseline" and being able to say how
much the default settings cost.

Inputs are the headerless query/target/value TSVs that
`--userfields query+target+id` emits, matching build_historical_per_query.py.

Usage
-----
    python3 analysis/compare_hits.py \
        --old ~/Downloads/its-novelty/benchmark_current/lgo_identity.raw.tsv \
        --new runs/identity_exhaustive/hist_lgo.exhaustive.tsv \
        --label LGO

    # several pairs at once, writing a summary table
    python3 analysis/compare_hits.py \
        --pair LGO old_lgo.tsv new_lgo.tsv \
        --pair LSO old_lso.tsv new_lso.tsv \
        --out data/search_correction_audit.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CENSOR = 49.999


def read_hits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["query_id", "target", "identity"], dtype={0: str, 1: str})
    df["identity"] = pd.to_numeric(df["identity"], errors="coerce")
    if df["identity"].isna().any():
        raise SystemExit(f"non-numeric identity value in {path}")
    return (df.sort_values("identity", ascending=False)
              .drop_duplicates("query_id", keep="first")
              .reset_index(drop=True))


def compare(label: str, old_path: Path, new_path: Path) -> dict:
    old = read_hits(old_path).set_index("query_id")
    new = read_hits(new_path).set_index("query_id")

    all_ids = old.index.union(new.index)
    both = old.index.intersection(new.index)
    only_new = new.index.difference(old.index)   # rescued from false "no hit"
    only_old = old.index.difference(new.index)   # should be empty

    o = old.loc[both]
    n = new.loc[both]
    changed_target = (o["target"].to_numpy() != n["target"].to_numpy())
    delta = n["identity"].to_numpy() - o["identity"].to_numpy()

    rec = {
        "set": label,
        "queries_with_hit_old": len(old),
        "queries_with_hit_new": len(new),
        "rescued_from_no_hit": len(only_new),
        "lost_hit": len(only_old),
        "shared": len(both),
        "target_changed": int(changed_target.sum()),
        "target_changed_pct": 100 * changed_target.mean() if len(both) else np.nan,
        "identity_improved": int((delta > 1e-9).sum()),
        "identity_worsened": int((delta < -1e-9).sum()),
        "mean_identity_gain": float(delta.mean()) if len(both) else np.nan,
        "median_identity_gain": float(np.median(delta)) if len(both) else np.nan,
        "max_identity_gain": float(delta.max()) if len(both) else np.nan,
    }

    print(f"\n=== {label} ===")
    print(f"  queries with a hit:   {len(old):,} -> {len(new):,} "
          f"({len(only_new):,} rescued from a false no-hit)")
    if len(only_old):
        print(f"  WARNING: {len(only_old):,} queries lost their hit. An exhaustive "
              "search should be a superset; check that both runs used the same "
              "query and reference files.")
    print(f"  best hit changed:     {rec['target_changed']:,} of {len(both):,} "
          f"({rec['target_changed_pct']:.1f}%)")
    print(f"  identity improved:    {rec['identity_improved']:,}   "
          f"worsened: {rec['identity_worsened']:,}")
    print(f"  identity gain:        mean {rec['mean_identity_gain']:+.2f} pp, "
          f"median {rec['median_identity_gain']:+.2f} pp, "
          f"max {rec['max_identity_gain']:+.2f} pp")
    if rec["identity_worsened"] > 0:
        print("  note: an exhaustive search cannot return a worse hit than a "
              "truncated one for the same query. Any 'worsened' rows mean the "
              "two runs differ in inputs or settings beyond the caps.")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--new", type=Path)
    ap.add_argument("--label", default="hits")
    ap.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "OLD", "NEW"),
                    help="repeatable: label, old TSV, new TSV")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    jobs = []
    if args.old and args.new:
        jobs.append((args.label, args.old, args.new))
    for lab, o, n in (args.pair or []):
        jobs.append((lab, Path(o), Path(n)))
    if not jobs:
        raise SystemExit("give --old/--new or at least one --pair")

    rows = [compare(lab, o, n) for lab, o, n in jobs]
    df = pd.DataFrame(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    print("\n" + "-" * 72)
    print("Sentence for Methods, once you have filled in the numbers:\n")
    tot_shared = int(df["shared"].sum())
    tot_changed = int(df["target_changed"].sum())
    tot_rescued = int(df["rescued_from_no_hit"].sum())
    pct = 100 * tot_changed / tot_shared if tot_shared else float("nan")
    print(f"  An exhaustive rerun (--maxaccepts 0 --maxrejects 0) returned a\n"
          f"  different best hit for {tot_changed:,} of {tot_shared:,} queries ({pct:.1f}%) and\n"
          f"  recovered a hit for {tot_rescued:,} queries that the default settings\n"
          f"  reported as unmatched. All identity results below use the exhaustive\n"
          f"  search; the default-settings run is retained in the repository for\n"
          f"  comparison.")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
