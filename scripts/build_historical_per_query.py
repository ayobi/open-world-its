#!/usr/bin/env python3
"""
Build the per-query score table that paired_uncertainty.py consumes.

The identity and cosine arms live in different trees and have different row
counts, because VSEARCH writes a row only for queries that obtained a hit at
the 0.5 floor. The cosine arm scores every query. The difference is exactly
the censored set the manuscript reports (542 LGO, 21 LSO), and those queries
must be carried into the output at the censoring value rather than dropped:
they are the most anomalous identity scores available and account for a large
part of identity's detection at stringent alpha. An inner join would remove
them and quietly change the result.

Inputs are headerless TSVs of the form  query <TAB> target <TAB> value,
which is what `--userfields query+target+id` and the cosine dump both emit.

Usage
-----
    python3 analysis/build_historical_per_query.py \
        --lgo-identity ~/Downloads/its-novelty/benchmark_current/lgo_identity.raw.tsv \
        --lso-identity ~/Downloads/its-novelty/benchmark_current/lso_identity.raw.tsv \
        --lgo-cosine   runs/historical_benchmark/lgo_cosine.raw.tsv \
        --lso-cosine   runs/historical_benchmark/lso_cosine.raw.tsv \
        --genus-map    data/openworld/unite2025_splits.tsv \
        --out          data/historical_per_query.csv

The seed-0 half of the LSO queries becomes role=cal and the rest role=known,
reproducing the split described in Section 2.7. Pass --cal-seed to change it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

CENSOR = 49.999


def read_pairs(path: str, value_name: str) -> pd.DataFrame:
    """Read a headerless query/target/value TSV, keeping the best row per query."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 2],
                     names=["query_id", value_name], dtype={0: str})
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    if df[value_name].isna().any():
        sys.exit(f"error: non-numeric value in {path}")
    # top_hits_only can still emit ties; keep the best and warn.
    n_before = len(df)
    df = (df.sort_values(value_name, ascending=False)
            .drop_duplicates("query_id", keep="first")
            .reset_index(drop=True))
    if len(df) != n_before:
        print(f"  note: {path} had {n_before - len(df):,} tied extra row(s); "
              f"kept the best per query")
    return df


def load_genus_map(path: str, query_ids: set, id_col: str | None = None) -> dict:
    """
    Pull a query_id -> genus mapping out of the split table.

    The split table may key on a UNITE identifier while the benchmark queries
    are GenBank accessions, so the identifier column is chosen by how well it
    actually overlaps the query set rather than by name. Guessing from the name
    fails silently: every lookup misses and clustering is quietly dropped.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    gen_col = next((cols[c] for c in ("genus", "genus_name") if c in cols), None)
    if gen_col is None:
        sys.exit(f"error: no genus column in {path}; columns are "
                 f"{', '.join(df.columns)}")

    # UNITE headers are compound, e.g.
    #   Abrothallus_subhalei|MT153946|SH1227328.10FU|refs|k__Fungi;p__...
    # and the benchmark queries are bare accessions, so the identifier may be
    # one field of a delimited string rather than the whole cell. Score each
    # column whole, then each field position within it, and take the best.
    def candidates(col: pd.Series):
        yield None, col
        for delim in ("|", ";", " "):
            sample = col.dropna()
            if sample.empty or not sample.iloc[0].count(delim):
                continue
            parts = sample.str.split(delim, regex=False)
            for k in range(max(parts.str.len().min(), 0)):
                yield (delim, k), parts.str[k]

    def score(col: pd.Series):
        best_key, best_vals, best_hits = None, col, -1
        for key, vals in candidates(col):
            hits = len(set(vals.dropna()) & query_ids)
            if hits > best_hits:
                best_key, best_vals, best_hits = key, vals, hits
        return best_key, best_vals, best_hits

    if id_col is not None:
        if id_col not in df.columns:
            sys.exit(f"error: --genus-id-col '{id_col}' not in {path}; "
                     f"columns are {', '.join(df.columns)}")
        key, vals, best_n = score(df[id_col])
        best = id_col
    else:
        ranked = [(c, *score(df[c])) for c in df.columns]
        best, key, vals, best_n = max(ranked, key=lambda t: t[3])

    field = "" if key is None else f", field {key[1]} of '{key[0]}'-delimited"

    if best_n == 0:
        print(f"\n  no column in {path} matches the query identifiers. "
              f"First values per column:")
        for c in df.columns[:8]:
            vals = ", ".join(map(str, df[c].dropna().unique()[:2]))
            print(f"    {c:<20} {vals[:60]}")
        print(f"  query identifiers look like: "
              f"{', '.join(sorted(query_ids)[:3])}")
        sys.exit("error: cannot map genus; pass --genus-id-col explicitly or "
                 "omit --genus-map to proceed without clustering.")

    pct = 100 * best_n / max(len(query_ids), 1)
    print(f"  genus map: '{best}'{field} -> '{gen_col}' ({len(df):,} rows, "
          f"matches {best_n:,}/{len(query_ids):,} queries, {pct:.1f}%)")
    if pct < 90:
        print(f"  warning: only {pct:.1f}% of queries were matched; the "
              f"unmatched ones will be dropped from clustering")
    return dict(zip(vals, df[gen_col]))


def arm(ident_path, cos_path, role_label, tag):
    ident = read_pairs(ident_path, "identity")
    cos = read_pairs(cos_path, "cosine")

    # Left join on the cosine arm: it is the complete query set.
    df = cos.merge(ident, on="query_id", how="left")
    n_cens = int(df["identity"].isna().sum())
    df["identity"] = df["identity"].fillna(CENSOR)
    df["role"] = role_label
    df["censored"] = df["identity"].eq(CENSOR)

    missing_cos = set(ident["query_id"]) - set(cos["query_id"])
    if missing_cos:
        sys.exit(f"error: {len(missing_cos):,} {tag} queries have an identity "
                 f"score but no cosine score, e.g. "
                 f"{sorted(missing_cos)[:3]}. The cosine arm should cover "
                 f"every query; check you are using the right run directory.")

    print(f"  {tag}: {len(df):,} queries, {n_cens:,} censored "
          f"({100 * n_cens / max(len(df), 1):.1f}%)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lgo-identity", required=True)
    ap.add_argument("--lso-identity", required=True)
    ap.add_argument("--lgo-cosine", required=True)
    ap.add_argument("--lso-cosine", required=True)
    ap.add_argument("--genus-map", default=None,
                    help="split table carrying a genus column (recommended)")
    ap.add_argument("--genus-id-col", default=None,
                    help="identifier column in the genus map; auto-detected "
                         "by overlap with the query ids when omitted")
    ap.add_argument("--cal-seed", type=int, default=0,
                    help="seed for the LSO calibration/test halving (default 0)")
    ap.add_argument("--expect-lgo", type=int, default=4444)
    ap.add_argument("--expect-lso", type=int, default=3264)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("building arms")
    lgo = arm(args.lgo_identity, args.lgo_cosine, "novel", "LGO")
    lso = arm(args.lso_identity, args.lso_cosine, None, "LSO")

    genus = None
    if args.genus_map:
        ids = set(lgo["query_id"]) | set(lso["query_id"])
        genus = load_genus_map(args.genus_map, ids, args.genus_id_col)

    # Halve the LSO set into calibration and test, as in Section 2.7.
    rng = np.random.default_rng(args.cal_seed)
    idx = rng.permutation(len(lso))
    half = len(lso) // 2
    lso = lso.copy()
    lso["role"] = "known"
    lso.iloc[idx[:half], lso.columns.get_loc("role")] = "cal"

    out = pd.concat([lso, lgo], ignore_index=True)

    if genus:
        out["genus"] = out["query_id"].map(genus)
        n_nogen = int(out["genus"].isna().sum())
        if n_nogen:
            print(f"  warning: {n_nogen:,}/{len(out):,} queries have no genus "
                  f"and will be dropped from clustering")

    ok = True
    if len(lgo) != args.expect_lgo:
        print(f"  MISMATCH: {len(lgo):,} LGO queries, expected "
              f"{args.expect_lgo:,}"); ok = False
    if len(lso) != args.expect_lso:
        print(f"  MISMATCH: {len(lso):,} LSO queries, expected "
              f"{args.expect_lso:,}"); ok = False

    cols = ["query_id", "role", "identity", "cosine", "censored"]
    if genus:
        cols.insert(2, "genus")
    out[cols].to_csv(args.out, index=False)

    print(f"\nwrote {args.out}")
    print(out.groupby("role").size().to_string())
    if genus:
        print(f"novel genera: {out.loc[out.role == 'novel', 'genus'].nunique():,}")
    print("counts match the manuscript" if ok else
          "CHECK THE COUNTS ABOVE before using this table")


if __name__ == "__main__":
    main()
