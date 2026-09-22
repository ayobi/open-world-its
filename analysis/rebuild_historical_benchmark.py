#!/usr/bin/env python3
"""Regenerate data/historical_benchmark.csv from deposited inputs.

Every row of Table 7 and every panel of Figure 6 reads that CSV. Until now the
file was hand-maintained: nothing in the repository wrote it, so its numbers
could not be reproduced from the deposit. This script closes that gap. It
computes the six score rows from the per-query table and the two family-recovery
rows from the raw best-hit tables, so Table 7 is regenerated end to end.

  AUROC and conformal rows  <- data/historical_per_query.csv
  family recovery rows      <- the four raw query/target/value TSVs + the split
                               table, which carries the taxonomy

The conformal procedure matches Equation 8 and paired_uncertainty.py exactly:
CAL rows calibrate, novel queries are scored against them, and a query is
flagged when its conformal p-value is at or below alpha.

-----------------------------------------------------------------------------
ON THE LSO FAMILY-RECOVERY ROW
-----------------------------------------------------------------------------
Family recovery is a property of the best hit, not of the conformal threshold,
so it does not need a calibration split. This script reports it two ways:

  all      all 3,264 LSO queries      -> independent of the cal/known halving
  testhalf the 1,632 role=known rows  -> moves with the halving

--lso-scope selects which one is written to the CSV. The default is "all",
because a split-independent number is the one that stays reproducible when the
halving convention is stated rather than guessed at. The printout always shows
both, so the difference between them is visible rather than assumed.

-----------------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------------
    python3 analysis/rebuild_historical_benchmark.py \
        --per-query    data/historical_per_query.csv \
        --lgo-identity ~/Downloads/its-novelty/benchmark_current/lgo_identity.raw.tsv \
        --lso-identity ~/Downloads/its-novelty/benchmark_current/lso_identity.raw.tsv \
        --lgo-cosine   runs/historical_benchmark/lgo_cosine.raw.tsv \
        --lso-cosine   runs/historical_benchmark/lso_cosine.raw.tsv \
        --splits       data/openworld/unite2025_splits.tsv \
        --out          data/historical_benchmark.csv

The split table's `id` is the full pipe-delimited UNITE header, whose second
field is the accession that the hit tables use. Both INSD accessions
(MK984582) and UNITE internal ones (UDB02644331) sit in that field, so one
lookup resolves both. The script reports the match rate and refuses to write
the file if it falls below --min-match, because a silent lookup miss would
depress family recovery rather than raise an error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHAS = [0.01, 0.05, 0.10, 0.20]
RANK = "family"


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC with midranks, so censored ties are handled."""
    from scipy import stats
    r = stats.rankdata(score)
    m = int(positive.sum())
    n = score.size - m
    return float((r[positive].sum() - m * (m + 1) / 2) / (m * n))


def conformal_p(cal_anom: np.ndarray, q_anom: np.ndarray) -> np.ndarray:
    n = cal_anom.size
    ge = (cal_anom[None, :] >= q_anom[:, None]).sum(axis=1)
    return (1 + ge) / (n + 1)


def best_hits(path: Path) -> pd.DataFrame:
    """Best-scoring target per query from a headerless query/target/value TSV."""
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["query_id", "target", "value"],
                     dtype={0: str, 1: str})
    df = df.sort_values("value", ascending=False).drop_duplicates("query_id")
    return df.set_index("query_id")


def accession_family_map(splits: Path) -> pd.Series:
    """accession -> family from the pipe-delimited UNITE id.

    Field 2 is the accession, INSD or UNITE internal, and resolves almost every
    target. A small number of reference records are named in the hit tables by
    their field-3 SH code instead, so SH codes are added as a fallback. An SH is
    a clustering unit that several records can share, so only SH codes whose
    records all agree on the family are included: the fallback can fill a gap
    but cannot invent a placement. Field 2 always takes precedence.
    """
    df = pd.read_csv(splits, sep="\t", usecols=["id", RANK], dtype=str)
    parts = df["id"].str.split("|", regex=False)
    primary = pd.Series(df[RANK].to_numpy(), index=parts.str[1])
    primary = primary[primary.index.notna() & ~primary.index.duplicated(keep="first")]

    sh = pd.DataFrame({"sh": parts.str[2], RANK: df[RANK].to_numpy()}).dropna()
    unambiguous = sh.groupby("sh")[RANK].nunique().eq(1)
    fallback = (sh[sh.sh.isin(unambiguous[unambiguous].index)]
                .drop_duplicates("sh").set_index("sh")[RANK])
    fallback = fallback[~fallback.index.isin(primary.index)]
    print(f"  taxonomy keys: {len(primary):,} accessions "
          f"+ {len(fallback):,} unambiguous SH codes as fallback")
    return pd.concat([primary, fallback])


def family_recovery(hits: pd.DataFrame, query_ids, fam: pd.Series) -> dict:
    """Fraction of queries whose best hit shares a family with the query.

    Queries with no hit cannot place and count as failures, so the denominator
    is the whole query set rather than the hit table.
    """
    q = pd.Index(query_ids)
    tgt = hits["target"].reindex(q)
    q_fam = fam.reindex(q)
    t_fam = pd.Series(fam.reindex(tgt.fillna("")).to_numpy(), index=q)
    ok = (q_fam.notna() & t_fam.notna() & (q_fam == t_fam))
    return {
        "rate": float(ok.sum()) / len(q),
        "n": len(q),
        "no_hit": int(tgt.isna().sum()),
        "query_unmatched": int(q_fam.isna().sum()),
        "target_unmatched": int((tgt.notna() & t_fam.isna()).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-query", type=Path, required=True)
    ap.add_argument("--lgo-identity", type=Path, required=True)
    ap.add_argument("--lso-identity", type=Path, required=True)
    ap.add_argument("--lgo-cosine", type=Path, required=True)
    ap.add_argument("--lso-cosine", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/historical_benchmark.csv"))
    ap.add_argument("--lso-scope", choices=["all", "testhalf"], default="all")
    ap.add_argument("--min-match", type=float, default=0.95,
                    help="refuse to write if fewer than this fraction of "
                         "queries or targets resolve to a family")
    args = ap.parse_args()

    # ---- score rows, from the per-query table -----------------------------
    d = pd.read_csv(args.per_query)
    cal = d[d.role == "cal"]
    known = d[d.role == "known"]
    novel = d[d.role == "novel"]
    ev = pd.concat([known, novel])
    pos = (ev.role == "novel").to_numpy()

    out = {}
    for name, col in [("identity", "identity"), ("m4_cosine", "cosine")]:
        a_ev = -ev[col].to_numpy(float)
        a_cal = -cal[col].to_numpy(float)
        vals = {"AUROC": auroc(a_ev, pos)}
        p_nov = conformal_p(a_cal, -novel[col].to_numpy(float))
        p_kn = conformal_p(a_cal, -known[col].to_numpy(float))
        for al in ALPHAS:
            vals[f"Novel detection @ alpha={al:.2f}"] = float((p_nov <= al).mean())
        vals["False novelty @ alpha=0.05"] = float((p_kn <= 0.05).mean())
        out[name] = vals

    print(f"per-query table: {len(cal):,} cal, {len(known):,} known, "
          f"{len(novel):,} novel")
    print(f"  AUROC  identity {out['identity']['AUROC']:.4f}  "
          f"cosine {out['m4_cosine']['AUROC']:.4f}")

    # ---- family recovery, from the raw hit tables --------------------------
    fam = accession_family_map(args.splits)
    print(f"taxonomy map: {len(fam):,} accessions -> {RANK}")

    lso_all = pd.Index(d[d.role != "novel"].query_id)
    lso_half = pd.Index(known.query_id)
    lgo_ids = pd.Index(novel.query_id)

    worst = 1.0
    for label, hits_path, scope_rows in [
            ("identity", args.lso_identity, None),
            ("m4_cosine", args.lso_cosine, None)]:
        hits = best_hits(hits_path)
        r_all = family_recovery(hits, lso_all, fam)
        r_half = family_recovery(hits, lso_half, fam)
        chosen = r_all if args.lso_scope == "all" else r_half
        out[label]["LSO family recovery"] = chosen["rate"]
        print(f"  LSO family recovery [{label:9}] "
              f"all {100*r_all['rate']:.2f}% (n={r_all['n']:,})  "
              f"testhalf {100*r_half['rate']:.2f}% (n={r_half['n']:,})")
        if r_all["target_unmatched"] or r_all["query_unmatched"]:
            print(f"      unresolved: {r_all['query_unmatched']} queries, "
                  f"{r_all['target_unmatched']} targets, "
                  f"{r_all['no_hit']} with no hit")
        worst = min(worst, max(0.0, 1 - max(r_all["query_unmatched"],
                                            r_all["target_unmatched"]) / r_all["n"]))

    for label, hits_path in [("identity", args.lgo_identity),
                             ("m4_cosine", args.lgo_cosine)]:
        hits = best_hits(hits_path)
        r = family_recovery(hits, lgo_ids, fam)
        out[label]["LGO family recovery"] = r["rate"]
        print(f"  LGO family recovery [{label:9}] {100*r['rate']:.2f}% "
              f"(n={r['n']:,}, no hit {r['no_hit']}, "
              f"unresolved {r['query_unmatched']}q/{r['target_unmatched']}t)")
        worst = min(worst, max(0.0, 1 - max(r["query_unmatched"],
                                            r["target_unmatched"]) / r["n"]))

    if worst < args.min_match:
        sys.exit(f"\nABORT: only {100*worst:.1f}% of ids resolved to a family, "
                 f"below --min-match {100*args.min_match:.0f}%. The accession "
                 f"lookup is wrong; family recovery would be understated. "
                 f"Check that the split table's id field 2 matches the "
                 f"accessions in the hit tables.")

    # ---- write -------------------------------------------------------------
    order = ["AUROC",
             "Novel detection @ alpha=0.01", "Novel detection @ alpha=0.05",
             "Novel detection @ alpha=0.10", "Novel detection @ alpha=0.20",
             "False novelty @ alpha=0.05",
             "LSO family recovery", "LGO family recovery"]
    df = pd.DataFrame({"metric": order,
                       "identity": [out["identity"][k] for k in order],
                       "m4_cosine": [out["m4_cosine"][k] for k in order]})
    df["identity"] = df["identity"].round(4)
    df["m4_cosine"] = df["m4_cosine"].round(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  (LSO scope: {args.lso_scope})")
    print(df.to_string(index=False))
    print("\nNow rerun analysis/make_tables.py and analysis/make_figures.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
