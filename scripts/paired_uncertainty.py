#!/usr/bin/env python3
"""
Paired uncertainty for the historical ITS2 benchmark (Table 7).

Percent identity and M4 cosine are scored on identical queries, so the
difference between them is a paired quantity and deserves a paired interval
rather than a point comparison. This script reports three things:

  1. AUROC for each score, with a DeLong standard error.
  2. The AUROC difference, with both a DeLong test and a stratified paired
     bootstrap percentile interval. The two should agree closely; if they do
     not, trust the bootstrap, because DeLong assumes the sample is large
     enough for the asymptotic normal approximation and the censored tail of
     the identity score is heavily tied.
  3. At each conformal level, the false-novelty and detection rates for both
     scores, plus McNemar's test on the paired flag sets. McNemar is the right
     test here because the two scores flag the *same* queries: a chi-square on
     the marginal rates would ignore that pairing and overstate the evidence.

The DeLong implementation uses midranks throughout (Sun & Xu 2014), which
matters for this benchmark specifically: 542 LGO queries have no VSEARCH hit
and share one censored identity value, and a tie-naive AUROC would score those
inconsistently between the two methods.

Input
-----
A CSV or TSV with one row per query and these columns:

    query_id   unique identifier
    role       one of: cal, known, novel
                 cal   - calibration knowns (the seed-0 LSO half)
                 known - test knowns (the other LSO half)
                 novel - LGO queries, the novel population
    identity   best-hit percent identity (higher = more familiar)
    cosine     max embedding cosine       (higher = more familiar)
    genus      OPTIONAL but strongly recommended: the query's genus.

Why the genus column matters. The 4,444 LGO queries come from 538 genera, and
queries from one genus share a reference neighbourhood, so they are not
independent draws. A bootstrap that resamples queries treats them as if they
were and returns an interval that is too narrow. With a genus column present
this script resamples *genera* instead, in both classes, and reports that
interval alongside the query-level one. The clustered interval is the one to
report; the query-level one is kept only so the difference is visible.

Both score columns are converted internally to anomaly scores A = -score, so
larger means more anomalous, matching Eq. 1 and Eq. 8 of the manuscript.

Usage
-----
    python3 analysis/paired_uncertainty.py --scores data/historical_per_query.csv
    python3 analysis/paired_uncertainty.py --scores scores.tsv --n-boot 20000 \
        --alphas 0.01 0.05 0.10 0.20 --seed 17 --latex

Add --latex to emit a \\item line ready to paste into the Table 7 tablenotes.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROLES = ("cal", "known", "novel")


# --------------------------------------------------------------------- AUROC --
def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks, so tied values share the average of the ranks they span."""
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie correction. pos = novel, neg = known."""
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _midrank(np.concatenate([pos, neg]))
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def delong(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong (Sun & Xu 2014) for k scores measured on the same samples.

    pos, neg: arrays of shape (k, n_pos) and (k, n_neg), anomaly-oriented.
    Returns (aucs, covariance matrix of the aucs).
    """
    k, m = pos.shape
    n = neg.shape[1]
    allx = np.concatenate([pos, neg], axis=1)

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(allx[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    cov = np.atleast_2d(cov)
    return aucs, cov


def delong_test(pos: np.ndarray, neg: np.ndarray) -> dict:
    aucs, cov = delong(pos, neg)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(var_diff, 0.0)))
    if se == 0.0:
        z, p = float("inf") if diff else 0.0, 0.0 if diff else 1.0
    else:
        z = diff / se
        p = 2.0 * stats.norm.sf(abs(z))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "se_a": float(np.sqrt(cov[0, 0])),
        "se_b": float(np.sqrt(cov[1, 1])),
        "diff": float(diff),
        "se_diff": se,
        "z": float(z),
        "p": float(p),
        "ci": (float(diff - 1.96 * se), float(diff + 1.96 * se)),
    }


def _cluster_index(labels: np.ndarray) -> list:
    """Row indices grouped by cluster label, for cluster resampling."""
    order = np.argsort(labels, kind="mergesort")
    out, start = [], 0
    srt = labels[order]
    for i in range(1, len(srt) + 1):
        if i == len(srt) or srt[i] != srt[start]:
            out.append(order[start:i])
            start = i
    return out


def paired_bootstrap_clustered(pos_a, neg_a, pos_b, neg_b,
                               pos_clusters, neg_clusters, n_boot, seed):
    """
    Paired bootstrap that resamples clusters (genera), not individual queries.

    Both classes are clustered: LGO queries share a genus, and LSO knowns do
    too, since the LSO design retains the query's genus in the reference. Each
    replicate draws whole genera with replacement, so replicate sizes vary
    slightly; that is the intended behaviour of a cluster bootstrap and is why
    the resulting interval is wider than the query-level one.
    """
    rng = np.random.default_rng(seed)
    n_pc, n_nc = len(pos_clusters), len(neg_clusters)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ip = np.concatenate([pos_clusters[j]
                             for j in rng.integers(0, n_pc, n_pc)])
        ineg = np.concatenate([neg_clusters[j]
                               for j in rng.integers(0, n_nc, n_nc)])
        diffs[b] = (auroc(pos_a[ip], neg_a[ineg])
                    - auroc(pos_b[ip], neg_b[ineg]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"ci": (float(lo), float(hi)),
            "mean": float(diffs.mean()),
            "p": float(min(p, 1.0))}


def paired_bootstrap(pos_a, neg_a, pos_b, neg_b, n_boot, seed):
    """
    Stratified paired bootstrap of the AUROC difference.

    Knowns and novels are resampled separately, so each replicate preserves the
    class balance. Both scores are evaluated on the *same* resampled indices,
    which is what makes the interval a paired one.
    """
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(pos_a), len(neg_a)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ip = rng.integers(0, n_pos, n_pos)
        ineg = rng.integers(0, n_neg, n_neg)
        diffs[b] = (auroc(pos_a[ip], neg_a[ineg])
                    - auroc(pos_b[ip], neg_b[ineg]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value for H0: difference = 0.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"ci": (float(lo), float(hi)),
            "mean": float(diffs.mean()),
            "p": float(min(p, 1.0))}


# ----------------------------------------------------------------- conformal --
def conformal_p(anom_cal: np.ndarray, anom_query: np.ndarray) -> np.ndarray:
    """Split-conformal p-values, Eq. 8. Super-uniform under exchangeability."""
    n_cal = len(anom_cal)
    cal_sorted = np.sort(anom_cal)
    # #{i : A_i >= A(q)}
    ge = n_cal - np.searchsorted(cal_sorted, anom_query, side="left")
    return (1.0 + ge) / (n_cal + 1.0)


def mcnemar(flag_a: np.ndarray, flag_b: np.ndarray) -> dict:
    """Paired test on two binary flag vectors over the same queries."""
    b = int(np.sum(flag_a & ~flag_b))   # a flags, b misses
    c = int(np.sum(~flag_a & flag_b))   # b flags, a misses
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "test": "no discordant pairs"}
    if n < 25:
        p = float(stats.binomtest(b, n, 0.5).pvalue)
        test = "exact binomial"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n      # continuity-corrected
        p = float(stats.chi2.sf(chi2, 1))
        test = "chi-square, continuity corrected"
    return {"b": b, "c": c, "p": p, "test": test}


# ---------------------------------------------------------------------- main --
def load(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith((".tsv", ".tab", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    required = {"role", "identity", "cosine"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"error: missing column(s): {', '.join(sorted(missing))}")
    bad = set(df["role"].unique()) - set(ROLES)
    if bad:
        sys.exit(f"error: unexpected role value(s): {', '.join(sorted(map(str, bad)))}. "
                 f"Expected one of {ROLES}.")
    for role in ROLES:
        if (df["role"] == role).sum() == 0:
            sys.exit(f"error: no rows with role='{role}'.")
    if df[["identity", "cosine"]].isna().any().any():
        sys.exit("error: identity and cosine must be non-null for every query. "
                 "No-hit identity queries should carry the censoring value "
                 "(49.999), not NaN.")
    return df


def fmt_p(p: float) -> str:
    return "< 1e-16" if p < 1e-16 else f"{p:.3g}"


def fmt_p_tex(p: float) -> str:
    """LaTeX math-mode p-value, for the pasteable tablenotes line."""
    return "p < 10^{-16}" if p < 1e-16 else f"p = {p:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired uncertainty for identity vs cosine (Table 7).")
    ap.add_argument("--scores", required=True,
                    help="per-query score table (csv/tsv); see module docstring")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--latex", action="store_true",
                    help="also emit a tablenotes \\item line")
    args = ap.parse_args()

    df = load(args.scores)

    # Anomaly orientation: larger = more anomalous, matching A(q) = -S(q).
    cal = df[df.role == "cal"]
    known = df[df.role == "known"]
    novel = df[df.role == "novel"]

    a_id = {k: -g["identity"].to_numpy(float) for k, g in
            (("cal", cal), ("known", known), ("novel", novel))}
    a_cos = {k: -g["cosine"].to_numpy(float) for k, g in
             (("cal", cal), ("known", known), ("novel", novel))}

    print("=" * 72)
    print("Paired uncertainty, historical ITS2 benchmark")
    print("=" * 72)
    print(f"calibration knowns : {len(cal):>6,}")
    print(f"test knowns        : {len(known):>6,}")
    print(f"novel-genus queries: {len(novel):>6,}")

    has_genus = "genus" in df.columns
    if has_genus:
        n_gen = int(novel["genus"].nunique())
        n_missing = int(novel["genus"].isna().sum())
        print(f"novel genera       : {n_gen:>6,}")
        if n_missing:
            print(f"  warning: {n_missing:,}/{len(novel):,} novel queries have "
                  f"no genus label")
        if n_gen == 0 or n_gen >= 0.9 * len(novel):
            print("  the genus column carries no usable grouping "
                  "(one cluster per query, or none at all), so clustering is "
                  "DISABLED. Rebuild the score table with a working genus map "
                  "before quoting an interval.")
            has_genus = False

    n_cens = int((df.loc[df.role == "novel", "identity"] < 50.0).sum())
    if n_cens:
        print(f"censored LGO queries (identity < 50%): {n_cens:,} "
              f"({100 * n_cens / len(novel):.1f}%)")

    # ---- AUROC and its paired difference -----------------------------------
    pos = np.vstack([a_id["novel"], a_cos["novel"]])
    neg = np.vstack([a_id["known"], a_cos["known"]])
    d = delong_test(pos, neg)

    print("\n--- AUROC -------------------------------------------------------")
    print(f"percent identity : {d['auc_a']:.4f}  (DeLong SE {d['se_a']:.4f})")
    print(f"M4 cosine        : {d['auc_b']:.4f}  (DeLong SE {d['se_b']:.4f})")
    print(f"difference       : {d['diff']:+.4f}")
    print(f"  DeLong    95% CI {d['ci'][0]:+.4f} to {d['ci'][1]:+.4f}, "
          f"z = {d['z']:.2f}, p = {fmt_p(d['p'])}")

    bs_q = paired_bootstrap(a_id["novel"], a_id["known"],
                            a_cos["novel"], a_cos["known"],
                            args.n_boot, args.seed)
    print(f"  bootstrap 95% CI {bs_q['ci'][0]:+.4f} to {bs_q['ci'][1]:+.4f} "
          f"({args.n_boot:,} stratified paired resamples over queries, "
          f"seed {args.seed}), p = {fmt_p(bs_q['p'])}")

    bs = bs_q
    if has_genus:
        pos_cl = _cluster_index(novel["genus"].to_numpy())
        neg_cl = _cluster_index(known["genus"].to_numpy())
        bs_c = paired_bootstrap_clustered(
            a_id["novel"], a_id["known"], a_cos["novel"], a_cos["known"],
            pos_cl, neg_cl, args.n_boot, args.seed)
        print(f"  clustered 95% CI {bs_c['ci'][0]:+.4f} to "
              f"{bs_c['ci'][1]:+.4f} "
              f"({len(pos_cl):,} novel genera, {len(neg_cl):,} known genera "
              f"resampled), p = {fmt_p(bs_c['p'])}")
        widen = ((bs_c["ci"][1] - bs_c["ci"][0])
                 / max(bs_q["ci"][1] - bs_q["ci"][0], 1e-12))
        print(f"  clustering widens the interval {widen:.2f}x. "
              f"Report the clustered interval.")
        bs = bs_c
    else:
        print("  NOTE: no 'genus' column, so the interval above resamples "
              "queries as if independent and is too narrow. Add the column "
              "and rerun before quoting it.")

    width_delong = d["ci"][1] - d["ci"][0]
    width_boot = bs["ci"][1] - bs["ci"][0]
    if width_boot > 1.5 * width_delong:
        print("  NOTE: the reported bootstrap interval is much wider than "
              "DeLong's, as expected when clusters are correlated. DeLong "
              "assumes within-class independence; lead with the bootstrap.")

    # ---- conformal operating points ----------------------------------------
    p_id_known = conformal_p(a_id["cal"], a_id["known"])
    p_id_novel = conformal_p(a_id["cal"], a_id["novel"])
    p_cos_known = conformal_p(a_cos["cal"], a_cos["known"])
    p_cos_novel = conformal_p(a_cos["cal"], a_cos["novel"])

    print("\n--- conformal operating points, paired ---------------------------")
    hdr = (f"{'alpha':>6} {'FN id':>7} {'FN cos':>7} {'det id':>7} "
           f"{'det cos':>8} {'b':>6} {'c':>6} {'McNemar p':>12}")
    print(hdr)
    print("-" * len(hdr))
    mcn_rows = {}
    for alpha in args.alphas:
        fi, fc = p_id_known <= alpha, p_cos_known <= alpha
        di, dc = p_id_novel <= alpha, p_cos_novel <= alpha
        m = mcnemar(di, dc)
        mcn_rows[alpha] = m
        print(f"{alpha:>6.2f} {fi.mean():>6.1%} {fc.mean():>7.1%} "
              f"{di.mean():>7.1%} {dc.mean():>8.1%} "
              f"{m['b']:>6,} {m['c']:>6,} {fmt_p(m['p']):>12}")
    print("\nb = flagged by identity only, c = flagged by cosine only, "
          "over the same novel queries.")

    if args.latex:
        a05 = 0.05 if 0.05 in mcn_rows else args.alphas[0]
        m = mcn_rows[a05]
        print("\n--- paste into tables/table7_historical.tex tablenotes -----------")
        print(
            f"\\item The AUROC difference is $+{d['diff']:.3f}$ "
            f"(95\\% paired bootstrap {bs['ci'][0]:+.3f} to {bs['ci'][1]:+.3f}, "
            f"{args.n_boot:,} resamples"
            f"{' clustered on query genus' if has_genus else ''}; "
            f"DeLong $z={d['z']:.1f}$, "
            f"${fmt_p_tex(d['p'])}$). At $\\alpha={a05:g}$ the two scores flag "
            f"discordant novel queries {m['b']:,} to {m['c']:,} in favour of "
            f"identity (McNemar ${fmt_p_tex(m['p'])}$)."
        )


if __name__ == "__main__":
    main()
