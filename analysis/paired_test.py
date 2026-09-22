#!/usr/bin/env python3
"""Paired comparison of percent identity and M4 cosine on the TEST partitions.

Both scores now exist on identical queries, so the AUROC comparison should be
paired rather than two independent bootstrap intervals that happen to overlap.
This runs a DeLong test and a query-level paired bootstrap per view, decomposes
the effect of censored (unalignable) queries, and reports how much the two
flagged sets actually overlap at a chosen alpha.

None of this selects anything. Both scores are fixed, the conformal threshold
comes from CAL_KNOWN by the procedure already in the manuscript, and every
quantity here is a decomposition of results already reported.

-----------------------------------------------------------------------------
INPUTS
-----------------------------------------------------------------------------
--hits      the per-search identity TSVs, named <partition>_<view>_vs_<ref>.*
            exactly as run_identity_searches.sh writes them.
--manifest  data/openworld/test_query_manifest.csv (query_id, partition, view)
--cosine    CSV with columns: query_id, view, cosine
            One row per query-view, the maximum cosine against the matching
            TRAIN_REF view, from the frozen M4 checkpoint. This is the same
            quantity that produced Table 5, so look for it in runs/ before
            recomputing it.

-----------------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------------
    python3 analysis/paired_test.py \
        --hits $(for V in core its1 its2; do for P in cal_known test_known test_novel; do \
                   echo runs/identity_cov08/${P}_${V}_vs_${V}.exhaustive.tsv; done; done) \
        --manifest data/openworld/test_query_manifest.csv \
        --cosine data/m4_test_cosine.csv \
        --out data/paired_test_comparison.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

VIEWS = ["core", "its1", "its2"]
NAME_RE = re.compile(
    r"^(cal_known|test_known|test_novel)_(core|its1|its2)_vs_(core|its1|its2)\b")


# ---- fast DeLong (Sun & Xu). Midranks handle the block of tied censored
# ---- identities correctly; a naive AUROC would not.
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N)
    T2[J] = T + 1
    return T2


def delong(scores, positive):
    scores = np.atleast_2d(np.asarray(scores, float))
    positive = np.asarray(positive, bool)
    order = np.argsort(-positive.astype(int), kind="mergesort")
    s = scores[:, order]
    m = int(positive.sum())
    n = s.shape[1] - m
    k = s.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(s[r, :m])
        ty[r] = _midrank(s[r, m:])
        tz[r] = _midrank(s[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def auroc(score, positive):
    r = stats.rankdata(score)
    m = int(positive.sum())
    n = len(score) - m
    return float((r[positive].sum() - m * (m + 1) / 2) / (m * n))


def delong_test(a, b, positive):
    aucs, cov = delong(np.vstack([a, b]), positive)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    diff = float(aucs[0] - aucs[1])
    z = diff / se if se > 0 else np.inf
    return {"auc_identity": float(aucs[0]), "auc_cosine": float(aucs[1]),
            "diff": diff, "se": se, "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z))),
            "ci95": [diff - 1.96 * se, diff + 1.96 * se]}


def paired_bootstrap(a, b, positive, n_boot=10000, seed=17):
    rng = np.random.default_rng(seed)
    pi = np.flatnonzero(positive)
    ni = np.flatnonzero(~positive)
    d = np.empty(n_boot)
    for t in range(n_boot):
        i = np.concatenate([rng.choice(pi, pi.size, True),
                            rng.choice(ni, ni.size, True)])
        lab = np.zeros(i.size, bool)
        lab[:pi.size] = True
        d[t] = auroc(a[i], lab) - auroc(b[i], lab)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)]}


def conformal_p(cal_anom, q_anom):
    n = cal_anom.size
    ge = (cal_anom[None, :] >= q_anom[:, None]).sum(axis=1)
    return (1 + ge) / (n + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", type=Path, nargs="+", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cosine", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=49.999)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path("data/paired_test_comparison.json"))
    args = ap.parse_args()

    frames, searched = [], set()
    for p in args.hits:
        m = NAME_RE.match(p.name)
        if not m:
            raise SystemExit(f"cannot read partition/view from '{p.name}'")
        part, qv, _ = m.groups()
        df = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["query_id", "target", "identity"], dtype={0: str, 1: str})
        df["view"] = qv
        frames.append(df)
        searched.add((part, qv))
    hits = (pd.concat(frames, ignore_index=True)
              .sort_values("identity", ascending=False)
              .drop_duplicates(["query_id", "view"], keep="first"))

    man = pd.read_csv(args.manifest)
    man = man[[(p, v) in searched for p, v in zip(man.partition, man.view)]].copy()

    cos = pd.read_csv(args.cosine)
    for c in ("query_id", "view", "cosine"):
        if c not in cos.columns:
            raise SystemExit(f"--cosine file is missing column '{c}'")
    cos = cos.drop_duplicates(["query_id", "view"])

    df = (man.merge(hits[["query_id", "view", "identity"]],
                    on=["query_id", "view"], how="left", validate="one_to_one")
             .merge(cos[["query_id", "view", "cosine"]],
                    on=["query_id", "view"], how="left", validate="one_to_one"))
    miss = df["cosine"].isna().sum()
    if miss:
        raise SystemExit(
            f"{miss:,} query-view rows have no cosine score. The two files must "
            "cover the same queries or the comparison is not paired.")
    df["identity_hit"] = df["identity"].notna().astype(int)
    df["identity"] = df["identity"].fillna(args.floor)

    res = {"alpha": args.alpha, "views": {}}
    for v in VIEWS:
        sub = df[df.view == v]
        cal = sub[sub.partition == "cal_known"]
        ev = sub[sub.partition.isin(["test_known", "test_novel"])]
        if not len(cal) or not len(ev):
            continue
        pos = (ev.partition == "test_novel").to_numpy()
        a_id = -ev["identity"].to_numpy(float)
        a_cos = -ev["cosine"].to_numpy(float)

        r = {"n_known": int((~pos).sum()), "n_novel": int(pos.sum())}
        r["delong"] = delong_test(a_id, a_cos, pos)
        r["bootstrap"] = paired_bootstrap(a_id, a_cos, pos, args.boot, args.seed)

        hit = ev["identity_hit"].to_numpy(bool)
        r["censoring"] = {
            "censored_novel": int((~hit & pos).sum()),
            "censored_known": int((~hit & ~pos).sum()),
            "auroc_alignable": {
                "identity": auroc(a_id[hit], pos[hit]),
                "cosine": auroc(a_cos[hit], pos[hit])},
        }

        p_id = conformal_p(-cal["identity"].to_numpy(float), a_id)
        p_cos = conformal_p(-cal["cosine"].to_numpy(float), a_cos)
        f_id, f_cos = p_id <= args.alpha, p_cos <= args.alpha
        both = int((f_id & f_cos & pos).sum())
        oi = int((f_id & ~f_cos & pos).sum())
        oc = int((~f_id & f_cos & pos).sum())
        r["agreement"] = {
            "spearman_novel": float(stats.spearmanr(
                ev.loc[pos, "identity"], ev.loc[pos, "cosine"]).statistic),
            "both": both, "identity_only": oi, "cosine_only": oc,
            "jaccard": both / (both + oi + oc) if (both + oi + oc) else 0.0,
            "union_detection": (both + oi + oc) / max(int(pos.sum()), 1),
            "union_false_novelty": float((f_id | f_cos)[~pos].mean()),
        }
        res["views"][v] = r

        d, b, c, g = r["delong"], r["bootstrap"], r["censoring"], r["agreement"]
        print(f"\n=== {v} ===")
        print(f"  identity {d['auc_identity']:.3f}   cosine {d['auc_cosine']:.3f}   "
              f"difference {d['diff']:+.3f}")
        print(f"  DeLong 95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}, P = {d['p']:.3g}")
        print(f"  paired bootstrap 95% CI {b['ci95'][0]:+.3f} to {b['ci95'][1]:+.3f}")
        print(f"  restricted to alignable queries: identity "
              f"{c['auroc_alignable']['identity']:.3f}, cosine "
              f"{c['auroc_alignable']['cosine']:.3f} "
              f"({c['censored_novel']} novel, {c['censored_known']} known censored)")
        print(f"  at alpha={args.alpha}: both {g['both']}, identity only "
              f"{g['identity_only']}, cosine only {g['cosine_only']}, "
              f"Jaccard {g['jaccard']:.2f}, Spearman {g['spearman_novel']:.2f}")
        print(f"  union would detect {100*g['union_detection']:.1f}% at "
              f"{100*g['union_false_novelty']:.1f}% false novelty (descriptive only)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    print("\nThe union figures describe how much the two scores disagree. They are "
          "not a combined method: the rule was chosen after seeing these data, so "
          "reporting it as a result would need a fresh holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
