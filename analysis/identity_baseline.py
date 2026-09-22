#!/usr/bin/env python3
"""Percent-identity baseline on the held-out TEST partitions.

Why this is safe to run on an already-opened TEST set: it selects nothing. The
reference collection is fixed, the score is fixed, and the threshold comes from
CAL_KNOWN through the same split-conformal procedure the manuscript already
uses. There is no free parameter, so nothing leaks back into a frozen model. It
adds a baseline row; it does not re-open model selection.

IMPORTANT: hits are keyed by (query_id, view), not by query_id alone. The same
record contributes an ITS-core, an ITS1 and an ITS2 query, all carrying the same
id, so a view-blind merge silently produces a cartesian product and assigns one
view's identity to another view's row. Pass the per-search hit files directly
and let the script read the view from each filename.

Usage
-----
Same-view baseline, all three views in one call:

    python3 analysis/identity_baseline.py \
        --hits runs/identity_exhaustive/{cal_known,test_known,test_novel}_{core,its1,its2}_vs_{core,its1,its2}.exhaustive.tsv \
        --manifest data/openworld/test_query_manifest.csv \
        --ref-taxonomy data/openworld/ref_taxonomy.csv \
        --query-taxonomy data/openworld/query_taxonomy.csv \
        --outdir data

In practice let the shell expand only the matched pairs:

    python3 analysis/identity_baseline.py --outdir data \
        --hits $(for V in core its1 its2; do for P in cal_known test_known test_novel; do \
                   echo runs/identity_exhaustive/${P}_${V}_vs_${V}.exhaustive.tsv; done; done) \
        --manifest data/openworld/test_query_manifest.csv \
        --ref-taxonomy data/openworld/ref_taxonomy.csv \
        --query-taxonomy data/openworld/query_taxonomy.csv

Cross-view (each spacer against the ITS-core references):

    python3 analysis/identity_baseline.py --outdir data/crossview \
        --hits $(for V in its1 its2; do for P in cal_known test_known test_novel; do \
                   echo runs/identity_exhaustive/${P}_${V}_vs_core.exhaustive.tsv; done; done) \
        --manifest data/openworld/test_query_manifest.csv \
        --ref-taxonomy data/openworld/ref_taxonomy.csv \
        --query-taxonomy data/openworld/query_taxonomy.csv

Filenames must be <partition>_<queryview>_vs_<refview>.*, which is what
run_identity_searches.sh writes. For any other naming, pass a single file plus
--view and --partition.

Needs pandas and scipy:  pip install pandas scipy

Writes <outdir>/test_identity_metrics.csv and <outdir>/test_identity_conformal.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

VIEWS = ["core", "its1", "its2"]
PARTS = ["cal_known", "test_known", "test_novel"]
ALPHAS = [0.01, 0.05, 0.10, 0.20]
NAME_RE = re.compile(
    r"^(cal_known|test_known|test_novel)_(core|its1|its2)_vs_(core|its1|its2)\b")


def auroc(score, positive):
    r = stats.rankdata(score)
    m = int(positive.sum())
    n = len(score) - m
    return float((r[positive].sum() - m * (m + 1) / 2) / (m * n))


def boot_ci(score, positive, n_boot=2000, seed=17):
    rng = np.random.default_rng(seed)
    pi = np.flatnonzero(positive)
    ni = np.flatnonzero(~positive)
    out = np.empty(n_boot)
    for t in range(n_boot):
        i = np.concatenate([rng.choice(pi, pi.size, True),
                            rng.choice(ni, ni.size, True)])
        lab = np.zeros(i.size, bool)
        lab[:pi.size] = True
        out[t] = auroc(score[i], lab)
    return tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))


def conformal_p(cal_anom, q_anom):
    n = cal_anom.size
    ge = (cal_anom[None, :] >= q_anom[:, None]).sum(axis=1)
    return (1 + ge) / (n + 1)


def read_hits(paths, view=None, partition=None):
    """Read per-search hit files, tagging each with its partition and views."""
    frames, combos = [], []
    for p in paths:
        p = Path(p)
        m = NAME_RE.match(p.name)
        if m:
            part, qview, rview = m.groups()
        elif view and partition:
            part, qview, rview = partition, view, view
        else:
            raise SystemExit(
                f"cannot read partition/view from '{p.name}'. Either rename to "
                "<partition>_<queryview>_vs_<refview>.tsv or pass a single file "
                "with --view and --partition.")
        df = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["query_id", "target", "identity"],
                         dtype={0: str, 1: str})
        df["identity"] = pd.to_numeric(df["identity"], errors="coerce")
        if df["identity"].isna().any():
            raise SystemExit(f"non-numeric identity value in {p}")
        df["partition"] = part
        df["view"] = qview
        df["reference_view"] = rview
        frames.append(df)
        combos.append((part, qview, rview))
        print(f"  read {p.name}: {len(df):,} rows "
              f"({part}, query {qview} vs reference {rview})")
    if not frames:
        raise SystemExit("no hit files given")
    hits = pd.concat(frames, ignore_index=True)
    # one best hit per (query_id, view); ties resolved by identity
    n_before = len(hits)
    hits = (hits.sort_values("identity", ascending=False)
                .drop_duplicates(["query_id", "view"], keep="first")
                .reset_index(drop=True))
    if len(hits) != n_before:
        print(f"  note: collapsed {n_before - len(hits):,} extra tied row(s)")
    return hits, combos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", type=Path, nargs="+", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ref-taxonomy", type=Path)
    ap.add_argument("--query-taxonomy", type=Path)
    ap.add_argument("--view", help="only with a single, non-standard filename")
    ap.add_argument("--partition", help="only with a single, non-standard filename")
    ap.add_argument("--floor", type=float, default=49.999)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    args = ap.parse_args()

    print("reading hits")
    hits, combos = read_hits(args.hits, args.view, args.partition)
    searched = {(p, v) for p, v, _ in combos}
    refview = {v: r for _, v, r in combos}

    man = pd.read_csv(args.manifest)
    for c in ("query_id", "partition", "view"):
        if c not in man.columns:
            raise SystemExit(f"manifest is missing column '{c}'")
    # only the partition/view combinations that were actually searched, so that
    # an unsearched combination is never mistaken for a censored query
    man = man[[(p, v) in searched
               for p, v in zip(man.partition, man.view)]].copy()

    df = man.merge(hits.drop(columns=["partition"]), on=["query_id", "view"],
                   how="left", validate="one_to_one")
    censored = df["identity"].isna()
    df["identity_hit"] = (~censored).astype(int)
    df.loc[censored, "identity"] = args.floor
    print(f"\n{len(df):,} query-view rows; {int(censored.sum()):,} with no hit "
          f"({100*censored.mean():.2f}%), censored at {args.floor}")

    tax = None
    if args.ref_taxonomy and args.query_taxonomy:
        rt = pd.read_csv(args.ref_taxonomy).set_index("ref_id")
        qt = pd.read_csv(args.query_taxonomy).set_index("query_id")
        tax = (rt, qt)

    rows, conf_rows = [], []
    for v in VIEWS:
        sub = df[df.view == v]
        cal = sub[sub.partition == "cal_known"]
        kn = sub[sub.partition == "test_known"]
        nv = sub[sub.partition == "test_novel"]
        if not len(cal) or not len(kn) or not len(nv):
            print(f"  [{v}] skipped: not all of cal/test_known/test_novel searched")
            continue

        ev = pd.concat([kn, nv])
        pos = (ev.partition == "test_novel").to_numpy()
        anom = -ev["identity"].to_numpy(float)
        a = auroc(anom, pos)
        lo, hi = boot_ci(anom, pos, args.boot, args.seed)

        rec = {"view": v, "reference_view": refview.get(v, v),
               "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
               "n_test_known": len(kn), "n_test_novel": len(nv),
               "censored_known": int((kn.identity_hit == 0).sum()),
               "censored_novel": int((nv.identity_hit == 0).sum())}

        if tax is not None:
            rt, qt = tax

            def agreement(frame, rank):
                """Fraction of `frame` whose best hit shares `rank` with the
                query. Queries with no hit cannot place, so they count as
                failures: the denominator is the whole partition."""
                if rank not in rt.columns or rank not in qt.columns:
                    return None
                hit = frame.dropna(subset=["target"])
                ok = sum(
                    r.target in rt.index and r.query_id in qt.index
                    and rt.at[r.target, rank] == qt.at[r.query_id, rank]
                    for r in hit.itertuples()
                )
                return float(ok) / len(frame)

            g = agreement(kn, "genus")
            if g is not None:
                rec["known_genus"] = g
            for rank in ("family", "order", "class"):
                val = agreement(nv, rank)
                if val is not None:
                    rec[f"novel_{rank}"] = val

        rows.append(rec)

        cal_anom = -cal["identity"].to_numpy(float)
        for al in ALPHAS:
            fk = conformal_p(cal_anom, -kn["identity"].to_numpy(float)) <= al
            fn = conformal_p(cal_anom, -nv["identity"].to_numpy(float)) <= al
            conf_rows.append({"view": v, "reference_view": refview.get(v, v),
                              "alpha": al,
                              "false_novelty": float(fk.mean()),
                              "novel_detection": float(fn.mean())})
        extra = "" if rec["reference_view"] == v else f" vs {rec['reference_view']} refs"
        fam = f", novel family {100*rec['novel_family']:.1f}%" if "novel_family" in rec else ""
        print(f"  [{v}{extra}] identity AUROC {a:.3f} ({lo:.3f}-{hi:.3f}){fam}")

    if not rows:
        raise SystemExit("nothing computed; check that cal, test_known and "
                         "test_novel were all searched for at least one view")

    args.outdir.mkdir(parents=True, exist_ok=True)
    m = args.outdir / "test_identity_metrics.csv"
    c = args.outdir / "test_identity_conformal.csv"
    pd.DataFrame(rows).to_csv(m, index=False)
    pd.DataFrame(conf_rows).to_csv(c, index=False)
    print(f"\nwrote {m}\nwrote {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
