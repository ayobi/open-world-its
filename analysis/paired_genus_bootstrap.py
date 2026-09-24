#!/usr/bin/env python3
"""Paired genus-cluster bootstrap of primary TEST identity versus corrected M4.

No embedding, training, alignment search, or threshold optimization is performed.
Novel and known genera are resampled separately, with all observed queries in
each sampled genus carried together. Both methods receive identical resampling
weights. Point estimates remain query-weighted. Detection intervals condition
on the observed CAL_KNOWN scores (calibration is not resampled).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

VIEWS = ("core", "its1", "its2")
PARTITIONS = ("cal_known", "test_known", "test_novel")
KEY = ["query_id", "partition", "view"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def conformal_p(cal_anomaly, query_anomaly):
    ordered = np.sort(np.asarray(cal_anomaly, dtype=float))
    if not len(ordered):
        raise ValueError("calibration population is empty")
    return (1 + len(ordered) - np.searchsorted(ordered, query_anomaly, side="left")) / (len(ordered) + 1)


class WeightedAUC:
    """Mann-Whitney AUROC with half credit for ties and per-row weights."""
    def __init__(self, scores, positive):
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("AUROC scores must be finite")
        _, self.group = np.unique(scores, return_inverse=True)
        self.positive = np.asarray(positive, dtype=bool)
        if not self.positive.any() or self.positive.all():
            raise ValueError("AUROC requires both classes")
        self.n_groups = int(self.group.max()) + 1

    def __call__(self, weights):
        weights = np.asarray(weights, dtype=float)
        pos = np.bincount(self.group[self.positive], weights=weights[self.positive], minlength=self.n_groups)
        neg = np.bincount(self.group[~self.positive], weights=weights[~self.positive], minlength=self.n_groups)
        denominator = pos.sum() * neg.sum()
        if denominator <= 0:
            raise ValueError("resample lost a class")
        return float(np.dot(pos, np.cumsum(neg) - 0.5 * neg) / denominator)


def load_paired(predictions, manifest, hits_dir, floor):
    pred = pd.read_csv(predictions, sep="\t", dtype=str, keep_default_na=False)
    required = {"query_id", "view", "split", "query_genus", "max_cosine", "conformal_p"}
    if not required.issubset(pred):
        raise ValueError(f"predictions missing columns: {sorted(required - set(pred))}")
    pred["partition"] = pred["split"].str.lower()
    if not pred.partition.isin(PARTITIONS).all() or not pred.view.isin(VIEWS).all():
        raise ValueError("predictions must contain only the three primary CAL/TEST partitions and views")
    if pred.duplicated(KEY).any() or pred.duplicated(["query_id", "view"]).any():
        raise ValueError("duplicate query/view rows in primary predictions")
    if (pred.query_genus.str.strip() == "").any():
        raise ValueError("missing query_genus; supply the evaluator's taxonomy, not parsed header names")
    for col in ("max_cosine", "conformal_p"):
        pred[col] = pd.to_numeric(pred[col], errors="raise")
        if not np.isfinite(pred[col]).all():
            raise ValueError(f"nonfinite {col}")
    if (pred.max_cosine.abs() > 1.00001).any():
        raise ValueError("cosine values outside [-1,1]")
    if not pred.conformal_p.between(0, 1).all():
        raise ValueError("conformal p-values outside [0,1]")
    man = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if not set(KEY).issubset(man):
        raise ValueError("manifest must contain query_id, partition, view")
    man["partition"] = man.partition.str.lower()
    if man.duplicated(KEY).any():
        raise ValueError("duplicate manifest rows")
    agreement = man[KEY].merge(pred[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one")
    if (agreement._merge != "both").any():
        raise ValueError("manifest and predictions do not have identical query/partition/view keys")
    frames, hit_paths = [], []
    for view in VIEWS:
        for part in PARTITIONS:
            sub = pred[(pred.view == view) & (pred.partition == part)].copy()
            if not len(sub):
                raise ValueError(f"missing prediction population: {part}/{view}")
            path = Path(hits_dir) / f"{part}_{view}_vs_{view}.exhaustive.tsv"
            if not path.is_file():
                raise ValueError(f"missing search file (cannot count an unsearched set as no-hit): {path}")
            hit_paths.append(path)
            try:
                hits = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
            except pd.errors.EmptyDataError:
                hits = pd.DataFrame(columns=range(3))
            if hits.shape[1] != 3:
                raise ValueError(f"expected query/target/identity columns in {path}")
            hits.columns = ["query_id", "target", "identity"]
            hits["identity"] = pd.to_numeric(hits.identity, errors="raise")
            if not np.isfinite(hits.identity).all() or not hits.identity.between(50, 100).all():
                raise ValueError(f"invalid identity or search floor differs from 50%: {path}")
            if not set(hits.query_id).issubset(set(sub.query_id)):
                raise ValueError(f"unexpected query IDs in {path}")
            # Only the scalar maximum matters here; no taxonomy-aware tie choice.
            best = hits.groupby("query_id", sort=False).identity.max()
            sub["identity"] = sub.query_id.map(best)
            sub["identity_hit"] = sub.identity.notna()
            sub["identity"] = sub.identity.fillna(floor)
            frames.append(sub)
    return pd.concat(frames, ignore_index=True), hit_paths


def cluster_info(labels):
    genera, codes = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    if len(genera) < 2:
        raise ValueError("at least two genera per test class are required for cluster uncertainty")
    sizes = np.bincount(codes)
    return genera, codes, {"n_genera": len(genera), "n_queries": len(codes),
                           "largest_genus_queries": int(sizes.max()),
                           "largest_genus_fraction": float(sizes.max() / sizes.sum())}


def analyze_view(sub, n_boot, seed, alpha, progress=False):
    cal = sub[sub.partition == "cal_known"]
    ev = sub[sub.partition.isin(("test_known", "test_novel"))].reset_index(drop=True)
    positive = (ev.partition == "test_novel").to_numpy()
    ni, pi = np.flatnonzero(~positive), np.flatnonzero(positive)
    known_genus, known_codes, known_info = cluster_info(ev.loc[ni, "query_genus"])
    novel_genus, novel_codes, novel_info = cluster_info(ev.loc[pi, "query_genus"])
    if set(known_genus) & set(novel_genus):
        raise ValueError("TEST_KNOWN and TEST_NOVEL share genus labels; primary split needs checking")
    scores = {"identity": -ev.identity.to_numpy(float), "cosine": -ev.max_cosine.to_numpy(float)}
    auc = {m: WeightedAUC(s, positive) for m, s in scores.items()}
    flags = {
        "identity": conformal_p(-cal.identity.to_numpy(float), scores["identity"]) <= alpha,
        # Use the full-precision evaluator's decisions stored in its predictions.
        "cosine": ev.conformal_p.to_numpy(float) <= alpha,
    }
    # Eight-decimal score serialization can introduce ties; expose this instead
    # of silently replacing the evaluator's full-precision calibrated decisions.
    reconstructed = conformal_p(-cal.max_cosine.to_numpy(float), scores["cosine"]) <= alpha
    flag_disagreements = int(np.count_nonzero(reconstructed != flags["cosine"]))
    def statistics(w):
        w_known, w_novel = w[ni], w[pi]
        result = {}
        for method in scores:
            result[f"auc_{method}"] = auc[method](w)
            result[f"false_novelty_{method}"] = float(np.dot(w_known, flags[method][ni]) / w_known.sum())
            result[f"detection_{method}"] = float(np.dot(w_novel, flags[method][pi]) / w_novel.sum())
        for metric in ("auc", "false_novelty", "detection"):
            result[f"{metric}_cosine_minus_identity"] = result[f"{metric}_cosine"] - result[f"{metric}_identity"]
        return result
    point = statistics(np.ones(len(ev)))
    draws = {key: np.empty(n_boot) for key in point}
    rng = np.random.default_rng(seed)
    nk, nn = len(known_genus), len(novel_genus)
    for b in range(n_boot):
        w = np.empty(len(ev), dtype=float)
        w[ni] = rng.multinomial(nk, np.full(nk, 1/nk))[known_codes]
        w[pi] = rng.multinomial(nn, np.full(nn, 1/nn))[novel_codes]
        for key, value in statistics(w).items():
            draws[key][b] = value
        if progress and (b + 1) % 2000 == 0:
            print(f"  bootstrap {b + 1:,}/{n_boot:,}", flush=True)
    result = {
        "known": known_info, "novel": novel_info, "n_cal_known": len(cal),
        "censored_known": int((~ev.loc[ni, "identity_hit"]).sum()),
        "censored_novel": int((~ev.loc[pi, "identity_hit"]).sum()),
        "cosine_flag_disagreements_from_score_rounding": flag_disagreements,
        "metrics": {key: {"estimate": value, "ci95_percentile": np.quantile(draws[key], [.025, .975]).tolist()}
                    for key, value in point.items()},
        "auroc_difference_ci98_333_percentile_bonferroni_3_views":
            np.quantile(draws["auc_cosine_minus_identity"], [.05/(2*3), 1-.05/(2*3)]).tolist(),
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=Path("runs/m4_final_exact_v1/final_predictions.tsv"))
    ap.add_argument("--manifest", type=Path, default=Path("data/openworld/test_query_manifest.csv"))
    ap.add_argument("--hits-dir", type=Path, default=Path("runs/identity_cov08"))
    ap.add_argument("--out", type=Path, default=Path("runs/m4_final_exact_v1/paired_genus_bootstrap.json"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--alpha", type=float, default=.05)
    args = ap.parse_args()
    if args.boot < 1000 or not 0 < args.alpha < 1:
        ap.error("use at least 1000 bootstrap replicates and 0 < alpha < 1")
    if args.out.exists():
        ap.error("output exists; choose a new --out path")
    df, paths = load_paired(args.predictions, args.manifest, args.hits_dir, 49.999)
    result = {
        "method": "paired genus-cluster percentile bootstrap; stratified by test class",
        "estimand": "query-weighted AUROC/rates; genera resampled with all observed queries",
        "difference_direction": "cosine minus identity",
        "conditioning": "fixed trained model, references, scores, and CAL_KNOWN calibration",
        "scope": "primary same-view evaluation only; not the overlapping historical benchmark",
        "limitations": "assumes independent sampled genera within each class; does not account for family-level dependence, training or calibration variability",
        "boot": args.boot, "seed": args.seed, "alpha": args.alpha,
        "identity_censoring_value": 49.999,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "source_sha256": sha256(__file__),
        "input_sha256": {str(p): sha256(p) for p in [args.predictions, args.manifest, *paths]},
        "views": {},
    }
    print("Differences below are COSINE MINUS IDENTITY (opposite sign to paired_test.py).", flush=True)
    for view in VIEWS:
        print(f"\n=== {view} ===", flush=True)
        r = analyze_view(df[df.view == view], args.boot, args.seed, args.alpha, progress=True)
        result["views"][view] = r
        metrics = r["metrics"]
        difference = metrics["auc_cosine_minus_identity"]
        lo, hi = difference["ci95_percentile"]
        print(f"  genera: {r['known']['n_genera']:,} known; {r['novel']['n_genera']:,} novel")
        print(f"  identity {metrics['auc_identity']['estimate']:.4f}; cosine {metrics['auc_cosine']['estimate']:.4f}")
        print(f"  paired difference {difference['estimate']:+.4f}; genus-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
        lo, hi = r["auroc_difference_ci98_333_percentile_bonferroni_3_views"]
        print(f"  Bonferroni 3-view interval [{lo:+.4f}, {hi:+.4f}]")
        for method in ("identity", "cosine"):
            print(f"  {method}: false novelty {100*metrics[f'false_novelty_{method}']['estimate']:.2f}%; "
                  f"detection {100*metrics[f'detection_{method}']['estimate']:.2f}%")
        for metric in ("false_novelty", "detection"):
            m = metrics[f"{metric}_cosine_minus_identity"]
            lo, hi = m["ci95_percentile"]
            print(f"  {metric} difference {100*m['estimate']:+.2f} pp; 95% CI [{100*lo:+.2f}, {100*hi:+.2f}] pp")
        if r["cosine_flag_disagreements_from_score_rounding"]:
            print("  NOTE: rounded-score reconstruction changes some flags; stored evaluator p-values used.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {args.out}\nRate intervals condition on the observed calibration set; no thresholds were tuned.")


if __name__ == "__main__":
    main()
