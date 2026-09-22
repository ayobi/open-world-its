#!/usr/bin/env python3
"""Locked CAL/TEST evaluation for the frozen open-world model.

Primary evaluation is same-locus retrieval for core, ITS1 and ITS2.  TRAIN_REF is the
reference library, CAL_KNOWN is the sole conformal calibration population, and
TEST_KNOWN / TEST_NOVEL are evaluated exactly once.  DEV rows are ignored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from itsnet.openworld.benchmark import VIEWS, embed_rows
from itsnet.openworld.conformal import conformal_pvalues, novelty_flags
from itsnet.openworld.m0 import binary_auroc, nearest_reference
from itsnet.openworld.model import BarcodeEncoder

REQUIRED_SPLITS = (
    "TRAIN_REF", "DEV_KNOWN", "DEV_NOVEL", "CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"
)
FINAL_SPLITS = ("TRAIN_REF", "CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def parse_alphas(text: str):
    vals = []
    for tok in text.split(","):
        x = float(tok.strip())
        if not 0.0 < x < 1.0:
            raise argparse.ArgumentTypeError("alphas must lie strictly between 0 and 1")
        vals.append(x)
    if not vals:
        raise argparse.ArgumentTypeError("at least one alpha is required")
    return tuple(vals)


def load_final_rows(path: Path):
    rows = []
    counts = Counter()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"id", "family", "genus", "species", "split", *VIEWS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise SystemExit("split TSV missing columns: " + ", ".join(sorted(missing)))
        for r in reader:
            counts[r["split"]] += 1
            if r["split"] in FINAL_SPLITS:
                rows.append(r)
    missing_splits = [s for s in REQUIRED_SPLITS if counts[s] == 0]
    if missing_splits:
        raise SystemExit("required six-way splits missing: " + ", ".join(missing_splits))
    return rows, counts


def _mean_bool(xs):
    return float(np.mean(xs)) if xs else float("nan")


def macro_detection(rows, flags, field):
    groups = defaultdict(list)
    for r, flag in zip(rows, flags):
        key = r.get(field, "")
        if key:
            groups[key].append(bool(flag))
    if not groups:
        return float("nan"), 0
    rates = [float(np.mean(v)) for v in groups.values()]
    return float(np.mean(rates)), len(groups)


def score_against_reference(query_rows, query_z, ref_rows, ref_z, device, nn_chunk):
    scores, idxs = nearest_reference(query_z.to(device), ref_z.to(device), nn_chunk)
    return scores.numpy(), idxs.numpy()


def placement_metrics(query_rows, idxs, ref_rows, *, known: bool):
    if known:
        return {
            "nn_genus_accuracy": _mean_bool([
                q["genus"] == ref_rows[int(ix)]["genus"]
                for q, ix in zip(query_rows, idxs)
            ])
        }
    return {
        "nn_family_accuracy": _mean_bool([
            q["family"] == ref_rows[int(ix)]["family"]
            for q, ix in zip(query_rows, idxs)
        ]),
        "nn_order_accuracy": _mean_bool([
            bool(q.get("order", "")) and q.get("order", "") == ref_rows[int(ix)].get("order", "")
            for q, ix in zip(query_rows, idxs)
        ]),
        "nn_class_accuracy": _mean_bool([
            bool(q.get("class", "")) and q.get("class", "") == ref_rows[int(ix)].get("class", "")
            for q, ix in zip(query_rows, idxs)
        ]),
    }


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--frozen-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--alphas", type=parse_alphas, default=parse_alphas("0.01,0.05,0.10,0.20"))
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--nn-chunk", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    frozen_dir = Path(args.frozen_dir).resolve()
    manifest_path = frozen_dir / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"freeze manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "FROZEN":
        raise SystemExit("manifest is not marked FROZEN")

    checkpoint_path = frozen_dir / manifest["checkpoint_file"]
    if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise SystemExit("FROZEN CHECKPOINT HASH MISMATCH -- refusing final evaluation")
    if sha256_file(input_path) != manifest["split_tsv_sha256"]:
        raise SystemExit("SPLIT TSV HASH MISMATCH -- refusing final evaluation")

    rows, split_counts = load_final_rows(input_path)
    device = torch.device(args.device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    carg = ckpt.get("args", {})
    model = BarcodeEncoder(
        channels=int(carg.get("channels", 64)),
        embedding_dim=int(carg.get("embedding_dim", 256)),
        dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print("LOCKED FINAL EVALUATION")
    print(f"experiment     : {ckpt.get('experiment', 'unknown')}")
    print(f"checkpoint sha : {manifest['checkpoint_sha256']}")
    print(f"dataset sha    : {manifest['split_tsv_sha256']}")
    print(f"device         : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print("calibration    : CAL_KNOWN only")
    print("test           : TEST_KNOWN vs TEST_NOVEL only")
    print("DEV            : not embedded or scored")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": ckpt.get("experiment", "unknown"),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "split_tsv_sha256": manifest["split_tsv_sha256"],
        "alphas": list(args.alphas),
        "novelty_score": "negative max cosine to same-view TRAIN_REF",
        "conformal_pvalue": "(1 + # calibration novelty scores >= query novelty score) / (n_cal + 1)",
        "input_split_counts": dict(split_counts),
        "views": {},
    }
    prediction_rows = []

    for view in VIEWS:
        subsets = {}
        for split in FINAL_SPLITS:
            sr = [r for r in rows if r["split"] == split]
            rr, zz = embed_rows(model, sr, view, device, args.eval_batch_size)
            subsets[split] = (rr, zz)

        ref_rows, ref_z = subsets["TRAIN_REF"]
        cal_rows, cal_z = subsets["CAL_KNOWN"]
        tk_rows, tk_z = subsets["TEST_KNOWN"]
        tn_rows, tn_z = subsets["TEST_NOVEL"]
        if min(map(len, (ref_rows, cal_rows, tk_rows, tn_rows))) == 0:
            raise RuntimeError(f"{view}: one or more final-evaluation sets are empty")

        cal_cos, cal_idx = score_against_reference(cal_rows, cal_z, ref_rows, ref_z, device, args.nn_chunk)
        tk_cos, tk_idx = score_against_reference(tk_rows, tk_z, ref_rows, ref_z, device, args.nn_chunk)
        tn_cos, tn_idx = score_against_reference(tn_rows, tn_z, ref_rows, ref_z, device, args.nn_chunk)

        cal_novelty = -cal_cos
        tk_novelty = -tk_cos
        tn_novelty = -tn_cos
        tk_p = conformal_pvalues(cal_novelty, tk_novelty)
        tn_p = conformal_pvalues(cal_novelty, tn_novelty)

        auc = binary_auroc(
            [0] * len(tk_novelty) + [1] * len(tn_novelty),
            list(tk_novelty) + list(tn_novelty),
        )
        metrics = {
            "n_reference": len(ref_rows),
            "n_cal_known": len(cal_rows),
            "n_test_known": len(tk_rows),
            "n_test_novel": len(tn_rows),
            "cal_known_max_cosine_mean": float(np.mean(cal_cos)),
            "test_known_max_cosine_mean": float(np.mean(tk_cos)),
            "test_novel_max_cosine_mean": float(np.mean(tn_cos)),
            "test_novelty_auroc": float(auc),
            "test_known_placement": placement_metrics(tk_rows, tk_idx, ref_rows, known=True),
            "test_novel_placement": placement_metrics(tn_rows, tn_idx, ref_rows, known=False),
            "operating_points": {},
        }
        for alpha in args.alphas:
            k_flags = novelty_flags(tk_p, alpha)
            n_flags = novelty_flags(tn_p, alpha)
            macro_genus, n_genera = macro_detection(tn_rows, n_flags, "genus")
            metrics["operating_points"][f"{alpha:g}"] = {
                "alpha": float(alpha),
                "known_false_novelty_rate": float(np.mean(k_flags)),
                "novel_detection_rate": float(np.mean(n_flags)),
                "novel_detection_macro_genus": macro_genus,
                "n_novel_genera": n_genera,
            }
        report["views"][view] = metrics

        for split, qs, cos, idxs, ps in (
            ("CAL_KNOWN", cal_rows, cal_cos, cal_idx, conformal_pvalues(cal_novelty, cal_novelty)),
            ("TEST_KNOWN", tk_rows, tk_cos, tk_idx, tk_p),
            ("TEST_NOVEL", tn_rows, tn_cos, tn_idx, tn_p),
        ):
            for i, q in enumerate(qs):
                ref = ref_rows[int(idxs[i])]
                row = {
                    "view": view,
                    "split": split,
                    "query_id": q["id"],
                    "query_family": q["family"],
                    "query_genus": q["genus"],
                    "query_species": q["species"],
                    "max_cosine": f"{float(cos[i]):.8f}",
                    "novelty_score": f"{-float(cos[i]):.8f}",
                    "conformal_p": f"{float(ps[i]):.10g}",
                    "nn_id": ref["id"],
                    "nn_family": ref["family"],
                    "nn_genus": ref["genus"],
                    "nn_species": ref["species"],
                }
                for alpha in args.alphas:
                    row[f"novel_at_{alpha:g}"] = int(float(ps[i]) <= alpha)
                prediction_rows.append(row)

        op05 = metrics["operating_points"].get("0.05")
        print(
            f"{view:5s}  AUROC={auc:.3f}  "
            f"known_cos={np.mean(tk_cos):.3f}  novel_cos={np.mean(tn_cos):.3f}  "
            f"known_genus={metrics['test_known_placement']['nn_genus_accuracy']:.3f}  "
            f"novel_family={metrics['test_novel_placement']['nn_family_accuracy']:.3f}"
        )
        if op05:
            print(
                f"       alpha=.05 false_novel={op05['known_false_novelty_rate']:.3f}  "
                f"novel_detect={op05['novel_detection_rate']:.3f}  "
                f"macro_genus={op05['novel_detection_macro_genus']:.3f}"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_path = outdir / "final_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = [
        "view", "split", "query_id", "query_family", "query_genus", "query_species",
        "max_cosine", "novelty_score", "conformal_p", "nn_id", "nn_family", "nn_genus", "nn_species",
        *[f"novel_at_{a:g}" for a in args.alphas],
    ]
    pred_path = outdir / "final_predictions.tsv"
    with pred_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(prediction_rows)

    print(f"\nwrote {metrics_path}")
    print(f"wrote {pred_path}")
    print("FINAL SPLITS HAVE NOW BEEN OPENED. Do not return to DEV model tuning.")


if __name__ == "__main__":
    main()
