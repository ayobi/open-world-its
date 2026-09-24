#!/usr/bin/env python3
"""DEV-only same-locus and cross-locus evaluation of an open-world checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from itsnet.openworld.inference_exact import configure_inference, inference_metadata, sha256_file

from itsnet.openworld.benchmark import (
    VIEWS,
    evaluate_embedding_pair,
    load_dev_rows,
)
from itsnet.openworld.model import BarcodeEncoder
from itsnet.openworld.inference_exact import embed_dev_matrix_exact as embed_dev_matrix


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--nn-chunk", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write-predictions", action="store_true")
    args = ap.parse_args()
    if args.eval_batch_size < 1 or args.nn_chunk < 1:
        ap.error("batch size and nn-chunk must be positive")
    configure_inference()

    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit("Output directory must be new or empty; retain the original evaluation.")
    device = torch.device(args.device)
    rows, split_counts = load_dev_rows(args.input)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    carg = ckpt.get("args", {})
    model = BarcodeEncoder(
        channels=int(carg.get("channels", 64)),
        embedding_dim=int(carg.get("embedding_dim", 256)),
        dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print(f"checkpoint    : {ckpt.get('experiment', 'unknown')}")
    print(f"device        : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print("protected     : CAL_KNOWN / TEST_KNOWN / TEST_NOVEL not evaluated")
    emb = embed_dev_matrix(model, rows, device, args.eval_batch_size)

    metrics = []
    preds = []
    for qv in VIEWS:
        for rv in VIEWS:
            ref_rows, ref_z = emb[rv]["TRAIN_REF"]
            known_rows, known_z = emb[qv]["DEV_KNOWN"]
            novel_rows, novel_z = emb[qv]["DEV_NOVEL"]
            m, p = evaluate_embedding_pair(
                ref_rows, ref_z, known_rows, known_z, novel_rows, novel_z,
                query_view=qv, reference_view=rv, device=device, nn_chunk=args.nn_chunk,
            )
            metrics.append(m)
            if args.write_predictions:
                preds.extend(p)

    print("\nDEV retrieval matrix (query -> reference; TEST/CAL untouched)")
    for m in metrics:
        print(
            f"{m['query_view']:5s}->{m['reference_view']:5s}  "
            f"AUROC={m['novelty_auroc']:.3f}  "
            f"known_genus={m['dev_known_nn_genus_accuracy']:.3f}  "
            f"novel_family={m['dev_novel_nn_family_accuracy']:.3f}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation_status": "existing_checkpoint_inference_correction_no_selection",
        "inference": inference_metadata(device, args.eval_batch_size),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split_tsv_sha256": sha256_file(args.input),
        "evaluator_sha256": sha256_file(__file__),
        "checkpoint": str(args.checkpoint),
        "experiment": ckpt.get("experiment", "unknown"),
        "protected_splits": ["CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"],
        "input_split_counts": dict(split_counts),
        "retrieval_pairs": metrics,
    }
    with open(outdir / "dev_retrieval_matrix.json", "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    if args.write_predictions:
        fields = [
            "query_view", "reference_view", "split", "query_id",
            "query_family", "query_genus", "query_species", "max_cosine",
            "nn_id", "nn_family", "nn_genus", "nn_species",
        ]
        with open(outdir / "dev_retrieval_predictions.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
            w.writeheader(); w.writerows(preds)
    print(f"\nwrote {outdir / 'dev_retrieval_matrix.json'}")


if __name__ == "__main__":
    main()
