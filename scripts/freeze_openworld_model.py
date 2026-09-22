#!/usr/bin/env python3
"""Freeze the selected open-world checkpoint and dataset contract.

This is an administrative lock, not another training step.  It copies the selected
checkpoint into a dedicated directory and records SHA-256 hashes of both the model and
split TSV.  Final evaluation verifies those hashes before touching CAL/TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="six-way split TSV used for development")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    src_ckpt = Path(args.checkpoint).resolve()
    split_tsv = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    if not src_ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {src_ckpt}")
    if not split_tsv.is_file():
        raise SystemExit(f"split TSV not found: {split_tsv}")

    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    experiment = str(ckpt.get("experiment", ""))
    if not experiment.startswith("M4_"):
        raise SystemExit(
            f"refusing to freeze non-M4 checkpoint: experiment={experiment!r}"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    dst_ckpt = outdir / "frozen_m4_checkpoint.pt"
    manifest_path = outdir / "freeze_manifest.json"
    if dst_ckpt.exists() or manifest_path.exists():
        raise SystemExit(
            f"freeze destination already exists: {outdir}\n"
            "Use a new directory rather than overwriting a frozen model."
        )

    source_hash = sha256_file(src_ckpt)
    dataset_hash = sha256_file(split_tsv)
    shutil.copy2(src_ckpt, dst_ckpt)
    frozen_hash = sha256_file(dst_ckpt)
    if frozen_hash != source_hash:
        raise RuntimeError("checkpoint hash changed during freeze copy")

    manifest = {
        "status": "FROZEN",
        "experiment": experiment,
        "checkpoint_file": dst_ckpt.name,
        "checkpoint_sha256": frozen_hash,
        "source_checkpoint": str(src_ckpt),
        "split_tsv": str(split_tsv),
        "split_tsv_sha256": dataset_hash,
        "training_args": ckpt.get("args", {}),
        "selection_rule": {
            "development_splits": ["DEV_KNOWN", "DEV_NOVEL"],
            "selected_model": "M4",
            "post_freeze_splits": ["CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"],
            "primary_novelty_score": "negative max cosine to same-view TRAIN_REF",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("FROZEN")
    print(f"experiment      : {experiment}")
    print(f"checkpoint      : {dst_ckpt}")
    print(f"checkpoint sha  : {frozen_hash}")
    print(f"split TSV sha   : {dataset_hash}")
    print(f"manifest        : {manifest_path}")
    print("next            : CAL_KNOWN / TEST_KNOWN / TEST_NOVEL may now be evaluated")


if __name__ == "__main__":
    main()
