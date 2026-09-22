#!/usr/bin/env python3
"""Train and evaluate the M0 open-world ITS embedding baseline.

Scientific contract
-------------------
* Gradients: TRAIN_REF only.
* Training locus: ITS-core only.
* Objective: cosine-proxy genus classification only.
* Development evaluation: DEV_KNOWN vs DEV_NOVEL only.
* CAL_KNOWN / TEST_KNOWN / TEST_NOVEL are never embedded or scored here.
* Open-world score: max cosine to a TRAIN_REF sequence embedding.

This is intentionally a weak, interpretable baseline.  M1 will add explicit
locus/view invariance; M2 taxonomic-distance geometry; M3 pseudo-novel episodes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from itsnet.openworld.m0 import CosineClassifier, binary_auroc, nearest_reference
from itsnet.openworld.model import BarcodeEncoder


SAFE_SPLITS = {"TRAIN_REF", "DEV_KNOWN", "DEV_NOVEL"}
FORBIDDEN_EVAL_SPLITS = {"CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"}
VIEWS = ("core", "its1", "its2")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: str):
    keep = []
    seen_splits = Counter()
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"id", "family", "genus", "species", "split", *VIEWS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise SystemExit("split TSV missing columns: " + ", ".join(sorted(missing)))
        for r in reader:
            seen_splits[r["split"]] += 1
            if r["split"] in SAFE_SPLITS:
                keep.append(r)
    for name in FORBIDDEN_EVAL_SPLITS:
        if seen_splits[name] == 0:
            raise SystemExit(f"expected protected split {name!r} in input; refusing ambiguous dataset")
    return keep, seen_splits


class CoreDataset(Dataset):
    def __init__(self, rows, genus_to_idx):
        self.rows = [r for r in rows if r["split"] == "TRAIN_REF" and r["core"]]
        self.genus_to_idx = genus_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r["core"], self.genus_to_idx[r["genus"]]


# ASCII -> 0(ambiguous/pad), 1=A, 2=C, 3=G, 4=T.
_ASCII_LUT = np.zeros(256, dtype=np.uint8)
for _b, _i in ((ord("A"), 1), (ord("C"), 2), (ord("G"), 3), (ord("T"), 4),
               (ord("a"), 1), (ord("c"), 2), (ord("g"), 3), (ord("t"), 4)):
    _ASCII_LUT[_b] = _i


def encode_batch(seqs: Iterable[str], device: torch.device):
    """Fast padded one-hot encoding directly to the target device."""
    seqs = list(seqs)
    lengths = [len(s) for s in seqs]
    L = max(lengths)
    codes = np.zeros((len(seqs), L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        raw = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
        codes[i, : len(raw)] = _ASCII_LUT[raw]
    c = torch.from_numpy(codes).to(device=device, non_blocking=True)
    # [B,L,5] -> drop ambiguity/pad channel -> [B,4,L]
    x = F.one_hot(c.long(), num_classes=5)[..., 1:].permute(0, 2, 1).float()
    ar = torch.arange(L, device=device)[None, :]
    mask = ar < torch.tensor(lengths, device=device)[:, None]
    return x, mask


def collate_core(batch):
    seqs, labels = zip(*batch)
    return list(seqs), torch.tensor(labels, dtype=torch.long)


def make_sampler(ds: CoreDataset):
    counts = Counter(ds.rows[i]["genus"] for i in range(len(ds)))
    # Equal total sampling mass per genus. This prevents abundant UNITE genera
    # from dominating the representation while retaining every TRAIN_REF record.
    weights = [1.0 / counts[r["genus"]] for r in ds.rows]
    return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)


@torch.no_grad()
def embed_rows(model, rows, view, device, batch_size):
    selected = [r for r in rows if r.get(view, "")]
    zs = []
    model.eval()
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        x, mask = encode_batch([r[view] for r in batch], device)
        z = model(x, mask)
        zs.append(z.cpu())
    z = torch.cat(zs, dim=0) if zs else torch.empty(0, model.proj[-1].out_features)
    return selected, z


def mean_or_nan(xs):
    return float(np.mean(xs)) if xs else float("nan")


def evaluate_view(model, rows, view, device, batch_size, nn_chunk):
    ref_rows = [r for r in rows if r["split"] == "TRAIN_REF"]
    known_rows = [r for r in rows if r["split"] == "DEV_KNOWN"]
    novel_rows = [r for r in rows if r["split"] == "DEV_NOVEL"]

    ref_rows, ref_z = embed_rows(model, ref_rows, view, device, batch_size)
    known_rows, known_z = embed_rows(model, known_rows, view, device, batch_size)
    novel_rows, novel_z = embed_rows(model, novel_rows, view, device, batch_size)
    if not len(ref_rows) or not len(known_rows) or not len(novel_rows):
        raise RuntimeError(f"{view}: empty reference/known/novel view after filtering")

    # Move the compact embedding library to GPU only for nearest-neighbour search.
    ref_dev = ref_z.to(device)
    k_score, k_idx = nearest_reference(known_z.to(device), ref_dev, nn_chunk)
    n_score, n_idx = nearest_reference(novel_z.to(device), ref_dev, nn_chunk)

    known_genus_correct = [
        known_rows[i]["genus"] == ref_rows[int(k_idx[i])]["genus"]
        for i in range(len(known_rows))
    ]
    novel_family_correct = [
        novel_rows[i]["family"] == ref_rows[int(n_idx[i])]["family"]
        for i in range(len(novel_rows))
    ]
    novel_order_correct = [
        novel_rows[i].get("order", "") == ref_rows[int(n_idx[i])].get("order", "")
        and bool(novel_rows[i].get("order", ""))
        for i in range(len(novel_rows))
    ]
    novel_class_correct = [
        novel_rows[i].get("class", "") == ref_rows[int(n_idx[i])].get("class", "")
        and bool(novel_rows[i].get("class", ""))
        for i in range(len(novel_rows))
    ]

    # Novel is the positive class; lower cosine should mean more novel.
    labels = [0] * len(k_score) + [1] * len(n_score)
    novelty_scores = [-float(x) for x in k_score] + [-float(x) for x in n_score]
    auc = binary_auroc(labels, novelty_scores)

    metrics = {
        "view": view,
        "n_reference": len(ref_rows),
        "n_dev_known": len(known_rows),
        "n_dev_novel": len(novel_rows),
        "known_max_cosine_mean": float(k_score.mean()),
        "novel_max_cosine_mean": float(n_score.mean()),
        "novelty_auroc": float(auc),
        "dev_known_nn_genus_accuracy": mean_or_nan(known_genus_correct),
        "dev_novel_nn_family_accuracy": mean_or_nan(novel_family_correct),
        "dev_novel_nn_order_accuracy": mean_or_nan(novel_order_correct),
        "dev_novel_nn_class_accuracy": mean_or_nan(novel_class_correct),
    }

    preds = []
    for split_name, qs, scores, idxs in (
        ("DEV_KNOWN", known_rows, k_score, k_idx),
        ("DEV_NOVEL", novel_rows, n_score, n_idx),
    ):
        for i, q in enumerate(qs):
            ref = ref_rows[int(idxs[i])]
            preds.append({
                "view": view,
                "split": split_name,
                "query_id": q["id"],
                "query_family": q["family"],
                "query_genus": q["genus"],
                "query_species": q["species"],
                "max_cosine": f"{float(scores[i]):.8f}",
                "nn_id": ref["id"],
                "nn_family": ref["family"],
                "nn_genus": ref["genus"],
                "nn_species": ref["species"],
            })
    return metrics, preds


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="six-way split TSV")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--nn-chunk", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    rows, split_counts = load_rows(args.input)
    train_rows = [r for r in rows if r["split"] == "TRAIN_REF" and r["core"]]
    genera = sorted({r["genus"] for r in train_rows})
    genus_to_idx = {g: i for i, g in enumerate(genera)}
    ds = CoreDataset(rows, genus_to_idx)
    sampler = make_sampler(ds)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_core,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = BarcodeEncoder(
        channels=args.channels,
        embedding_dim=args.embedding_dim,
        dropout=0.1,
    ).to(device)
    head = CosineClassifier(len(genera), args.embedding_dim, args.temperature).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"device        : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"TRAIN_REF core: {len(ds):,}")
    print(f"train genera  : {len(genera):,}")
    print(f"parameters    : {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()):,}")
    print("protected     : CAL_KNOWN / TEST_KNOWN / TEST_NOVEL not evaluated")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); head.train()
        total_loss = 0.0
        total_n = 0
        correct = 0
        for seqs, y in loader:
            y = y.to(device, non_blocking=True)
            x, mask = encode_batch(seqs, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                z = model(x, mask)
                logits = head(z)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            n = y.numel()
            total_loss += float(loss.detach()) * n
            total_n += n
            correct += int((logits.detach().argmax(dim=1) == y).sum())
        sched.step()
        rec = {
            "epoch": epoch,
            "train_loss": total_loss / total_n,
            "train_proxy_accuracy": correct / total_n,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(rec)
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"loss={rec['train_loss']:.4f}  proxy_acc={rec['train_proxy_accuracy']:.3f}  "
            f"lr={rec['lr']:.2e}"
        )

    ckpt = {
        "experiment": "M0_core_cosine_genus",
        "model_state": model.state_dict(),
        "head_state": head.state_dict(),
        "genera": genera,
        "args": vars(args),
        "history": history,
    }
    torch.save(ckpt, outdir / "m0_checkpoint.pt")

    all_metrics = []
    all_preds = []
    print("\nDEV evaluation (TEST/CAL remain untouched)")
    for view in VIEWS:
        metrics, preds = evaluate_view(
            model, rows, view, device, args.eval_batch_size, args.nn_chunk
        )
        all_metrics.append(metrics)
        all_preds.extend(preds)
        print(
            f"{view:5s}  AUROC={metrics['novelty_auroc']:.3f}  "
            f"known_cos={metrics['known_max_cosine_mean']:.3f}  "
            f"novel_cos={metrics['novel_max_cosine_mean']:.3f}  "
            f"known_genus={metrics['dev_known_nn_genus_accuracy']:.3f}  "
            f"novel_family={metrics['dev_novel_nn_family_accuracy']:.3f}"
        )

    report = {
        "experiment": "M0_core_cosine_genus",
        "scientific_firewall": {
            "gradient_split": "TRAIN_REF",
            "model_selection_splits": ["DEV_KNOWN", "DEV_NOVEL"],
            "untouched_splits": ["CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"],
            "training_view": "core",
            "novelty_score": "negative max cosine to TRAIN_REF for AUROC (novel positive)",
        },
        "input_split_counts": dict(split_counts),
        "train_core_records": len(ds),
        "train_genera": len(genera),
        "history": history,
        "views": all_metrics,
    }
    with open(outdir / "m0_metrics.json", "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    pred_fields = [
        "view", "split", "query_id", "query_family", "query_genus", "query_species",
        "max_cosine", "nn_id", "nn_family", "nn_genus", "nn_species",
    ]
    with open(outdir / "m0_dev_predictions.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pred_fields, delimiter="\t")
        w.writeheader(); w.writerows(all_preds)

    print(f"\nwrote {outdir / 'm0_checkpoint.pt'}")
    print(f"wrote {outdir / 'm0_metrics.json'}")
    print(f"wrote {outdir / 'm0_dev_predictions.tsv'}")


if __name__ == "__main__":
    main()
