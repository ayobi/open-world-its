"""Shared DEV-only utilities for open-world embedding experiments."""
from __future__ import annotations

import csv
import random
from collections import Counter
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from itsnet.openworld.m0 import binary_auroc, nearest_reference

SAFE_SPLITS = {"TRAIN_REF", "DEV_KNOWN", "DEV_NOVEL"}
FORBIDDEN_EVAL_SPLITS = {"CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"}
VIEWS = ("core", "its1", "its2")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dev_rows(path: str):
    """Load only TRAIN/DEV rows while verifying protected splits exist."""
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
            raise SystemExit(
                f"expected protected split {name!r} in input; refusing ambiguous dataset"
            )
    return keep, seen_splits


# ASCII -> 0(ambiguous/pad), 1=A, 2=C, 3=G, 4=T.
_ASCII_LUT = np.zeros(256, dtype=np.uint8)
for _b, _i in (
    (ord("A"), 1), (ord("C"), 2), (ord("G"), 3), (ord("T"), 4),
    (ord("a"), 1), (ord("c"), 2), (ord("g"), 3), (ord("t"), 4),
):
    _ASCII_LUT[_b] = _i


def encode_batch(seqs: Iterable[str], device: torch.device):
    """Fast padded one-hot encoding directly to the target device."""
    seqs = list(seqs)
    if not seqs:
        return (
            torch.empty(0, 4, 0, device=device),
            torch.empty(0, 0, dtype=torch.bool, device=device),
        )
    lengths = [len(s) for s in seqs]
    L = max(lengths)
    codes = np.zeros((len(seqs), L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        raw = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
        codes[i, : len(raw)] = _ASCII_LUT[raw]
    c = torch.from_numpy(codes).to(device=device, non_blocking=True)
    x = F.one_hot(c.long(), num_classes=5)[..., 1:].permute(0, 2, 1).float()
    ar = torch.arange(L, device=device)[None, :]
    mask = ar < torch.tensor(lengths, device=device)[:, None]
    return x, mask


@torch.no_grad()
def embed_rows(model, rows, view: str, device: torch.device, batch_size: int):
    selected = [r for r in rows if r.get(view, "")]
    zs = []
    model.eval()
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        x, mask = encode_batch([r[view] for r in batch], device)
        z = model(x, mask)
        zs.append(z.cpu())
    dim = model.proj[-1].out_features
    z = torch.cat(zs, dim=0) if zs else torch.empty(0, dim)
    return selected, z


def _mean_bool(xs):
    return float(np.mean(xs)) if xs else float("nan")


def evaluate_embedding_pair(
    ref_rows,
    ref_z: torch.Tensor,
    known_rows,
    known_z: torch.Tensor,
    novel_rows,
    novel_z: torch.Tensor,
    *,
    query_view: str,
    reference_view: str,
    device: torch.device,
    nn_chunk: int,
):
    """Evaluate one query-view -> reference-view retrieval problem on DEV only."""
    if not len(ref_rows) or not len(known_rows) or not len(novel_rows):
        raise RuntimeError(
            f"{query_view}->{reference_view}: empty reference/known/novel embeddings"
        )
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

    labels = [0] * len(k_score) + [1] * len(n_score)
    novelty_scores = [-float(x) for x in k_score] + [-float(x) for x in n_score]
    auc = binary_auroc(labels, novelty_scores)

    metrics = {
        "query_view": query_view,
        "reference_view": reference_view,
        "n_reference": len(ref_rows),
        "n_dev_known": len(known_rows),
        "n_dev_novel": len(novel_rows),
        "known_max_cosine_mean": float(k_score.mean()),
        "novel_max_cosine_mean": float(n_score.mean()),
        "novelty_auroc": float(auc),
        "dev_known_nn_genus_accuracy": _mean_bool(known_genus_correct),
        "dev_novel_nn_family_accuracy": _mean_bool(novel_family_correct),
        "dev_novel_nn_order_accuracy": _mean_bool(novel_order_correct),
        "dev_novel_nn_class_accuracy": _mean_bool(novel_class_correct),
    }

    preds = []
    for split_name, qs, scores, idxs in (
        ("DEV_KNOWN", known_rows, k_score, k_idx),
        ("DEV_NOVEL", novel_rows, n_score, n_idx),
    ):
        for i, q in enumerate(qs):
            ref = ref_rows[int(idxs[i])]
            preds.append({
                "query_view": query_view,
                "reference_view": reference_view,
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
    del ref_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, preds


def embed_dev_matrix(model, rows, device: torch.device, batch_size: int):
    """Embed TRAIN_REF, DEV_KNOWN and DEV_NOVEL once for every available view."""
    out = {}
    for view in VIEWS:
        out[view] = {}
        for split in ("TRAIN_REF", "DEV_KNOWN", "DEV_NOVEL"):
            subset = [r for r in rows if r["split"] == split]
            rr, zz = embed_rows(model, subset, view, device, batch_size)
            out[view][split] = (rr, zz)
    return out
