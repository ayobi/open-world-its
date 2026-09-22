"""Utilities for the M0 open-world embedding baseline.

M0 is deliberately simple: an ITS-core encoder is trained with a cosine
classifier over TRAIN_REF genera.  The classifier head is discarded at
evaluation; open-world scoring uses nearest-reference embedding cosine.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """Learned normalized class proxies with temperature-scaled cosine logits."""

    def __init__(self, n_classes: int, embedding_dim: int, temperature: float = 0.07):
        super().__init__()
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ F.normalize(self.weight, dim=1).T / self.temperature


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUROC with tie handling; labels use 1 for the positive class.

    Implemented here to keep the benchmark independent of scikit-learn.
    Returns NaN if either class is absent.
    """
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        # Average 1-based rank for ties.
        avg_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


@torch.no_grad()
def nearest_reference(
    query_z: torch.Tensor,
    ref_z: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return max cosine and reference index for each normalized query embedding.

    Embeddings are expected to be L2-normalized.  Chunking avoids materializing a
    full N_query x N_reference similarity matrix.
    """
    if query_z.ndim != 2 or ref_z.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if query_z.shape[1] != ref_z.shape[1]:
        raise ValueError("embedding dimensions differ")
    if ref_z.shape[0] == 0:
        raise ValueError("reference embedding set is empty")
    vals, inds = [], []
    for start in range(0, query_z.shape[0], chunk_size):
        q = query_z[start:start + chunk_size]
        sim = q @ ref_z.T
        v, ix = sim.max(dim=1)
        vals.append(v.cpu())
        inds.append(ix.cpu())
    return torch.cat(vals), torch.cat(inds)
