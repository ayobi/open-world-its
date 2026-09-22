"""Compact locus-aware embedding CNN for fungal ITS sequences."""
from __future__ import annotations

from typing import Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from itsnet.model import ResidualBlock, encode_sequence


class BarcodeEncoder(nn.Module):
    """Sequence -> L2-normalized embedding.

    Deliberately small for v0. The novelty should come from the objective and
    open-world benchmark, not parameter count.
    """

    def __init__(self, channels: int = 64, embedding_dim: int = 256,
                 dilations=(1, 2, 4, 8, 16, 32), repeats: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Conv1d(4, channels, kernel_size=9, padding=4)
        blocks = []
        for _ in range(repeats):
            for d in dilations:
                blocks.append(ResidualBlock(channels, kernel=5, dilation=d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(8, channels)
        self.proj = nn.Sequential(
            nn.Linear(channels * 2, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, embedding_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B,4,L], mask [B,L] -> normalized z [B,D]."""
        h = self.norm(self.blocks(self.stem(x)))
        if mask is None:
            mean = h.mean(dim=-1)
            mx = h.amax(dim=-1)
        else:
            m = mask[:, None, :].to(h.dtype)
            denom = m.sum(dim=-1).clamp_min(1.0)
            mean = (h * m).sum(dim=-1) / denom
            neg_inf = torch.finfo(h.dtype).min
            mx = h.masked_fill(~mask[:, None, :], neg_inf).amax(dim=-1)
        return F.normalize(self.proj(torch.cat([mean, mx], dim=-1)), dim=-1)


def encode_sequences(seqs: Iterable[str]):
    """Pad variable-length strings into (x, mask)."""
    seqs = list(seqs)
    if not seqs:
        return torch.empty(0, 4, 0), torch.empty(0, 0, dtype=torch.bool)
    L = max(map(len, seqs))
    x = torch.zeros(len(seqs), 4, L, dtype=torch.float32)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        e = encode_sequence(seq)
        x[i, :, :e.shape[-1]] = e
        mask[i, :e.shape[-1]] = True
    return x, mask
