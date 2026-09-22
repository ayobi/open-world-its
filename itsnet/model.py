"""Base-wise encoder producing CRF emissions.

Architecture: SpliceAI-style stack of dilated residual convolutions, optional
BiLSTM, 1x1 head. The dilation schedule is chosen so the receptive field
comfortably exceeds a full rDNA operon segment -- 5.8S is ~158 bp and the ITS
spacers are 150-400 bp, so a receptive field of ~2 kb means every base "sees"
both flanking conserved genes. That is the property HMMER gets from its profile
alignment and that a naive small-kernel CNN would lack.

The BiLSTM is OFF by default and exists as an ablation. With a ~2 kb receptive
field the convolutions already span the whole operon, and the CRF supplies the
global ordering constraint, so the LSTM has little left to contribute -- while
costing sequential (non-parallelisable) compute that would undermine the
CPU-throughput target and complicate the Rust/ONNX deployment path.

GroupNorm is used rather than BatchNorm so that train and eval behaviour are
identical and no running statistics are polluted by padding. Batch by length
bucket to keep padding small regardless.
"""
from __future__ import annotations

import torch
import torch.nn as nn

BASES = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}


def encode_sequence(seq: str) -> torch.Tensor:
    """One-hot encode to [4, L]. Ambiguity codes and N become all-zero columns."""
    seq = seq.upper()
    x = torch.zeros(4, len(seq), dtype=torch.float32)
    for i, ch in enumerate(seq):
        j = BASE_IDX.get(ch)
        if j is not None:
            x[j, i] = 1.0
    return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.drop(h)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ItsEncoder(nn.Module):
    """Dilated CNN (+ optional BiLSTM) -> per-base emission logits.

    forward(x: [B, C_in, L]) -> emissions [B, L, n_states]
    """

    def __init__(
        self,
        n_states: int = 7,
        in_channels: int = 4,
        channels: int = 64,
        kernel: int = 5,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        repeats: int = 2,
        downsample: int = 1,
        use_lstm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = dict(
            n_states=n_states, in_channels=in_channels, channels=channels, kernel=kernel,
            dilations=tuple(dilations), repeats=repeats, downsample=downsample,
            use_lstm=use_lstm,
        )
        self.downsample = downsample
        self.stem = nn.Conv1d(in_channels, channels, 9, padding=4)

        if downsample > 1:
            # Exact-tiling stride so output length is L // downsample with no
            # padding arithmetic. Callers must pad L to a multiple of
            # `downsample` (see pad_to_multiple) -- keeping this out of the
            # graph is what lets the ONNX export stay fully dynamic in length.
            self.down = nn.Conv1d(channels, channels, downsample, stride=downsample)
            self.up = nn.ConvTranspose1d(channels, channels, downsample, stride=downsample)
            self.fuse = nn.Conv1d(2 * channels, channels, 5, padding=2)
            self.refine = ResidualBlock(channels, kernel, 1, dropout=dropout)

        blocks = []
        for _ in range(repeats):
            for d in dilations:
                blocks.append(ResidualBlock(channels, kernel, d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(8, channels)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)
        self.head = nn.Conv1d(channels, n_states, 1)

    @property
    def receptive_field(self) -> int:
        """Number of input bases influencing one output base.

        With downsampling, the dilated stack operates on a coarser grid, so its
        receptive field is multiplied by the downsampling factor -- which is
        precisely why downsampling buys throughput without costing context.
        """
        rf = 0
        for _ in range(self.cfg["repeats"]):
            for d in self.cfg["dilations"]:
                rf += 2 * d * (self.cfg["kernel"] - 1)
        rf *= self.downsample
        rf += 9  # stem
        if self.downsample > 1:
            rf += 2 * self.downsample + 4 + 2 * (self.cfg["kernel"] - 1)
        return rf

    def forward(self, x):
        f0 = self.stem(x)
        if self.downsample > 1:
            h = self.down(f0)
            h = self.blocks(h)
            h = self.up(h)
            h = self.fuse(torch.cat([h, f0], dim=1))
            h = self.refine(h)
        else:
            h = self.blocks(f0)
        h = self.norm(h)
        if self.use_lstm:
            h, _ = self.lstm(h.transpose(1, 2))
            h = h.transpose(1, 2)
        return self.head(h).transpose(1, 2)  # [B, L, n_states]


class ItsFlankDetector(nn.Module):
    """Encoder + CRF. Training front-end only; inference exports the encoder."""

    def __init__(self, crf, **kwargs):
        super().__init__()
        self.encoder = ItsEncoder(**kwargs)
        self.crf = crf

    def forward(self, x, tags, mask=None):
        return self.crf(self.encoder(x), tags, mask)

    @torch.no_grad()
    def decode(self, x, mask=None):
        return self.crf.viterbi(self.encoder(x), mask)


def pad_to_multiple(x: torch.Tensor, multiple: int):
    """Right-pad a [B, C, L] batch so L is divisible by `multiple`.

    Returns (padded, original_length). Trim emissions back with
    `emissions[:, :original_length]`. Kept outside the module so the exported
    ONNX graph has no dynamic padding op.
    """
    if multiple <= 1:
        return x, x.shape[-1]
    L = x.shape[-1]
    pad = (-L) % multiple
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, L
