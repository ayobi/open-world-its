"""Inference for existing BarcodeEncoder weights without variable-length padding.

Group by EXACT sequence length, then restore the incoming row order. This is
mathematically equivalent to encoding each sequence alone in eval mode, subject
to ordinary floating-point kernel differences. It does not modify training or
make BarcodeEncoder.forward itself insensitive to padding.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import platform

import torch

from itsnet.openworld.benchmark import VIEWS, encode_batch

POLICY = "exact_length_unpadded_fp32_v1"


def configure_inference():
    """Use full FP32 arithmetic; do not change the checkpoint or random seed."""
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inference_metadata(device, batch_size):
    import itsnet.model as residual_source
    import itsnet.openworld.model as model_source
    import itsnet.openworld.benchmark as encoding_source
    import itsnet.openworld.m0 as retrieval_source
    device = torch.device(device)
    return {
        "policy": POLICY,
        "batch_size_upper_bound": batch_size,
        "batching": "exact sequence length; original row order restored",
        "dtype": "float32; autocast and TF32 disabled",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "source_sha256": {
            "inference_exact.py": sha256_file(__file__),
            "itsnet/model.py": sha256_file(residual_source.__file__),
            "itsnet/openworld/model.py": sha256_file(model_source.__file__),
            "itsnet/openworld/benchmark.py": sha256_file(encoding_source.__file__),
            "itsnet/openworld/m0.py": sha256_file(retrieval_source.__file__),
        },
    }


@torch.inference_mode()
def embed_sequences_exact(model, sequences, device, batch_size):
    """Return CPU FP32 embeddings in input order; never pad unequal lengths.

    Empty collections are supported. Empty sequences and non-ASCII input are
    rejected rather than silently changing an input. ASCII ambiguity codes keep
    the original all-zero nucleotide representation and count as real positions.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    seqs = list(sequences)
    dim = model.proj[-1].out_features
    groups = defaultdict(list)
    for i, seq in enumerate(seqs):
        if not isinstance(seq, str) or not seq:
            raise ValueError(f"sequence {i} must be a nonempty string")
        if not seq.isascii() or any(c.isspace() for c in seq):
            raise ValueError(f"sequence {i} contains non-ASCII or whitespace characters")
        groups[len(seq)].append(i)
    result = torch.empty((len(seqs), dim), dtype=torch.float32)
    if not seqs:
        return result
    model.eval()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("exact inference requires FP32 model parameters")
    with torch.autocast(device_type=device.type, enabled=False):
        for length in sorted(groups):
            indices = groups[length]
            for start in range(0, len(indices), batch_size):
                idx = indices[start:start + batch_size]
                x, mask = encode_batch([seqs[i] for i in idx], device)
                z = model(x, mask).float().cpu()
                if not torch.isfinite(z).all():
                    raise RuntimeError(f"nonfinite embedding in sequence length {length}")
                result[idx] = z
    return result


def embed_fasta_rows_exact(model, rows, device, batch_size):
    return embed_sequences_exact(model, (seq for _, seq in rows), device, batch_size)


def embed_rows_exact(model, rows, view, device, batch_size):
    # Match the original evaluator's view selection exactly.
    selected = [r for r in rows if r.get(view, "")]
    return selected, embed_sequences_exact(model, (r[view] for r in selected), device, batch_size)


def embed_dev_matrix_exact(model, rows, device, batch_size):
    return {
        view: {
            split: embed_rows_exact(model, [r for r in rows if r["split"] == split],
                                    view, device, batch_size)
            for split in ("TRAIN_REF", "DEV_KNOWN", "DEV_NOVEL")
        }
        for view in VIEWS
    }
