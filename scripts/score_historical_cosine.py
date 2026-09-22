#!/usr/bin/env python3
"""Score a historical FASTA query set by max cosine to a FASTA reference set.

Output is the same simple 3-column contract accepted by its-novelty's
score_recovery.py: query_id, target_id, score (higher = closer).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from itsnet.openworld.benchmark import encode_batch
from itsnet.openworld.m0 import nearest_reference
from itsnet.openworld.model import BarcodeEncoder


def read_fasta(path: Path):
    rows = []
    header = None
    seq = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    rows.append((header.split()[0], "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header is not None:
        rows.append((header.split()[0], "".join(seq)))
    return rows


@torch.no_grad()
def embed(model, rows, device, batch_size):
    zs = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        x, mask = encode_batch([s for _, s in batch], device)
        zs.append(model(x, mask).cpu())
    dim = model.proj[-1].out_features
    return torch.cat(zs, dim=0) if zs else torch.empty(0, dim)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--nn-chunk", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    carg = ckpt.get("args", {})
    model = BarcodeEncoder(
        channels=int(carg.get("channels", 64)),
        embedding_dim=int(carg.get("embedding_dim", 256)),
        dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    refs = read_fasta(Path(args.refs))
    queries = read_fasta(Path(args.queries))
    if not refs or not queries:
        raise SystemExit("refs and queries must both be non-empty FASTA files")

    print(f"device     : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"checkpoint : {ckpt.get('experiment', 'unknown')}")
    print(f"refs       : {len(refs):,}")
    print(f"queries    : {len(queries):,}")

    ref_z = embed(model, refs, device, args.batch_size)
    qry_z = embed(model, queries, device, args.batch_size)
    scores, idxs = nearest_reference(qry_z.to(device), ref_z.to(device), args.nn_chunk)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for (qid, _), score, ix in zip(queries, scores.tolist(), idxs.tolist()):
            fh.write(f"{qid}\t{refs[int(ix)][0]}\t{float(score):.8f}\n")
    print(f"wrote      : {out}")


if __name__ == "__main__":
    main()
