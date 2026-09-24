#!/usr/bin/env python3
"""Score a historical FASTA query set by max cosine to a FASTA reference set.

Output is the same simple 3-column contract accepted by its-novelty's
score_recovery.py: query_id, target_id, score (higher = closer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from itsnet.openworld.inference_exact import configure_inference, inference_metadata, sha256_file

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


from itsnet.openworld.inference_exact import embed_fasta_rows_exact as embed


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
    if args.batch_size < 1 or args.nn_chunk < 1:
        ap.error("batch size and nn-chunk must be positive")
    configure_inference()

    out = Path(args.out)
    metadata_path = out.with_name(out.name + ".inference.json")
    if out.exists() or metadata_path.exists():
        raise SystemExit("Output already exists; choose a new path to preserve previous scores.")
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
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
    metadata = inference_metadata(device, args.batch_size)
    metadata.update({
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "refs_sha256": sha256_file(args.refs),
        "queries_sha256": sha256_file(args.queries),
        "scorer_sha256": sha256_file(__file__),
        "experiment": ckpt.get("experiment", "unknown"),
        "n_references": len(refs), "n_queries": len(queries),
        "nn_chunk": args.nn_chunk,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote      : {out}")


if __name__ == "__main__":
    main()
