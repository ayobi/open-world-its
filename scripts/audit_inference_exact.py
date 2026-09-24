#!/usr/bin/env python3
"""Check padding and batch invariance with frozen weights; no paper metrics.

Default sequences are synthetic diagnostic inputs, not biological test data.
Optional FASTAs allow the same checks on small user-supplied subsets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from itsnet.openworld.benchmark import encode_batch
from itsnet.openworld.inference_exact import (
    configure_inference, embed_sequences_exact, embed_rows_exact,
    embed_fasta_rows_exact, inference_metadata, sha256_file,
)
from itsnet.openworld.model import BarcodeEncoder
from itsnet.openworld.m0 import nearest_reference
from score_historical_cosine_exact import read_fasta


@torch.inference_mode()
def legacy_embed(model, seqs, device, batch_size):
    result = []
    for start in range(0, len(seqs), batch_size):
        x, mask = encode_batch(seqs[start:start + batch_size], device)
        result.append(model(x, mask).cpu())
    return torch.cat(result)


def differences(a, b):
    d = (a - b).abs()
    return {"mean_abs": float(d.mean()), "max_abs": float(d.max())}


def synthetic_sequences(seed):
    rng = random.Random(seed)
    return ["".join(rng.choices("ACGT", k=n))
            for n in (64, 128, 256, 512, 64, 128, 256, 512, 33, 96, 192, 384)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--queries")
    ap.add_argument("--refs")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if bool(args.queries) != bool(args.refs):
        ap.error("provide both --queries and --refs, or neither")
    if args.limit < 2 or args.threads < 1:
        ap.error("limit must be >=2 and threads must be positive")
    out = Path(args.out)
    if out.exists():
        ap.error("output already exists; choose a new path")
    checkpoint_sha = sha256_file(args.checkpoint)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        if checkpoint_sha != manifest["checkpoint_sha256"]:
            raise SystemExit("checkpoint hash does not match manifest")
    torch.set_num_threads(args.threads)
    configure_inference()
    device = torch.device(args.device)
    c = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BarcodeEncoder(channels=int(c["args"].get("channels", 64)),
                           embedding_dim=int(c["args"].get("embedding_dim", 256)))
    model.load_state_dict(c["model_state"], strict=True)
    model.to(device).eval()
    if args.queries:
        qrows = read_fasta(Path(args.queries))[:args.limit]
        rrows = read_fasta(Path(args.refs))[:args.limit]
    else:
        qrows = [(f"synthetic_query_{i}", s) for i, s in enumerate(synthetic_sequences(17))]
        rrows = [(f"synthetic_ref_{i}", s) for i, s in enumerate(synthetic_sequences(29))]
    if not qrows or not rrows:
        raise SystemExit("diagnostic inputs must be nonempty")
    qs, rs = [s for _, s in qrows], [s for _, s in rrows]
    report = {
        "scope": "small inference diagnostic; not a full biological evaluation",
        "input_kind": "FASTA subsets" if args.queries else "synthetic DNA",
        "checkpoint_sha256": checkpoint_sha,
        "inference": inference_metadata(device, 64),
        "n_queries": len(qs), "n_references": len(rs),
        "groupnorm_layers": sum(isinstance(m, torch.nn.GroupNorm) for m in model.modules()),
    }
    q1 = legacy_embed(model, qs, device, 1)
    r1 = legacy_embed(model, rs, device, 1)
    qp = legacy_embed(model, qs, device, 64)
    rp = legacy_embed(model, rs, device, 64)
    score = lambda q, r: nearest_reference(q.to(device), r.to(device), 7)
    s1, i1 = score(q1, r1)
    sp, ip = score(qp, rp)
    report["legacy_padded_vs_singleton"] = {
        "query_embeddings": differences(q1, qp),
        "reference_embeddings": differences(r1, rp),
        "max_cosine": differences(s1, sp),
        "nearest_reference_changes": int((i1 != ip).sum()),
        "query_only_score_change": differences(s1, score(qp, r1)[0]),
        "reference_only_score_change": differences(s1, score(q1, rp)[0]),
    }
    exact_checks = {}
    tolerance = 1e-5
    for bs in (1, 2, 64, 256):
        qe = embed_sequences_exact(model, qs, device, bs)
        re = embed_sequences_exact(model, rs, device, bs)
        se, ie = score(qe, re)
        exact_checks[str(bs)] = {
            "query_embeddings": differences(q1, qe),
            "reference_embeddings": differences(r1, re),
            "max_cosine": differences(s1, se),
            "nearest_reference_changes": int((i1 != ie).sum()),
        }
    report["exact_vs_singleton"] = exact_checks
    order = list(reversed(range(len(qs))))
    reversed_z = embed_sequences_exact(model, [qs[i] for i in order], device, 64)
    restored = torch.empty_like(reversed_z)
    restored[order] = reversed_z
    report["query_order_check"] = differences(q1, restored)
    extra = "ACGT" * (max(map(len, qs)) // 4 + 37)
    appended = embed_sequences_exact(model, qs + [extra], device, 64)[:-1]
    report["long_companion_check"] = differences(q1, appended)
    mapping_rows = [{"id": qid, "core": seq} for qid, seq in qrows]
    _, tsv_z = embed_rows_exact(model, mapping_rows, "core", device, 256)
    fasta_z = embed_fasta_rows_exact(model, qrows, device, 64)
    report["tsv_vs_fasta_adapter"] = differences(tsv_z, fasta_z)

    # Isolate GroupNorm: hold valid stem activations fixed, append zero feature
    # columns, and compare ONLY the original positions after normalization.
    with torch.inference_mode():
        x, _ = encode_batch([qs[0]], device)
        h = model.stem(x)
        norm = model.blocks[0].norm1
        h_pad = F.pad(h, (0, h.shape[-1]))
        report["groupnorm_only_padding_probe"] = differences(norm(h), norm(h_pad)[..., :h.shape[-1]])
    required = [report[k]["max_abs"] for k in
                ("query_order_check", "long_companion_check", "tsv_vs_fasta_adapter")]
    for check in exact_checks.values():
        required.extend(check[key]["max_abs"] for key in
                        ("query_embeddings", "reference_embeddings", "max_cosine"))
    report["absolute_tolerance"] = tolerance
    report["passed"] = all(x <= tolerance for x in required)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Exact-length checks failed; investigate before full evaluation.")


if __name__ == "__main__":
    main()
