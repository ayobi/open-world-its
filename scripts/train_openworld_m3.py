#!/usr/bin/env python3
"""Train M3: M1 view invariance + pseudo-novel leave-one-genus episodes.

Scientific firewall
-------------------
* Gradients: TRAIN_REF only.
* Proxy supervision: ITS-core only (identical to M0/M1).
* View term: core <-> randomly sampled ITS1/ITS2 (identical to M1).
* New M3 term: each eligible core query is treated as pseudo-novel at genus rank;
  it must retrieve its family from learned genus proxies after its own genus proxy
  has been explicitly removed from that family's prototype.
* Development evaluation: DEV_KNOWN vs DEV_NOVEL only.
* CAL_KNOWN / TEST_KNOWN / TEST_NOVEL are never embedded or scored here.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from itsnet.openworld.benchmark import (
    VIEWS,
    embed_dev_matrix,
    encode_batch,
    evaluate_embedding_pair,
    load_dev_rows,
    seed_everything,
)
from itsnet.openworld.losses import (
    leave_one_genus_family_proxy_loss,
    view_invariance_loss,
)
from itsnet.openworld.m0 import CosineClassifier
from itsnet.openworld.model import BarcodeEncoder


class PairedViewDataset(Dataset):
    """TRAIN_REF core sequence plus one available alternate ITS locus."""

    def __init__(self, rows, genus_to_idx):
        self.rows = []
        for r in rows:
            if r["split"] != "TRAIN_REF" or not r["core"]:
                continue
            alts = [v for v in ("its1", "its2") if r.get(v, "")]
            if alts:
                self.rows.append((r, alts))
        self.genus_to_idx = genus_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, alts = self.rows[i]
        v = random.choice(alts)
        return r["core"], r[v], r["id"], self.genus_to_idx[r["genus"]], v


def collate_pair(batch):
    cores, alts, ids, labels, alt_views = zip(*batch)
    return list(cores), list(alts), list(ids), torch.tensor(labels, dtype=torch.long), list(alt_views)


def make_sampler(ds: PairedViewDataset):
    counts = Counter(r["genus"] for r, _ in ds.rows)
    weights = [1.0 / counts[r["genus"]] for r, _ in ds.rows]
    return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)


def build_genus_family_index(train_rows, genera):
    """Return contiguous family IDs for every genus and enforce taxonomy consistency."""
    genus_family = {}
    for r in train_rows:
        g, f = r["genus"], r["family"]
        prev = genus_family.setdefault(g, f)
        if prev != f:
            raise ValueError(f"genus {g!r} occurs in multiple families: {prev!r}, {f!r}")
    missing = [g for g in genera if g not in genus_family]
    if missing:
        raise ValueError(f"missing family for {len(missing)} genera")
    families = sorted({genus_family[g] for g in genera})
    family_to_idx = {f: i for i, f in enumerate(families)}
    idx = torch.tensor([family_to_idx[genus_family[g]] for g in genera], dtype=torch.long)
    return families, idx


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=96, help="source-record pairs per batch")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07, help="genus proxy temperature")
    ap.add_argument("--view-temperature", type=float, default=0.10)
    ap.add_argument("--lambda-view", type=float, default=1.0)
    ap.add_argument("--episode-temperature", type=float, default=0.10)
    ap.add_argument("--lambda-episode", type=float, default=0.5)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--nn-chunk", type=int, default=384)
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

    rows, split_counts = load_dev_rows(args.input)
    train_core = [r for r in rows if r["split"] == "TRAIN_REF" and r["core"]]
    genera = sorted({r["genus"] for r in train_core})
    genus_to_idx = {g: i for i, g in enumerate(genera)}
    families, genus_family_index_cpu = build_genus_family_index(train_core, genera)
    genus_family_counts_cpu = torch.bincount(genus_family_index_cpu, minlength=len(families))
    eligible_genera = int((genus_family_counts_cpu[genus_family_index_cpu] >= 2).sum())

    ds = PairedViewDataset(rows, genus_to_idx)
    sampler = make_sampler(ds)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_pair,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = BarcodeEncoder(channels=args.channels, embedding_dim=args.embedding_dim, dropout=0.1).to(device)
    head = CosineClassifier(len(genera), args.embedding_dim, args.temperature).to(device)
    genus_family_index = genus_family_index_cpu.to(device)
    genus_family_counts = genus_family_counts_cpu.to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    alt_counts = Counter()
    for r, alts in ds.rows:
        for v in alts:
            alt_counts[v] += 1

    print(f"device          : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"TRAIN_REF pairs : {len(ds):,}")
    print(f"train genera    : {len(genera):,}")
    print(f"train families  : {len(families):,}")
    print(f"episode genera  : {eligible_genera:,} with >=2 represented genera in family")
    print(f"alternate views : ITS1={alt_counts['its1']:,} ITS2={alt_counts['its2']:,}")
    print(
        "objective       : proxy(core) + "
        f"{args.lambda_view:g} * L_view + {args.lambda_episode:g} * L_episode(leave-one-genus family)"
    )
    print(f"parameters      : {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()):,}")
    print("protected       : CAL_KNOWN / TEST_KNOWN / TEST_NOVEL not evaluated")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); head.train()
        total_loss = total_proxy = total_view = total_episode = total_pair_cos = 0.0
        total_n = correct = total_episode_n = 0
        for cores, alts, ids, y, alt_views in loader:
            y = y.to(device, non_blocking=True)
            x, mask = encode_batch(cores + alts, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                z = model(x, mask)
                b = len(cores)
                z_core, z_alt = z[:b], z[b:]
                logits = head(z_core)
                proxy_loss = F.cross_entropy(logits, y)
                view_loss = view_invariance_loss(z, ids + ids, temperature=args.view_temperature)
                episode_loss = leave_one_genus_family_proxy_loss(
                    z_core,
                    y,
                    head.weight,
                    genus_family_index,
                    temperature=args.episode_temperature,
                )
                loss = proxy_loss + args.lambda_view * view_loss + args.lambda_episode * episode_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            n = y.numel()
            episode_n = int((genus_family_counts[genus_family_index[y]] >= 2).sum())
            total_n += n
            total_episode_n += episode_n
            total_loss += float(loss.detach()) * n
            total_proxy += float(proxy_loss.detach()) * n
            total_view += float(view_loss.detach()) * n
            total_episode += float(episode_loss.detach()) * max(episode_n, 1)
            total_pair_cos += float((z_core.detach() * z_alt.detach()).sum(dim=1).sum())
            correct += int((logits.detach().argmax(dim=1) == y).sum())
        sched.step()
        rec = {
            "epoch": epoch,
            "train_loss": total_loss / total_n,
            "train_proxy_loss": total_proxy / total_n,
            "train_view_loss": total_view / total_n,
            "train_episode_loss": total_episode / max(total_episode_n, 1),
            "train_episode_queries": total_episode_n,
            "train_pair_cosine": total_pair_cos / total_n,
            "train_proxy_accuracy": correct / total_n,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(rec)
        print(
            f"epoch {epoch:02d}/{args.epochs}  loss={rec['train_loss']:.4f}  "
            f"proxy={rec['train_proxy_loss']:.4f}  view={rec['train_view_loss']:.4f}  "
            f"episode={rec['train_episode_loss']:.4f}  pair_cos={rec['train_pair_cosine']:.3f}  "
            f"proxy_acc={rec['train_proxy_accuracy']:.3f}  lr={rec['lr']:.2e}"
        )

    ckpt = {
        "experiment": "M3_M1_plus_leave_one_genus_family_episode",
        "model_state": model.state_dict(),
        "head_state": head.state_dict(),
        "genera": genera,
        "families": families,
        "genus_family_index": genus_family_index_cpu,
        "args": vars(args),
        "history": history,
    }
    torch.save(ckpt, outdir / "m3_checkpoint.pt")

    print("\nDEV retrieval matrix (query -> reference; TEST/CAL remain untouched)")
    emb = embed_dev_matrix(model, rows, device, args.eval_batch_size)
    all_metrics, all_preds = [], []
    for qv in VIEWS:
        for rv in VIEWS:
            ref_rows, ref_z = emb[rv]["TRAIN_REF"]
            known_rows, known_z = emb[qv]["DEV_KNOWN"]
            novel_rows, novel_z = emb[qv]["DEV_NOVEL"]
            metrics, preds = evaluate_embedding_pair(
                ref_rows, ref_z, known_rows, known_z, novel_rows, novel_z,
                query_view=qv, reference_view=rv, device=device, nn_chunk=args.nn_chunk,
            )
            all_metrics.append(metrics)
            all_preds.extend(preds)
            print(
                f"{qv:5s}->{rv:5s}  AUROC={metrics['novelty_auroc']:.3f}  "
                f"known_genus={metrics['dev_known_nn_genus_accuracy']:.3f}  "
                f"novel_family={metrics['dev_novel_nn_family_accuracy']:.3f}"
            )

    report = {
        "experiment": "M3_M1_plus_leave_one_genus_family_episode",
        "scientific_firewall": {
            "gradient_split": "TRAIN_REF",
            "proxy_training_view": "core",
            "view_invariance_pairs": "core<->random(ITS1,ITS2)",
            "episode_definition": "core query; family prototype from TRAIN_REF learned genus proxies with query genus proxy removed",
            "model_selection_splits": ["DEV_KNOWN", "DEV_NOVEL"],
            "untouched_splits": ["CAL_KNOWN", "TEST_KNOWN", "TEST_NOVEL"],
            "novelty_score": "negative max cosine to TRAIN_REF (novel positive)",
        },
        "input_split_counts": dict(split_counts),
        "train_paired_records": len(ds),
        "train_genera": len(genera),
        "train_families": len(families),
        "episode_eligible_genera": eligible_genera,
        "history": history,
        "retrieval_pairs": all_metrics,
    }
    with open(outdir / "m3_metrics.json", "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    pred_fields = [
        "query_view", "reference_view", "split", "query_id", "query_family",
        "query_genus", "query_species", "max_cosine", "nn_id", "nn_family",
        "nn_genus", "nn_species",
    ]
    with open(outdir / "m3_dev_predictions.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pred_fields, delimiter="\t")
        w.writeheader(); w.writerows(all_preds)

    print(f"\nwrote {outdir / 'm3_checkpoint.pt'}")
    print(f"wrote {outdir / 'm3_metrics.json'}")
    print(f"wrote {outdir / 'm3_dev_predictions.tsv'}")


if __name__ == "__main__":
    main()
