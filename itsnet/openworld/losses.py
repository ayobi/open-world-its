"""Losses for locus invariance, taxonomic geometry and pseudo-novel episodes."""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn.functional as F


RANKS = ("species", "genus", "family", "order", "class", "phylum")


def view_invariance_loss(z: torch.Tensor, source_ids: Sequence[str], temperature: float = 0.1):
    """Supervised contrastive loss: different views of the same record are positives."""
    if z.shape[0] < 2:
        return z.sum() * 0.0
    sim = z @ z.T / temperature
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -torch.inf)
    ids = list(source_ids)
    positive = torch.tensor(
        [[(i != j and ids[i] == ids[j]) for j in range(len(ids))] for i in range(len(ids))],
        dtype=torch.bool, device=z.device,
    )
    valid = positive.any(dim=1)
    if not valid.any():
        return z.sum() * 0.0
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per = -(logp.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1).clamp_min(1))
    return per[valid].mean()


def leave_one_genus_family_proxy_loss(
    z: torch.Tensor,
    genus_targets: torch.Tensor,
    genus_proxy_weight: torch.Tensor,
    genus_family_index: torch.Tensor,
    temperature: float = 0.1,
):
    """Pseudo-novel family classification with the query genus removed.

    Each row in ``z`` is treated as if its genus were absent from the reference.
    Family prototypes are built from the normalized learned genus proxies, but the
    query genus proxy is explicitly subtracted from its own family's prototype.
    Queries whose family has only one represented genus are skipped because a
    leave-one-genus-out family target would otherwise be undefined.

    This is an efficient TRAIN_REF-only approximation to an episodic open-world
    support set: it keeps the ordinary M1 minibatch sampler unchanged while giving
    every eligible query a valid higher-rank target.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if z.ndim != 2 or genus_proxy_weight.ndim != 2:
        raise ValueError("z and genus_proxy_weight must be rank-2")
    if z.shape[1] != genus_proxy_weight.shape[1]:
        raise ValueError("embedding dimensions differ")
    if genus_targets.ndim != 1 or genus_targets.shape[0] != z.shape[0]:
        raise ValueError("genus_targets must have one entry per query")
    if genus_family_index.ndim != 1 or genus_family_index.shape[0] != genus_proxy_weight.shape[0]:
        raise ValueError("genus_family_index must have one entry per genus proxy")
    if z.shape[0] == 0:
        return z.sum() * 0.0

    genus_targets = genus_targets.to(device=z.device, dtype=torch.long)
    genus_family_index = genus_family_index.to(device=z.device, dtype=torch.long)
    if genus_family_index.numel() == 0 or int(genus_family_index.min()) < 0:
        raise ValueError("family indices must be non-negative")

    w = F.normalize(genus_proxy_weight, dim=1)
    n_families = int(genus_family_index.max().item()) + 1
    family_sums = z.new_zeros((n_families, z.shape[1]))
    family_sums.index_add_(0, genus_family_index, w)
    family_counts = torch.bincount(genus_family_index, minlength=n_families)

    target_family = genus_family_index[genus_targets]
    valid = family_counts[target_family] >= 2
    if not bool(valid.any()):
        return z.sum() * 0.0

    q = z[valid]
    g = genus_targets[valid]
    f = target_family[valid]

    # Ordinary family prototypes are the negatives.  For the true family only,
    # replace the prototype with a leave-one-genus-out version so the query cannot
    # succeed by matching its own learned genus proxy.
    family_proto = F.normalize(family_sums, dim=1)
    logits = (q @ family_proto.T) / temperature
    loo_proto = F.normalize(family_sums[f] - w[g], dim=1)
    loo_target_logit = (q * loo_proto).sum(dim=1) / temperature

    current = logits.gather(1, f[:, None]).squeeze(1)
    correction = loo_target_logit - current
    logits = logits + F.one_hot(f, num_classes=n_families).to(logits.dtype) * correction[:, None]
    return F.cross_entropy(logits, f)


def taxonomy_distribution_loss(z: torch.Tensor, lineages: Sequence[Sequence[str]],
                               temperature: float = 0.1, beta: float = 0.7):
    """Match embedding-neighbour probabilities to hierarchical taxonomic relatedness.

    ``lineages`` are ordered [species, genus, family, order, class, phylum].
    Pairwise distance is defined by the number of unshared ranks after walking from
    phylum toward species.  The implementation is vectorized over pairs so it stays
    practical inside the training loop.
    """
    n = z.shape[0]
    if n < 2:
        return z.sum() * 0.0
    if len(lineages) != n:
        raise ValueError(f"expected {n} lineages, got {len(lineages)}")

    # Convert each rank's strings to compact integer IDs once, then perform the
    # O(B^2) pairwise hierarchy calculation on-device instead of in Python loops.
    rank_ids = []
    rank_valid = []
    for rank_idx in range(len(RANKS)):
        vals = [(lin[rank_idx] if rank_idx < len(lin) else "") or "" for lin in lineages]
        mapping = {v: i for i, v in enumerate(sorted({v for v in vals if v}))}
        ids = torch.tensor([mapping.get(v, -1) for v in vals], device=z.device, dtype=torch.long)
        rank_ids.append(ids)
        rank_valid.append(ids >= 0)

    shared = torch.zeros((n, n), device=z.device, dtype=z.dtype)
    alive = torch.ones((n, n), device=z.device, dtype=torch.bool)
    # Walk phylum -> ... -> species. Once a pair differs, lower ranks no longer
    # count as shared hierarchy, matching the original definition.
    for rank_idx in reversed(range(len(RANKS))):
        ids = rank_ids[rank_idx]
        valid = rank_valid[rank_idx]
        eq = (ids[:, None] == ids[None, :]) & valid[:, None] & valid[None, :]
        alive = alive & eq
        shared = shared + alive.to(z.dtype)

    distance = float(len(RANKS)) - shared
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    target_logits = (-beta * distance).masked_fill(eye, -torch.inf)
    p = torch.softmax(target_logits, dim=1)

    sim = (z @ z.T) / temperature
    qlog = torch.log_softmax(sim.masked_fill(eye, -torch.inf), dim=1)
    # Diagonal has p=0, qlog=-inf; replace it before multiplication to avoid 0*-inf.
    qlog = qlog.masked_fill(eye, 0.0)
    return -(p * qlog).sum(dim=1).mean()


def _prototypes(z: torch.Tensor, labels: Sequence[str], keep: torch.Tensor):
    classes = sorted({labels[i] for i in range(len(labels)) if bool(keep[i])})
    protos = []
    for c in classes:
        idx = torch.tensor([bool(keep[i]) and labels[i] == c for i in range(len(labels))], device=z.device)
        protos.append(F.normalize(z[idx].mean(dim=0), dim=0))
    return classes, torch.stack(protos) if protos else z.new_empty((0, z.shape[1]))


def episodic_hierarchy_loss(z: torch.Tensor, genus: Sequence[str], family: Sequence[str],
                            support_mask: torch.Tensor, query_mask: torch.Tensor,
                            pseudo_novel_mask: torch.Tensor, temperature: float = 0.1):
    """Known queries classify to genus; pseudo-novel queries classify to family.

    pseudo_novel_mask marks queries whose genus was intentionally removed from the
    support set for this episode. This teaches higher-rank placement without a
    learned NOVEL class; conformal calibration remains responsible for abstention.
    """
    losses = []
    gclasses, gp = _prototypes(z, genus, support_mask)
    fclasses, fp = _prototypes(z, family, support_mask)
    for i in range(z.shape[0]):
        if not bool(query_mask[i]):
            continue
        if bool(pseudo_novel_mask[i]):
            if family[i] not in fclasses:
                continue
            logits = (z[i] @ fp.T) / temperature
            target = torch.tensor([fclasses.index(family[i])], device=z.device)
        else:
            if genus[i] not in gclasses:
                continue
            logits = (z[i] @ gp.T) / temperature
            target = torch.tensor([gclasses.index(genus[i])], device=z.device)
        losses.append(F.cross_entropy(logits[None, :], target))
    return torch.stack(losses).mean() if losses else z.sum() * 0.0
