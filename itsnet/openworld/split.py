"""Leakage-resistant open-world taxonomic splits.

The final novelty question is genus-level: known queries come from species absent
from the reference but genera represented in it; novel queries come from genera
absent from the reference while their family remains represented.

The split builder therefore works in two stages:
1. withhold complete genera into DEV_NOVEL / TEST_NOVEL;
2. among remaining genera, withhold complete species into DEV_KNOWN,
   CAL_KNOWN and TEST_KNOWN while leaving >=1 species of each genus in TRAIN_REF.

TEST_NOVEL genera never participate in training or model selection. CAL_KNOWN is
reserved for conformal calibration after the encoder and novelty score are frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import random
import re


@dataclass(frozen=True)
class TaxonRecord:
    record_id: str
    family: str
    genus: str
    species: str


@dataclass(frozen=True)
class OpenWorldSplitConfig:
    seed: int = 17
    min_records_per_genus: int = 2
    min_species_for_known_holdout: int = 4
    dev_novel_frac: float = 0.10
    test_novel_frac: float = 0.10
    max_novel_genera_per_family: int = 2


SPLITS = (
    "TRAIN_REF",
    "DEV_KNOWN",
    "DEV_NOVEL",
    "CAL_KNOWN",
    "TEST_KNOWN",
    "TEST_NOVEL",
)


def _clean(x: str) -> str:
    return (x or "").strip()


def _named_taxon(x: str) -> bool:
    x = _clean(x)
    return bool(x) and "Incertae_sedis" not in x and not x.lower().startswith("unidentified")


def _named_species(genus: str, species: str) -> bool:
    if not _named_taxon(genus) or not _named_taxon(species):
        return False
    return re.search(r"_(sp|cf|aff)(_|$)", species.lower()) is None


def _group(records: Iterable[TaxonRecord]):
    by_genus: Dict[Tuple[str, str], List[TaxonRecord]] = defaultdict(list)
    for r in records:
        fam, gen, sp = _clean(r.family), _clean(r.genus), _clean(r.species)
        if not _named_taxon(fam) or not _named_taxon(gen):
            continue
        by_genus[(fam, gen)].append(r)
    return by_genus


def _choose_novel_genera(
    by_genus: Mapping[Tuple[str, str], Sequence[TaxonRecord]], cfg: OpenWorldSplitConfig
):
    rng = random.Random(cfg.seed)
    by_family: Dict[str, List[str]] = defaultdict(list)
    for (fam, gen), rs in by_genus.items():
        if len(rs) >= cfg.min_records_per_genus:
            by_family[fam].append(gen)

    dev, test = set(), set()
    for fam in sorted(by_family):
        genera = sorted(set(by_family[fam]))
        rng.shuffle(genera)
        # Always leave at least one genus in TRAIN_REF so higher-rank placement
        # remains meaningful for every held-out novel genus.
        capacity = min(cfg.max_novel_genera_per_family, max(0, len(genera) - 1))
        if capacity == 0:
            continue

        target_dev = int(round(len(genera) * cfg.dev_novel_frac))
        target_test = int(round(len(genera) * cfg.test_novel_frac))
        if len(genera) >= 3 and cfg.dev_novel_frac > 0:
            target_dev = max(1, target_dev)
        if len(genera) >= 4 and cfg.test_novel_frac > 0:
            target_test = max(1, target_test)

        n_dev = min(target_dev, capacity)
        remaining = capacity - n_dev
        n_test = min(target_test, remaining)
        # If only one holdout slot exists, prefer final TEST_NOVEL over DEV_NOVEL.
        if capacity == 1 and target_test > 0:
            n_dev, n_test = 0, 1

        for gen in genera[:n_dev]:
            dev.add((fam, gen))
        for gen in genera[n_dev:n_dev + n_test]:
            test.add((fam, gen))

    return dev, test


def build_openworld_splits(
    records: Sequence[TaxonRecord], cfg: OpenWorldSplitConfig = OpenWorldSplitConfig()
) -> Dict[str, str]:
    """Return record_id -> split with genus/species leakage barriers.

    Invariants:
    - TEST_NOVEL and DEV_NOVEL genera are absent from TRAIN_REF.
    - Every novel genus has at least one other genus from the same family in TRAIN_REF.
    - DEV/CAL/TEST_KNOWN species are absent from TRAIN_REF, but their genus is present.
    - CAL_KNOWN is distinct from TEST_KNOWN.
    """
    by_genus = _group(records)
    dev_novel, test_novel = _choose_novel_genera(by_genus, cfg)
    assignment: Dict[str, str] = {}

    for key in dev_novel:
        for r in by_genus[key]:
            assignment[r.record_id] = "DEV_NOVEL"
    for key in test_novel:
        for r in by_genus[key]:
            assignment[r.record_id] = "TEST_NOVEL"

    rng = random.Random(cfg.seed + 1)
    for key in sorted(by_genus):
        if key in dev_novel or key in test_novel:
            continue
        rs = list(by_genus[key])
        by_species: Dict[str, List[TaxonRecord]] = defaultdict(list)
        for r in rs:
            by_species[_clean(r.species)].append(r)
        species = sorted(sp for sp in by_species if _named_species(key[1], sp))

        if len(species) >= cfg.min_species_for_known_holdout:
            rng.shuffle(species)
            # Hold out complete species, one per role. At least one species remains
            # in TRAIN_REF because min_species_for_known_holdout defaults to four.
            hold = {
                "DEV_KNOWN": species[0],
                "CAL_KNOWN": species[1],
                "TEST_KNOWN": species[2],
            }
            for split, sp in hold.items():
                for r in by_species[sp]:
                    assignment[r.record_id] = split
            held_species = set(hold.values())
            for sp, sp_records in by_species.items():
                if sp in held_species:
                    continue
                for r in sp_records:
                    assignment[r.record_id] = "TRAIN_REF"
        else:
            for r in rs:
                assignment[r.record_id] = "TRAIN_REF"

    # Records lacking complete family/genus/species are intentionally not emitted.
    return assignment


def validate_split_invariants(records: Sequence[TaxonRecord], assignment: Mapping[str, str]) -> List[str]:
    """Return human-readable invariant violations; empty means valid."""
    rec_by_id = {r.record_id: r for r in records}
    errors: List[str] = []

    train_genera = {(rec_by_id[i].family, rec_by_id[i].genus) for i, s in assignment.items() if s == "TRAIN_REF"}
    train_species = {(rec_by_id[i].family, rec_by_id[i].genus, rec_by_id[i].species) for i, s in assignment.items() if s == "TRAIN_REF"}
    train_families = {f for f, _ in train_genera}

    for i, split in assignment.items():
        r = rec_by_id[i]
        g = (r.family, r.genus)
        sp = (r.family, r.genus, r.species)
        if split in {"DEV_NOVEL", "TEST_NOVEL"}:
            if g in train_genera:
                errors.append(f"novel genus leaked into TRAIN_REF: {g}")
            if r.family not in train_families:
                errors.append(f"novel genus lacks represented family: {g}")
        elif split in {"DEV_KNOWN", "CAL_KNOWN", "TEST_KNOWN"}:
            if g not in train_genera:
                errors.append(f"known query genus absent from TRAIN_REF: {g}")
            if sp in train_species:
                errors.append(f"known query species leaked into TRAIN_REF: {sp}")
    return sorted(set(errors))
