#!/usr/bin/env python3
"""Prepare a training-safe open-world view table from canonical ITS views.

Actions:
- mark membership in a historical ITS2 ID set (e.g. the novelty-paper 99,808 pool);
- detect exact-sequence genus conflicts independently for core/ITS1/ITS2;
- optionally mask conflicting views rather than deleting whole records;
- add taxonomy-quality flags used for auditable split construction.

The original input table is never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict, Counter

VIEWS = ("core", "its1", "its2")


def named_taxon(x: str) -> bool:
    x = (x or "").strip()
    return bool(x) and "Incertae_sedis" not in x and not x.lower().startswith("unidentified")


def named_species(genus: str, species: str) -> bool:
    if not named_taxon(species):
        return False
    s = species.lower()
    # UNITE placeholders such as Aspergillus_sp, Genus_cf_x, Genus_aff_x are
    # not independent named species and must not define LSO-style holdouts.
    if re.search(r"_(sp|cf|aff)(_|$)", s):
        return False
    return named_taxon(genus)


def digest(seq: str) -> str:
    return hashlib.sha256(seq.upper().encode()).hexdigest()


def load_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    with open(path) as fh:
        return {line.rstrip("\n\r") for line in fh if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="canonical openworld view TSV")
    ap.add_argument("--output", required=True, help="prepared TSV")
    ap.add_argument("--audit", required=True, help="exact-conflict audit TSV")
    ap.add_argument("--historical-its2-ids", default=None,
                    help="optional one-full-header-per-line historical ITS2 ID set")
    ap.add_argument("--keep-conflicting-views", action="store_true",
                    help="annotate but do not blank views with contradictory genus labels")
    args = ap.parse_args()

    with open(args.input, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise SystemExit("empty input")
    required = {"id", "family", "genus", "species", *VIEWS}
    if not required.issubset(rows[0]):
        raise SystemExit("input missing required columns: " + ", ".join(sorted(required - set(rows[0]))))

    historical = load_ids(args.historical_its2_ids)

    taxa_by_key = defaultdict(set)
    ids_by_key = defaultdict(list)
    for r in rows:
        fam, gen = r["family"].strip(), r["genus"].strip()
        for view in VIEWS:
            seq = (r.get(view) or "").strip()
            if not seq or not gen:
                continue
            key = (view, digest(seq))
            taxa_by_key[key].add((fam, gen))
            ids_by_key[key].append(r["id"])

    conflict_keys = {k for k, taxa in taxa_by_key.items() if len(taxa) > 1}

    # JSON-like escaping is avoided in the main audit; emit one record per
    # conflicting sequence per record ID so UNITE's semicolons are harmless.
    with open(args.audit, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["view", "sha256", "family", "genus", "id"])
        for (view, sha) in sorted(conflict_keys):
            # Recover each record's own taxon from the source table below.
            wanted = set(ids_by_key[(view, sha)])
            for r in rows:
                if r["id"] in wanted:
                    w.writerow([view, sha, r["family"], r["genus"], r["id"]])

    extra = [
        "paper_its2", "named_family", "named_genus", "named_species",
        "core_conflict", "its1_conflict", "its2_conflict",
    ]
    fields = list(rows[0]) + [x for x in extra if x not in rows[0]]

    masked = Counter()
    paper_hits = 0
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r0 in rows:
            r = dict(r0)
            is_paper = r["id"] in historical if historical else False
            r["paper_its2"] = "1" if is_paper else "0"
            paper_hits += int(is_paper)
            r["named_family"] = "1" if named_taxon(r["family"]) else "0"
            r["named_genus"] = "1" if named_taxon(r["genus"]) else "0"
            r["named_species"] = "1" if named_species(r["genus"], r["species"]) else "0"
            for view in VIEWS:
                seq = (r.get(view) or "").strip()
                bad = bool(seq) and (view, digest(seq)) in conflict_keys
                r[f"{view}_conflict"] = "1" if bad else "0"
                if bad and not args.keep_conflicting_views:
                    r[view] = ""
                    masked[view] += 1
            w.writerow(r)

    print(f"records                  : {len(rows)}")
    print(f"named family + genus     : {sum(named_taxon(r['family']) and named_taxon(r['genus']) for r in rows)}")
    print(f"historical ITS2 IDs seen : {paper_hits}" + (f" / {len(historical)}" if historical else " (not supplied)"))
    print(f"conflict groups          : {len(conflict_keys)}")
    print(f"masked core views        : {masked['core']}")
    print(f"masked ITS1 views        : {masked['its1']}")
    print(f"masked ITS2 views        : {masked['its2']}")
    print(f"wrote {args.output}")
    print(f"wrote {args.audit}")


if __name__ == "__main__":
    main()
