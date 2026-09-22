"""Region label schema and ITSx-positions parsing.

Label states are STRICTLY ORDERED 5'->3'. The CRF transition mask is derived
from this ordering, so the index order here is load-bearing -- do not reorder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Ordered state vocabulary. OUTER5/OUTER3 absorb primers, adapters, vector and
# any unannotated flank, and give the CRF somewhere to put junk that is neither
# rRNA gene nor spacer. Two distinct outer states (rather than one shared
# "other") are required to keep the chain monotonic.
STATES = ["OUTER5", "SSU", "ITS1", "S58", "ITS2", "LSU", "OUTER3"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)

# Partial-supervision sentinel. A position labelled IGNORE is one whose true
# state is genuinely unknown, as opposed to known-to-be-outer. The CRF
# marginalises over all states at these positions instead of scoring one.
#
# This matters because reference databases are not all built the same way.
# UNITE records are primer-trimmed ITS amplicons: SSU and LSU are not
# "undetected", they are ABSENT FROM THE MOLECULE. Filling the bases ahead of
# ITS1 with OUTER5 would assert "this is not rRNA" over sequence whose identity
# nobody knows -- and would actively teach the model to call real SSU fragments
# outer, which is the exact opposite of what the detector is for.
IGNORE = -100

# The four boundaries ITSx reports and that we benchmark on.
BOUNDARIES = [
    ("ITS1", "start"),
    ("ITS1", "end"),
    ("ITS2", "start"),
    ("ITS2", "end"),
]

# ITSx region names -> our state names.
ITSX_REGIONS = {
    "SSU": "SSU",
    "ITS1": "ITS1",
    "5.8S": "S58",
    "ITS2": "ITS2",
    "LSU": "LSU",
}

_FIELD_RE = re.compile(r"(SSU|ITS1|5\.8S|ITS2|LSU):\s*(\d+)-(\d+)")
_LEN_RE = re.compile(r"^(\d+)\s*bp")


@dataclass
class ItsxRecord:
    """One line of an ITSx `.positions.txt` file."""

    seq_id: str
    length: int
    regions: dict  # state name -> (start, end), 0-based half-open
    chimeric: bool
    partial: bool
    raw: str

    @property
    def complete(self) -> bool:
        """True when all five ITSx regions are present (the 'easy' case)."""
        return all(r in self.regions for r in ("SSU", "ITS1", "S58", "ITS2", "LSU"))

    @property
    def has_its(self) -> bool:
        return "ITS1" in self.regions or "ITS2" in self.regions


def parse_positions_line(line):
    """Parse a single ITSx positions line.

    Format is tab-separated:
        <id>\t<N> bp.\tSSU: 1-32\tITS1: 33-220\t5.8S: Not found\t...\t<comment>

    Coordinates in the file are 1-based inclusive; we convert to 0-based
    half-open on the way in so everything downstream is Python-native.
    Regions reported as "Not found" are simply absent from `regions`.
    """
    line = line.rstrip("\n")
    if not line.strip():
        return None
    parts = line.split("\t")
    raw_id = parts[0].strip().lstrip(">")
    if not raw_id:
        return None
    # With ITSx --preserve T, positions retain the complete original FASTA
    # identifier.  Preserve the full whitespace-delimited identifier here too;
    # in UNITE the first pipe-delimited field is not unique.
    seq_id = raw_id.split()[0]

    length = 0
    for p in parts[1:]:
        m = _LEN_RE.match(p.strip())
        if m:
            length = int(m.group(1))
            break

    regions = {}
    for m in _FIELD_RE.finditer(line):
        name, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        # 1-based inclusive -> 0-based half-open
        regions[ITSX_REGIONS[name]] = (start - 1, end)

    low = line.lower()
    return ItsxRecord(
        seq_id=seq_id,
        length=length,
        regions=regions,
        chimeric="chimeric" in low,
        partial=("partial" in low) or ("broken" in low),
        raw=line,
    )


def read_positions(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            rec = parse_positions_line(line)
            if rec is not None:
                out[rec.seq_id] = rec
    return out


def labels_from_record(rec, seq_len=None):
    """Expand an ITSx record into a base-wise label vector.

    Flank handling is conditional on what ITSx actually anchored:

      * SSU detected  -> bases ahead of it are genuinely outside the operon,
                         labelled OUTER5.
      * SSU absent    -> bases ahead of the first region could be undetected
                         SSU or could be adapter. Labelled IGNORE.

    Same logic mirrored for LSU / OUTER3. Interior gaps are IGNORE rather than
    forward-filled: a gap means ITSx anchored both sides but not the middle,
    which is unknown, not "same as the left neighbour".

    A boundary is only supervised when both its flanking regions were found.
    For a typical UNITE record (ITS1 + 5.8S + ITS2, no SSU or LSU) that means
    ITS1-end and ITS2-start are real labels, while ITS1-start and ITS2-end are
    where the sequence happens to have been trimmed. Training on the latter as
    if they were boundaries teaches the model the primer design, not biology.
    """
    n = seq_len if seq_len is not None else rec.length
    if n <= 0:
        raise ValueError("%s: unknown sequence length" % rec.seq_id)

    y = np.full(n, IGNORE, dtype=np.int64)
    for state, (s, e) in rec.regions.items():
        s = max(0, s)
        e = min(n, e)
        if e > s:
            y[s:e] = STATE_IDX[state]

    covered = np.nonzero(y != IGNORE)[0]
    if covered.size == 0:
        return y  # nothing anchored: wholly unsupervised

    first, last = covered[0], covered[-1]
    if "SSU" in rec.regions:
        y[:first] = STATE_IDX["OUTER5"]
    if "LSU" in rec.regions:
        y[last + 1:] = STATE_IDX["OUTER3"]
    # interior gaps and unanchored flanks stay IGNORE
    return y


CORE = ("ITS1", "S58", "ITS2")
FULL = ("SSU", "ITS1", "S58", "ITS2", "LSU")


def trainable_tier(rec, min_len=100, max_len=8000):
    """Which supervision a record can provide: 'full', 'core', or None.

    Demanding all five regions is wrong for amplicon reference databases and
    silently discards ~99.5% of UNITE, because those records contain no SSU or
    LSU to find. The tiers separate what a record can actually teach:

      full  SSU..LSU present. Supervises every boundary INCLUDING the
            ITS1-start and ITS2-end anchors. Rare in UNITE (479 of ~101k in
            the July 2026 release), abundant in full-operon references like
            EUKARYOME -- which is why EUKARYOME is not an enrichment for this
            project but the only source of the flank signal.
      core  ITS1 + 5.8S + ITS2. Supervises the two 5.8S-anchored boundaries;
            the outer extents are IGNORE. This is the bulk of UNITE.
      None  no usable 5.8S anchor, or chimeric/partial -> evaluation only.
    """
    if rec.chimeric or rec.partial:
        return None
    if not (min_len <= rec.length <= max_len):
        return None

    def monotonic(names):
        prev = -1
        for state in names:
            if state not in rec.regions:
                return False
            s, e = rec.regions[state]
            if s < prev or e <= s:
                return False
            prev = e
        return True

    if monotonic(FULL):
        return "full"
    if monotonic(CORE):
        return "core"
    return None


def is_trainable(rec, min_len=100, max_len=8000):
    """Back-compatible boolean: True when the record gives any supervision."""
    return trainable_tier(rec, min_len, max_len) is not None


def boundaries_from_labels(y):
    """Extract the four ITSx-comparable boundaries from a label vector.

    Returns 0-based half-open (start, end) semantics: `start` is the first base
    of the region, `end` is one past the last.
    """
    out = {}
    for region, which in BOUNDARIES:
        idx = np.nonzero(y == STATE_IDX[region])[0]
        if idx.size == 0:
            out[(region, which)] = None
        else:
            out[(region, which)] = int(idx[0]) if which == "start" else int(idx[-1]) + 1
    return out


def regions_from_labels(y):
    """Collapse a label vector into {state: (start, end)} half-open spans."""
    out = {}
    for state, i in STATE_IDX.items():
        idx = np.nonzero(y == i)[0]
        if idx.size:
            out[state] = (int(idx[0]), int(idx[-1]) + 1)
    return out
