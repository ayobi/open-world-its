"""Dataset construction, taxonomy-aware splitting, and augmentation.

Two things in here are load-bearing for the science:

1. SPLITTING. Random splits leak badly: UNITE contains many near-identical
   sequences per species hypothesis, so a random split reports test accuracy
   that is really memorisation. We split by holding out whole clades at a
   chosen rank, which is the only way to measure the thing we actually claim to
   improve -- generalisation to lineages the model has never seen.

2. AUGMENTATION. Truncation and ONT-like error are applied WITH label tracking.
   Indels shift every downstream coordinate, so an error simulator that does not
   carry the label vector along silently corrupts every boundary label after the
   first indel.
"""
from __future__ import annotations

import gzip
import re

import numpy as np
import torch

from .labels import IGNORE, N_STATES, STATE_IDX

RANKS = ["k", "p", "c", "o", "f", "g", "s"]
_TAX_RE = re.compile(r"\b([kpcofgs])__([^;|\s]+)")
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def read_fasta(path):
    """Yield (header, sequence). Handles plain and gzipped files."""
    op = gzip.open if str(path).endswith(".gz") else open
    header, chunks = None, []
    with op(path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def parse_taxonomy(header):
    """Pull a {rank: taxon} dict out of a UNITE-style header.

    Handles both `...|k__Fungi;p__Ascomycota;...` and bare `k__...` forms.
    Missing ranks are simply absent.
    """
    return {m.group(1): m.group(2) for m in _TAX_RE.finditer(header)}


def seq_id(header):
    """Return the full whitespace-delimited FASTA identifier.

    UNITE headers are pipe-delimited and the first field is often a species
    label rather than a unique accession.  ITSx --preserve T retains the
    complete original identifier, so stripping at the first pipe causes many
    distinct records to collide.
    """
    return header.split()[0]


def clade_split(headers, rank="f", frac_test=0.15, frac_val=0.10, seed=0):
    """Hold out entire clades at `rank`. Returns {seq_id: 'train'|'val'|'test'}.

    Sequences with no assignment at `rank` fall back to the next coarser rank
    that is available, and go to train if none is.
    """
    rng = np.random.default_rng(seed)
    coarser = RANKS[: RANKS.index(rank) + 1][::-1]

    key_of = {}
    for h in headers:
        tax = parse_taxonomy(h)
        key = None
        for r in coarser:
            if r in tax:
                key = f"{r}__{tax[r]}"
                break
        key_of[seq_id(h)] = key

    clades = sorted({k for k in key_of.values() if k is not None})
    rng.shuffle(clades)
    n_test = int(len(clades) * frac_test)
    n_val = int(len(clades) * frac_val)
    assign = {}
    for i, c in enumerate(clades):
        assign[c] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return {sid: (assign.get(k, "train") if k else "train") for sid, k in key_of.items()}


# -- augmentation ---------------------------------------------------------

def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def random_truncate(seq, labels, rng, min_frac=0.35, p=0.5):
    """Randomly clip one or both ends.

    This is the single most important augmentation: it is what teaches the CRF
    to start and end the path mid-region, which is exactly the case where ITSx
    returns nothing because it cannot form a full anchor chain.
    """
    if rng.random() > p or len(seq) < 120:
        return seq, labels
    L = len(seq)
    keep = int(L * rng.uniform(min_frac, 1.0))
    start = rng.integers(0, L - keep + 1)
    return seq[start:start + keep], labels[start:start + keep]


def ont_errors(seq, labels, rng, sub=0.008, ins=0.004, dele=0.006, homopolymer_boost=3.0):
    """Inject R10.4-like error, carrying labels through indels.

    Indel rates are boosted inside homopolymer runs, which is the dominant ONT
    error mode. Deletions drop the base and its label; insertions duplicate the
    neighbouring label, which is the right choice because an inserted base sits
    inside whatever region its neighbours belong to.
    """
    out_seq, out_lab = [], []
    run_char, run_len = None, 0
    for i, ch in enumerate(seq):
        if ch == run_char:
            run_len += 1
        else:
            run_char, run_len = ch, 1
        boost = homopolymer_boost if run_len >= 4 else 1.0
        r = rng.random()
        if r < dele * boost:
            continue
        if r < (dele + ins) * boost:
            out_seq.append(rng.choice(list("ACGT")))
            out_lab.append(labels[i])
        if rng.random() < sub:
            out_seq.append(rng.choice([c for c in "ACGT" if c != ch]))
        else:
            out_seq.append(ch)
        out_lab.append(labels[i])
    return "".join(out_seq), np.asarray(out_lab, dtype=np.int64)


# -- dataset --------------------------------------------------------------

class ItsDataset(torch.utils.data.Dataset):
    """(sequence, base-wise labels) with optional augmentation.

    `records` is a list of (seq_id, sequence, label_vector).
    """

    def __init__(self, records, augment=False, max_len=6000, seed=0):
        self.records = records
        self.augment = augment
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        for sid, seq, lab in records:
            if len(seq) != len(lab):
                raise ValueError(f"{sid}: sequence/label length mismatch")
            obs = lab[lab != IGNORE]
            if obs.size and (np.diff(obs) < 0).any():
                raise ValueError(
                    f"{sid}: non-monotonic observed labels; the CRF cannot score "
                    "this path. Check labels_from_record."
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        sid, seq, lab = self.records[i]
        if self.augment:
            seq, lab = random_truncate(seq, lab, self.rng)
            seq, lab = ont_errors(seq, lab, self.rng)
        seq, lab = seq[: self.max_len], lab[: self.max_len]
        return sid, seq, np.asarray(lab, dtype=np.int64)


def collate(batch, downsample=1):
    """Pad to the longest member (and up to a multiple of `downsample`)."""
    from .model import encode_sequence

    lens = [len(s) for _, s, _ in batch]
    T = max(lens)
    if downsample > 1:
        T += (-T) % downsample
    B = len(batch)
    x = torch.zeros(B, 4, T)
    # Pad with IGNORE, not 0. Padded positions are excluded by `mask`, but a
    # 0 there means OUTER5, and any future code path that forgets the mask
    # would silently train on fabricated labels.
    y = torch.full((B, T), IGNORE, dtype=torch.long)
    m = torch.zeros(B, T, dtype=torch.bool)
    ids = []
    for i, (sid, seq, lab) in enumerate(batch):
        L = len(seq)
        x[i, :, :L] = encode_sequence(seq)
        y[i, :L] = torch.from_numpy(lab)
        m[i, :L] = True
        ids.append(sid)
    return ids, x, y, m


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batch similar-length sequences so padding (and wasted FLOPs) stay low."""

    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = np.argsort(self.lengths + self.rng.uniform(0, 50, len(self.lengths)))
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        for b in batches:
            yield list(b)

    def __len__(self):
        return int(np.ceil(len(self.lengths) / self.batch_size))
