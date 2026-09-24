# v3.6: protocol reframing on the corrected evaluation

## What changed

**Framing.** Title, abstract, Introduction, Results and Discussion rewritten.
The contribution is now the evaluation protocol, with the encoder as its worked
example. All three barcode views are reported with equal weight; nothing was
selected among them after seeing results.

**Results, now in this order**
1. The sealed evaluation reproduced exactly but was not batch-invariant (new).
2. The alignment baseline depends on its configuration: termination
   heuristics and the ITS-core coverage artefact (new).
3. A correctly configured alignment baseline matches or exceeds the encoder on
   the primary split, with query-level and genus-cluster uncertainty (new).
4. Identity dominates at every deployable operating point; AUROC is the wrong
   headline metric (new).
5. Cross-view retrieval is available to alignment through ITS-core (new).
6. The development-to-TEST gap is partition composition, not selection. This
   retracts the selection-optimism reading in v3.5.
7. The historical benchmark was contaminated by training data (kept).

**Methods corrected.** The encoder paragraph now states that padded positions
enter group-normalisation statistics and dilated receptive fields, and describes
exact-length inference. The VSEARCH description now gives the correct mechanism
(`--maxaccepts 1` returns the first acceptable candidate, not the best) and the
exhaustive, coverage-filtered configuration used everywhere except the sealed
historical run. New subsections: alignment baseline, paired uncertainty,
post-opening corrections and deviation log.

**Moved to the appendix, labelled sealed.** Training objectives and equations,
the development ladder (Table 3, Figure 2), the M4 cross-view matrix (Table 4,
Figure 3), and the historical benchmark (Table 7, Figure 6). These still carry
padded inference or default-settings identity.

**Pipeline.** New `analysis/make_primary.py` builds seven tables, three figures
and `numbers.tex`, all from new CSVs in `data/`. Every result number in the
abstract and Results prose is now a macro, so the text cannot drift from the
tables. Figure 1 panel c updated for the matched baseline.

**Bibliography.** Added Orsholm et al. 2026, DeLong et al. 1988, PROTAX-Fungi,
BayesANT, SINTAX, and Edgar 2018 (PeerJ), a benchmark-design paper on 16S and
fungal ITS reaching a compatible conclusion. v3.6.1 corrected the PROTAX-Fungi
title, which was wrong in v3.6, and verified the BayesANT and SINTAX
identifiers. PROTAX-Fungi volume and pages still carry a VERIFY.

## Errors caught while building this version

- The abstract first said default heuristics returned a "non-best hit" for
  50.7% of queries. That counts target changes, including swaps among equal
  ties. Queries that genuinely got a lower-identity hit were 48.3%; corrected.
- ITS-core class placement with the coverage filter appeared as 98.6% (sweep
  run) and 98.7% (head-to-head run). Same flags, but multithreaded VSEARCH does
  not resolve exact ties deterministically. Prose now quotes the head-to-head
  run, and the sweep table says why its row can differ by up to 0.1.
- A regex replacement silently turned `\ref` and `\todo` into control
  characters in the historical section. It compiled cleanly. Now fixed and
  guarded by an assertion.

## Still pending, in order

1. **Corrected development matrix.** Rescore DEV with exact-length inference.
   This replaces appendix Tables 3 and 4 and makes the M4 half of Table 9
   clean (currently padded DEV against exact TEST, which is conservative).
2. **Historical rebuild** with the coverage-filtered identity arm, via
   `build_historical_per_query.py` then `rebuild_historical_benchmark.py`.
   Replaces appendix Table 7 and Figure 6.
3. **Overlap and union on the corrected predictions.** The Jaccard and union
   figures in Section 3.4 are still from the sealed cosines:
   `paired_test.py --cosine` built from `runs/m4_final_exact_v1/final_predictions.tsv`.
4. VERIFY marks: lineage-aware tie rule, deviation-log send date and the
   second entry's hash, funding, CRediT, UNITE release note, PROTAX-Fungi
   volume and pages.
5. TODO marks: repository DOI, acknowledgements, environment export.

When a pending run lands, drop its CSV into `data/`, rerun the three
generators, and recompile. The prose macros update themselves.

## v3.6.2: corrected development ladder

- Tables 3 and 4, Figures 2 and 3, and Table S1 now use exact-length inference on
  all six saved checkpoints. Sealed values are kept as `data/*_sealed.csv`.
- M4 still meets all four integration targets under corrected inference, the ITS2
  placement target by 0.3 points. The Methods sentence now gives both sets of values.
- Padding had inflated every M4 cross-view cell (spacer-to-core genus accuracy by
  about 12 points) while leaving same-view AUROC within 0.015. Added to the
  batch-invariance Results as a concrete consequence.
- M3's AUROC cost relative to M1 reverses sign at ITS2 under correction, so the
  episode-versus-novelty trade-off survives only at ITS-core, where it is small.
- The development-to-TEST decomposition is now exact against exact; the residual
  stays negative at every view.
- Overlap and union recomputed on corrected predictions; that TODO is closed.
- The Table S1 note was hardcoded and became wrong under corrected values; it now
  reflects that the reduced taxonomy weight is worse on every metric at every view.
- Only the historical benchmark remains in the sealed appendix.

## v3.6.3: corrected historical benchmark

- Table 7 and Figure 6 rebuilt with both arms corrected: exhaustive,
  coverage-filtered identity, and exact-length cosine from the leakage-safe
  checkpoint (hash 99b91448..., distinct from the frozen model).
- AUROC 0.768 vs 0.703; difference +0.065, genus-clustered 95% CI +0.046 to
  +0.085. This is the one AUROC contrast in the paper that excludes zero, and it
  favours identity. Now reported in the main Results and the abstract.
- Five hardcoded Table 7 note values were sealed and have been replaced: the
  AUROC CI, DeLong z, interval widening, McNemar counts, and the censoring note.
- Censoring under the corrected search: 619 LGO and 30 LSO queries. About half of
  identity's detections at alpha = 0.05 are queries with no alignable reference.
- LSO-halving sensitivity recomputed over ten named seeds: +0.063 to +0.075, every
  clustered interval excluding zero; seed 0 lies near the low end.
- Added: padding's effect on AUROC varies in sign and size across partitions
  (TEST -0.023, DEV +0.003, historical +0.001), so it cannot be corrected after
  the fact.
- Nothing in the paper is sealed any more. Sealed values are in data/*_sealed.csv.

Remaining: lineage-aware tie-rule check (analysis); VERIFY items (send dates,
funding, CRediT, UNITE note, PROTAX-Fungi volume and pages); TODO items
(repository DOI, acknowledgements, environment export).

## v3.6.4

- Tie rule bracketed: counting a placement correct only when all tied best hits
  agree, or when any does, moves novel placement by at most 0.2 points. VERIFY
  closed. This also accounts for the 98.6 vs 98.7 ITS-core class discrepancy.
- PROTAX-Fungi DOI added (10.1111/nph.15301); volume and pages were correct.
- Every analysis is now complete. What remains is information only the authors
  hold.

## v3.6.5

- Environment TODO closed: environment.lock cited for training; runtime for the
  corrected evaluation and analyses pinned in the text and deposited as
  environment/requirements.txt.
- Leakage-safe checkpoint hash (99b91448...) added beside the frozen one.

## v3.6.6

- Funding confirmed (CORFO 23PTECCC-247149); VERIFY closed.
- CRediT rewritten: A.G. credited with conceptualization, methodology and validation,
  with the specific contributions his review made. Awaiting his confirmation.
- Acknowledgements: UNITE community only; TODO closed.

## v3.6.7

- Deviation-log passage rewritten to state only what the record shows: both entries
  were committed before their metrics (by 18 min and 64 s), on self-attested
  timestamps. The earlier claim that each hash was sent to A.G. first is removed.

## v3.6.8

- UNITE source file identified from ITSx input lengths (trimmed file, not _dev); VERIFY closed.
- Reproducibility appendix now states that every recorded source hash matches the deposited file.

## v3.7

- Abstract rewritten as four numbered paragraphs (343 words); keywords trimmed to eight, alphabetical.
- Data availability paths corrected to the repository layout.

## v3.7.1

- Email-attestation mark removed; the text already states the timestamps are self-attested.
- Draft marks hidden with \draftmarksfalse; the remaining reminders stay in the source.
- Data availability rephrased to read correctly with marks hidden.
- \clearpage before the bibliography so no appendix figure lands among the references.
