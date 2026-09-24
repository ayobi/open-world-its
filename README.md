# Open-world evaluation of a learned fungal ITS embedding

Code, deviation log and manuscript for:

> O'Brien, A. & Gardette, A. *A learned fungal ITS embedding does not outperform
> correctly configured alignment in open-world evaluation.*

The paper is an evaluation protocol with a purpose-trained ITS encoder as its
worked example. Everything needed to rerun the corrected evaluation and rebuild
every table and figure is here; large artefacts are in a separate data deposit.

## Layout

| path | contents |
|---|---|
| `itsnet/openworld/` | encoder, losses, split, conformal layer, exact-length inference |
| `itsnet/model.py`, `data.py`, `labels.py` | modules the encoder imports. `model.py` also defines an ITS segmentation model, which this paper does not use |
| `scripts/` | data preparation, training (M0 to M4), freezing, sealed and corrected evaluation |
| `analysis/` | alignment baseline, paired uncertainty, historical benchmark rebuild |
| `manuscript/` | LaTeX source, the CSVs behind every table and figure, and their generators |
| `DEVIATIONS.md` | post-opening corrections, each committed before the metric it governs |
| `environment.lock`, `environment/` | training and analysis environments |

## Verifying the code

Every corrected evaluation records the SHA-256 of its evaluator and model
sources. They match the files in this repository:

```bash
sha256sum itsnet/openworld/inference_exact.py itsnet/model.py itsnet/openworld/model.py \
          itsnet/openworld/benchmark.py itsnet/openworld/m0.py \
          scripts/eval_openworld_final_exact.py scripts/eval_openworld_dev_exact.py
# 9057a0cf  cbe5aee3  bc0ca969  1113c63d  7dc95d2f  3fca4737  0c57313e
```

## Reproducing the results

Run from the repository root with `PYTHONPATH=.`. Inputs come from the data
deposit (split table, checkpoints) and from UNITE.

1. **FASTAs from the split table:** `analysis/materialize_fastas.py`. It checks
   all 24 partition-by-view counts against the published figures.
2. **Alignment baseline:** `analysis/run_identity_searches.sh`, exhaustive VSEARCH
   with query coverage 0.8 (`QUERY_COV=0` gives the unfiltered search), then
   `analysis/identity_baseline.py` and `analysis/compare_hits.py`.
3. **Corrected encoder evaluation:** `scripts/run_primary_exact.sh` or
   `scripts/eval_openworld_final_exact.py`, then `scripts/eval_openworld_dev_exact.py`
   for each checkpoint and `scripts/score_historical_cosine_exact.py` for the
   leakage-safe checkpoint.
4. **Paired comparisons:** `analysis/paired_test.py`, `analysis/paired_genus_bootstrap.py`,
   `scripts/build_historical_per_query.py`, `analysis/rebuild_historical_benchmark.py`,
   `scripts/paired_uncertainty.py`.
5. **Manuscript:** in `manuscript/analysis/`, run `make_tables.py`,
   `make_figures.py` and `make_primary.py`, then `pdflatex main.tex` twice in
   `manuscript/`. Every number quoted in the prose is a macro in `numbers.tex`.

The sealed evaluators (`eval_openworld_final.py`, `eval_openworld_dev.py`,
`score_historical_cosine.py`) use padded inference and are kept for provenance;
their outputs appear in the manuscript only where labelled as sealed.

## Licences

Code: MIT. The split table and other UNITE-derived data in the data deposit are
redistributed under UNITE's CC BY-SA 4.0 licence, with attribution to the UNITE
general FASTA release for Fungi, 19 February 2025 (doi:10.15156/BIO/3301229).
