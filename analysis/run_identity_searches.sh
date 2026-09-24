#!/usr/bin/env bash
#
# Step one: rerun every percent-identity search with an exhaustive best-hit
# configuration, and produce the alignment baselines the review asks for, in a
# single pass so that every identity number in the paper comes from one
# documented configuration.
#
# WHY THIS IS A RERUN AND NOT A NEW ANALYSIS
# ------------------------------------------
# The deposited searches used the VSEARCH defaults --maxaccepts 1 and
# --maxrejects 32. Two consequences:
#
#   --maxaccepts 1   stops the search at the FIRST target that passes --id.
#                    At --id 0.5 almost any fungal ITS reference passes, so the
#                    search terminates almost immediately and the reported hit
#                    is close to the first candidate examined rather than the
#                    best one. --top_hits_only cannot help: with one acceptable
#                    hit there is nothing to choose between.
#   --maxrejects 32  stops after 32 consecutive failures, which manufactures
#                    "no hit" results for divergent queries. Part of the
#                    censored set is therefore a heuristic artefact rather than
#                    a property of the sequences.
#
# In VSEARCH, 0 means unlimited for both, which is what gives a true best hit.
#
# WHAT THIS PRODUCES
# ------------------
#   1. historical LSO/LGO, corrected: replaces the Table 7 / Figure 6 numbers.
#   2. same-view identity on CAL and TEST: the missing non-learned baseline on
#      the primary evaluation.
#   3. spacer-against-core identity: the missing baseline for the cross-view
#      claim. ITS-core contains ITS1 and ITS2, so an ITS1 query and an ITS-core
#      reference share homologous sequence and alignment can make this
#      comparison too. Identity is computed over the aligned region only
#      (VSEARCH default --iddef 2), which is the correct comparison here.
#
# Old outputs are never overwritten: new files carry the .exhaustive.tsv
# suffix so analysis/compare_hits.py can quantify what the fix changed.
#
# USAGE
# -----
#   bash analysis/run_identity_searches.sh --probe 200      # timing estimate
#   bash analysis/run_identity_searches.sh --dry-run        # print commands
#   bash analysis/run_identity_searches.sh                  # real run
#
set -euo pipefail

# ===========================================================================
# EDIT THIS BLOCK. Every path is checked before anything runs.
# ===========================================================================
OUT=${OUT:-runs/identity_exhaustive}

# historical benchmark. Each query set has its own reference collection by
# design, so they are paired and must not share one --db.
HIST=${HIST:-~/Downloads/its-novelty}
HIST_LGO_Q=${HIST_LGO_Q:-$HIST/out_lgo_its2/queries.fasta}
HIST_LGO_REF=${HIST_LGO_REF:-$HIST/out_lgo_its2/refs.fasta}
HIST_LSO_Q=${HIST_LSO_Q:-$HIST/out_lso_its2/queries.fasta}
HIST_LSO_REF=${HIST_LSO_REF:-$HIST/out_lso_its2/refs.fasta}

# open-world split, one reference FASTA per view
# open-world FASTAs, as written by analysis/materialize_fastas.py
FDIR=${FDIR:-data/openworld/fasta}
REF_core=${REF_core:-$FDIR/train_ref_core.fasta}
REF_its1=${REF_its1:-$FDIR/train_ref_its1.fasta}
REF_its2=${REF_its2:-$FDIR/train_ref_its2.fasta}
QDIR=${QDIR:-$FDIR}

# search settings
ID_FLOOR=${ID_FLOOR:-0.5}
STRAND=${STRAND:-plus}        # set to "both" if any orientation is uncertain
THREADS=${THREADS:-32}
# Fallback if the exhaustive run is too slow: export ACCEPTS=50 REJECTS=500
ACCEPTS=${ACCEPTS:-0}
REJECTS=${REJECTS:-0}
# The reported results use a query-coverage requirement of 0.8 (Section 2.4).
# QUERY_COV=0 disables it and reproduces the unfiltered exhaustive search.
QUERY_COV=${QUERY_COV:-0.8}
# Uncomment if your FASTA headers contain spaces and your manifest IDs do too.
# NOTRUNC="--notrunclabels"
NOTRUNC=${NOTRUNC:-}
# ===========================================================================

DRY=0
PROBE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --probe)   PROBE="$2"; shift 2 ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

VSEARCH=${VSEARCH:-$(command -v vsearch || true)}
[[ -x "$VSEARCH" ]] || { echo "vsearch not found; set VSEARCH=/path/to/vsearch" >&2; exit 1; }
# No pipe here on purpose. vsearch writes its banner to stderr and keeps
# writing after `head -1` exits, so `vsearch --version 2>&1 | head -1` returns
# 141 under `set -o pipefail` and kills this script with no output at all.
_vv=$("$VSEARCH" --version 2>&1 || true)
VSEARCH_VERSION=${_vv%%$'\n'*}

# job list: label | query fasta | reference fasta
JOBS=()
add() { JOBS+=("$1|$2|$3"); }

add hist_lgo "$HIST_LGO_Q" "$HIST_LGO_REF"
add hist_lso "$HIST_LSO_Q" "$HIST_LSO_REF"
for V in core its1 its2; do
  eval "REF=\$REF_$V"
  for P in cal_known test_known test_novel; do
    add "${P}_${V}_vs_${V}" "$QDIR/${P}_${V}.fasta" "$REF"
  done
done
# cross-view alignment baseline: each spacer against the core references
for V in its1 its2; do
  for P in cal_known test_known test_novel; do
    add "${P}_${V}_vs_core" "$QDIR/${P}_${V}.fasta" "$REF_core"
  done
done

# ---- fail fast: check every input before running anything ----------------
missing=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r label q db <<< "$job"
  q=${q/#\~/$HOME}; db=${db/#\~/$HOME}
  [[ -s "$q"  ]] || { echo "MISSING query:     $q   ($label)"; missing=1; }
  [[ -s "$db" ]] || { echo "MISSING reference: $db  ($label)"; missing=1; }
done
if [[ $missing -eq 1 ]]; then
  echo
  echo "Fix the paths in the EDIT THIS BLOCK section, or override them:"
  echo "  QDIR=/path/to/queries REF_core=/path/core.fasta bash $0 --dry-run"
  exit 1
fi
echo "all ${#JOBS[@]} input pairs found"

mkdir -p "$OUT"
PROV="$OUT/provenance.txt"
{
  echo "date:      $(date -Is)"
  echo "host:      $(hostname)"
  echo "vsearch:   $VSEARCH_VERSION"
  echo "id floor:  $ID_FLOOR"
  echo "maxaccepts: $ACCEPTS   maxrejects: $REJECTS   (0 = unlimited)"
  echo "query_cov: $QUERY_COV"
  echo "strand:    $STRAND     threads: $THREADS"
  echo "notrunclabels: ${NOTRUNC:-<not set>}"
  echo
} > "$PROV"

# ---- optional timing probe on a subset of the first job ------------------
if [[ "$PROBE" -gt 0 ]]; then
  IFS='|' read -r label q db <<< "${JOBS[0]}"
  q=${q/#\~/$HOME}; db=${db/#\~/$HOME}
  sub="$OUT/probe_${PROBE}.fasta"
  awk -v n="$PROBE" '/^>/{c++} c>n{exit} {print}' "$q" > "$sub"
  echo "timing probe: $PROBE queries from $label against $(basename "$db")"
  start=$(date +%s)
  "$VSEARCH" --usearch_global "$sub" --db "$db" --id "$ID_FLOOR" \
          --maxaccepts "$ACCEPTS" --maxrejects "$REJECTS" --query_cov "$QUERY_COV" --top_hits_only \
          --strand "$STRAND" --threads "$THREADS" $NOTRUNC \
          --userout "$OUT/probe.tsv" --userfields query+target+id \
          --quiet 2>/dev/null
  el=$(( $(date +%s) - start ))
  total=$(grep -c '^>' "$q" || true)
  echo "  ${el}s for $PROBE queries; this job has ${total} queries"
  if [[ "$PROBE" -gt 0 && "$el" -gt 0 ]]; then
    echo "  extrapolated: ~$(( el * total / PROBE / 60 )) min for this job alone"
  fi
  echo
  echo "Query IDs as VSEARCH emits them (check these match your manifest):"
  awk 'NR<=3{print "    " $1}' "$OUT/probe.tsv"
  echo
  echo "If the extrapolation is unacceptable, rerun with:"
  echo "  ACCEPTS=50 REJECTS=500 bash $0 --probe $PROBE"
  echo "and report whichever setting you finally use."
  exit 0
fi

# ---- run ------------------------------------------------------------------
for job in "${JOBS[@]}"; do
  IFS='|' read -r label q db <<< "$job"
  q=${q/#\~/$HOME}; db=${db/#\~/$HOME}
  tsv="$OUT/${label}.exhaustive.tsv"
  cmd=("$VSEARCH" --usearch_global "$q" --db "$db" --id "$ID_FLOOR"
       --maxaccepts "$ACCEPTS" --maxrejects "$REJECTS" --query_cov "$QUERY_COV" --top_hits_only
       --strand "$STRAND" --threads "$THREADS"
       --userout "$tsv" --userfields query+target+id
       --log "$OUT/${label}.vsearch.log")
  [[ -n "$NOTRUNC" ]] && cmd+=("$NOTRUNC")

  if [[ $DRY -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"; echo; continue
  fi

  echo "[$label] $(basename "$q") vs $(basename "$db")"
  start=$(date +%s)
  "${cmd[@]}" >/dev/null 2>&1
  el=$(( $(date +%s) - start ))
  nq=$(grep -c '^>' "$q" || true)
  nh=$(cut -f1 "$tsv" | sort -u | wc -l | tr -d ' ')
  echo "  ${el}s | $nq queries | $nh with a hit | $(( nq - nh )) censored"
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$nq" "$nh" "$(( nq - nh ))" "${el}s" >> "$PROV"
done

if [[ $DRY -eq 0 ]]; then
  echo
  echo "wrote $OUT/*.exhaustive.tsv and $PROV"
  echo
  echo "Next:"
  echo "  1. python3 analysis/compare_hits.py --old <old.raw.tsv> --new $OUT/hist_lgo.exhaustive.tsv"
  echo "     (repeat for hist_lso) to quantify what the fix changed."
  echo "  2. Rebuild the per-query table with the corrected identity arm:"
  echo "     python3 analysis/build_historical_per_query.py \\"
  echo "         --lgo-identity $OUT/hist_lgo.exhaustive.tsv \\"
  echo "         --lso-identity $OUT/hist_lso.exhaustive.tsv ..."
  echo "  3. Rerun paired_uncertainty.py, then identity_baseline.py for the"
  echo "     TEST and cross-view baselines."
fi
