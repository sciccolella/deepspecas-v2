#!/usr/bin/env bash
# One simulation batch, end to end: design -> count matrix -> FASTA ->
# polyester -> real STAR realignment -> dsignal.extract -> PSI-recovery check.
#
# This is the recipe that plans/signal-native-v2-checkpoint.md had only ever
# described in prose and that was driven by hand for every noise layer and for
# the 1260-event scale-up. The ~7,500-event scale-up runs it five times, so it
# is written down here instead: the parameters below (fraglen/fragsd/readlen,
# the layer-5 PCR/adapter settings, the gene depth) are exactly the ones every
# prior layer used, and changing them changes what a batch is comparable to.
#
# Usage (on pasteur, from /data/conda/ciccolella/deepSpecas-v2):
#   sim/run_batch.sh <tag> <seed> [prior_design.tsv ...]
#
# Each prior design.tsv is passed to select_events.py --exclude-genes, so
# batches select disjoint gene sets; without it, different seeds re-pick the
# same genes and leak across batches in a way per-batch GroupKFold cannot see.
#
# Every step skips itself if its output already exists, so a batch that dies
# halfway (STAR OOM, disk) can be resumed by re-running the same command.
set -euo pipefail

TAG=${1:?usage: run_batch.sh <tag> <seed> [prior_design.tsv ...]}
SEED=${2:?usage: run_batch.sh <tag> <seed> [prior_design.tsv ...]}
shift 2
PRIORS=("$@")

ROOT=${ROOT:-/data/conda/ciccolella/deepSpecas-v2}
ENVS=${ENVS:-/data/conda/ciccolella/envs}
PY=${PY:-$ENVS/DSpoly2v2/bin/python}
RSCRIPT=${RSCRIPT:-$ENVS/DSsim/bin/Rscript}
GFFREAD=${GFFREAD:-$ENVS/DSsim/bin/gffread}
THREADS=${THREADS:-24}
export PATH="$ENVS/DSpoly2v2/bin:$PATH"

REF=$ROOT/ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa
GTF=$ROOT/ref/Homo_sapiens.GRCh38.99.gtf
STAR_INDEX=$ROOT/star_index
EVENTS=$ROOT/sim/events

# --- design composition: 25% NN (720 positive + 240 negative), per
# --- plans/signal-native-v2-scaleup-7500.md. The 1260-event batch was 43% NN.
N_PER_LABEL=${N_PER_LABEL:-180}
N_NN=${N_NN:-80}
# split three ways (Stage 11 Stage 2a) -- each defaults to N_NN so an
# existing batch tag's re-run is unaffected unless overridden individually.
N_NN_PLAIN=${N_NN_PLAIN:-$N_NN}
N_NN_HARD=${N_NN_HARD:-$N_NN}
N_NN_NOVEL=${N_NN_NOVEL:-$N_NN}
# effect-size draw for positives: "uniform" (|dPSI| ~ U(0.3,0.8), the original)
# or "beta" (fitted to the curated benchmark; 81.6% of real events fall below
# the uniform draw's 0.30 floor). See sim/select_events.py::_sample_psi_pair.
PSI_DIST=${PSI_DIST:-uniform}
# Stage 3 knobs (plan Stage 3.1-3.3). All default to the pre-Stage-3 behaviour,
# so re-running any earlier batch tag reproduces it exactly.
#   NULL_DPSI_SIGMA  hard negatives get |dPSI| ~ HalfNormal(sigma) instead of
#                    exactly 0. 0.075 is the null median measured from this
#                    simulation's own replicate noise (sim/null_dpsi_scale.py),
#                    NOT from the benchmark.
#   COOC_FRAC        fraction of the gene's other catalogued events that also
#                    change between conditions, so a window is not "one event
#                    moves, everything else is frozen".
# Use them together with PSI_DIST=paired: moving positives down while negatives
# stay pinned at zero is what the beta batch did, and it collapsed the margin.
NULL_DPSI_SIGMA=${NULL_DPSI_SIGMA:-0}
COOC_FRAC=${COOC_FRAC:-0}
# Stage 11 knobs (plans/signal-native-v2-trash-sim.md Stage 2a/2b). All
# default to today's values / off, so re-running any earlier batch tag with
# no TRASH_* vars set reproduces design.tsv and count_matrix.tsv bit-identically.
PREMRNA_FRAC_LOW=${PREMRNA_FRAC_LOW:-0.01}
PREMRNA_FRAC_HIGH=${PREMRNA_FRAC_HIGH:-0.10}
NOVEL_FRAC_LOW=${NOVEL_FRAC_LOW:-0.005}
NOVEL_FRAC_HIGH=${NOVEL_FRAC_HIGH:-0.03}
NOVEL_NN_FRAC_LOW=${NOVEL_NN_FRAC_LOW:-0.05}
NOVEL_NN_FRAC_HIGH=${NOVEL_NN_FRAC_HIGH:-0.20}
NEIGHBOR_DEPTH_LOW=${NEIGHBOR_DEPTH_LOW:-100.0}
NEIGHBOR_DEPTH_HIGH=${NEIGHBOR_DEPTH_HIGH:-3000.0}
GENE_MEAN_DEPTH=${GENE_MEAN_DEPTH:-2000.0}
# TRASH layer: class-conditional rates (measured real rates, Stage 9), off by
# default (all 0). TRASH_RATE_MODE=uniform + TRASH_UNIFORM_RATE is the
# label-independent control arm (Stage 3 arm D).
TRASH_RATE_MODE=${TRASH_RATE_MODE:-class-conditional}
TRASH_UNIFORM_RATE=${TRASH_UNIFORM_RATE:-0}
TRASH_RATE_A3=${TRASH_RATE_A3:-0}
TRASH_RATE_A5=${TRASH_RATE_A5:-0}
TRASH_RATE_CSE=${TRASH_RATE_CSE:-0}
TRASH_RATE_RI=${TRASH_RATE_RI:-0}
TRASH_RATE_NN=${TRASH_RATE_NN:-0}
TRASH_LABEL_POLICY=${TRASH_LABEL_POLICY:-keep}
TRASH_PREMRNA_FRAC_LOW=${TRASH_PREMRNA_FRAC_LOW:-0.08}
TRASH_PREMRNA_FRAC_HIGH=${TRASH_PREMRNA_FRAC_HIGH:-0.16}
TRASH_NOVEL_FRAC_LOW=${TRASH_NOVEL_FRAC_LOW:-0.10}
TRASH_NOVEL_FRAC_HIGH=${TRASH_NOVEL_FRAC_HIGH:-0.25}
# read simulation, unchanged since noise layer 1 / layer 5
FRAGLEN=${FRAGLEN:-250}
FRAGSD=${FRAGSD:-25}
READLEN=${READLEN:-100}
PCR_RATE=${PCR_RATE:-0.1}
PCR_LAMBDA=${PCR_LAMBDA:-1}
ADAPTER=${ADAPTER:-TRUE}

B=$ROOT/sim/batches/$TAG
mkdir -p "$B" "$B/logs"
cd "$ROOT"

step() { echo -e "\n[$(date +%H:%M:%S)] === $* ===" >&2; }

# 1. design ------------------------------------------------------------------
if [[ ! -s $B/design.tsv ]]; then
  step "$TAG: select_events (seed $SEED, ${#PRIORS[@]} prior batches excluded)"
  EXCL=()
  for p in "${PRIORS[@]}"; do EXCL+=(--exclude-genes "$p"); done
  "$PY" -m sim.select_events \
    --events-dir "$EVENTS" \
    --gene-spans sim/gene_spans.tsv \
    --transcript-lengths sim/transcript_lengths.tsv \
    --chrom-lengths "$REF.fai" \
    --n-per-label "$N_PER_LABEL" \
    --n-nn-plain "$N_NN_PLAIN" --n-nn-hard "$N_NN_HARD" --n-nn-novel "$N_NN_NOVEL" \
    --psi-dist "$PSI_DIST" \
    --null-dpsi-sigma "$NULL_DPSI_SIGMA" \
    --seed "$SEED" --out "$B/design.tsv" 2>&1 | tee "$B/logs/select_events.log"
fi

# 2. background gene set -----------------------------------------------------
if [[ ! -s $B/neighbors.tsv ]]; then
  step "$TAG: gene_neighbors"
  "$PY" -m sim.gene_neighbors --design "$B/design.tsv" \
    --gene-spans sim/gene_spans.tsv --out "$B/neighbors.tsv" \
    2>&1 | tee "$B/logs/gene_neighbors.log"
fi

if [[ ! -s $B/exon_structure.tsv ]]; then
  step "$TAG: gtf_exon_structure"
  "$PY" -m sim.gtf_exon_structure --gtf "$GTF" --design "$B/design.tsv" \
    --neighbors "$B/neighbors.tsv" --out "$B/exon_structure.tsv" \
    2>&1 | tee "$B/logs/exon_structure.log"
fi

if [[ ! -s $B/novel.fa ]]; then
  step "$TAG: novel_junctions (noise layer 4)"
  "$PY" -m sim.novel_junctions --exon-structure "$B/exon_structure.tsv" \
    --reference "$REF" --out-fasta "$B/novel.fa" --out-tsv "$B/novel_junctions.tsv" \
    --seed "$SEED" 2>&1 | tee "$B/logs/novel_junctions.log"
fi

# 3. count matrix ------------------------------------------------------------
if [[ ! -s $B/count_matrix.tsv || ! -s $B/design_final.tsv ]]; then
  step "$TAG: build_experiment (layers 2-4 on)"
  "$PY" -m sim.build_experiment --design "$B/design.tsv" \
    --events-dir "$EVENTS" --transcript-lengths sim/transcript_lengths.tsv \
    --neighbors "$B/neighbors.tsv" --novel-junctions "$B/novel_junctions.tsv" \
    --out-matrix "$B/count_matrix.tsv" \
    --out-transcripts "$B/transcript_ids.txt" \
    --out-psi-check "$B/psi_check.tsv" \
    --out-design "$B/design_final.tsv" \
    --cooc-frac "$COOC_FRAC" \
    --gene-mean-depth "$GENE_MEAN_DEPTH" \
    --premrna-frac-low "$PREMRNA_FRAC_LOW" --premrna-frac-high "$PREMRNA_FRAC_HIGH" \
    --novel-frac-low "$NOVEL_FRAC_LOW" --novel-frac-high "$NOVEL_FRAC_HIGH" \
    --novel-nn-frac-low "$NOVEL_NN_FRAC_LOW" --novel-nn-frac-high "$NOVEL_NN_FRAC_HIGH" \
    --neighbor-depth-low "$NEIGHBOR_DEPTH_LOW" --neighbor-depth-high "$NEIGHBOR_DEPTH_HIGH" \
    --trash-rate-mode "$TRASH_RATE_MODE" --trash-uniform-rate "$TRASH_UNIFORM_RATE" \
    --trash-rate A3 "$TRASH_RATE_A3" --trash-rate A5 "$TRASH_RATE_A5" \
    --trash-rate CSE "$TRASH_RATE_CSE" --trash-rate RI "$TRASH_RATE_RI" \
    --trash-rate NN "$TRASH_RATE_NN" \
    --trash-label-policy "$TRASH_LABEL_POLICY" \
    --trash-premrna-frac-low "$TRASH_PREMRNA_FRAC_LOW" --trash-premrna-frac-high "$TRASH_PREMRNA_FRAC_HIGH" \
    --trash-novel-frac-low "$TRASH_NOVEL_FRAC_LOW" --trash-novel-frac-high "$TRASH_NOVEL_FRAC_HIGH" \
    --seed "$SEED" 2>&1 | tee "$B/logs/build_experiment.log"
fi

# 4. FASTA, in count-matrix row order ----------------------------------------
if [[ ! -s $B/transcripts.ordered.fa ]]; then
  step "$TAG: FASTA assembly"
  grep -v -e '^PREMRNA_' -e '^NOVEL_' "$B/transcript_ids.txt" > "$B/spliced_ids.txt"
  "$PY" sim/filter_gtf.py "$B/spliced_ids.txt" "$GTF" "$B/spliced.gtf"
  "$GFFREAD" -w "$B/spliced.fa" -g "$REF" "$B/spliced.gtf"
  "$PY" -m sim.build_premrna_fasta --design "$B/design.tsv" \
    --gene-spans sim/gene_spans.tsv --reference "$REF" \
    --neighbors "$B/neighbors.tsv" --out "$B/premrna.fa" \
    2>&1 | tee "$B/logs/premrna.log"
  cat "$B/spliced.fa" "$B/premrna.fa" "$B/novel.fa" > "$B/transcripts.fa"
  "$PY" sim/order_fasta.py "$B/transcript_ids.txt" "$B/transcripts.fa" \
    "$B/transcripts.ordered.fa" 2>&1 | tee "$B/logs/order_fasta.log"
fi

# 5. reads -------------------------------------------------------------------
if [[ ! -s $B/reads/sample_06_2.fasta ]]; then
  step "$TAG: polyester"
  mkdir -p "$B/reads"
  "$RSCRIPT" sim/run_polyester.R "$B/transcripts.ordered.fa" "$B/count_matrix.tsv" \
    "$B/reads" "$FRAGLEN" "$FRAGSD" "$READLEN" "$SEED" \
    "$PCR_RATE" "$PCR_LAMBDA" "$ADAPTER" 2>&1 | tee "$B/logs/polyester.log"
fi

# 6. real STAR realignment ---------------------------------------------------
if [[ ! -s $B/aligned/c2.bam.bai ]]; then
  step "$TAG: STAR"
  sim/align_reads.sh "$B/reads" "$STAR_INDEX" "$B/aligned" "$THREADS" fasta \
    2>&1 | tee "$B/logs/align.log"
fi

# 7. training records --------------------------------------------------------
# design_final.tsv (Stage 11) carries the TRASH layer's label rewrite; it is
# identical to design.tsv when the layer is off (TRASH_LABEL_POLICY=keep and
# all TRASH_RATE_* are 0).
if [[ ! -s $B/labels.tsv ]]; then
  step "$TAG: build_extract_csv"
  "$PY" -m sim.build_extract_csv --design "$B/design_final.tsv" \
    --bam1 "$B/aligned/c1.bam" --bam2 "$B/aligned/c2.bam" \
    --out "$B/extract.csv" --labels-out "$B/labels.tsv"
fi

# count, not mere existence: a half-finished extract leaves the directory there.
# `ls` on a missing directory exits non-zero, which under `set -e -o pipefail`
# kills the whole script -- hence the explicit guard rather than `ls 2>/dev/null`.
N_ROWS=$(wc -l < "$B/extract.csv")
if [[ -d $B/npz ]]; then
  N_NPZ=$(find "$B/npz" -maxdepth 1 -name '*.npz' | wc -l)
else
  N_NPZ=0
fi
if [[ $N_NPZ -ne $N_ROWS ]]; then
  step "$TAG: dsignal.extract ($N_NPZ/$N_ROWS records present)"
  "$PY" -m dsignal.extract --csv "$B/extract.csv" --out-dir "$B/npz" \
    2>&1 | tee "$B/logs/extract.log"
fi

# 8. gate inputs -------------------------------------------------------------
if [[ ! -s $B/psi_recovery.tsv ]]; then
  step "$TAG: psi_recovery_check"
  "$PY" -m sim.psi_recovery_check --design "$B/design.tsv" \
    --events-dir "$EVENTS" --bam1 "$B/aligned/c1.bam" --bam2 "$B/aligned/c2.bam" \
    --out "$B/psi_recovery.tsv" 2>&1 | tee "$B/logs/psi_recovery.log"
fi

step "$TAG: done -- $(ls "$B/npz" | wc -l) npz records"
