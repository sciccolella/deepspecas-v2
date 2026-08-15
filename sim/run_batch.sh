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
    --n-nn-plain "$N_NN" --n-nn-hard "$N_NN" --n-nn-novel "$N_NN" \
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
if [[ ! -s $B/count_matrix.tsv ]]; then
  step "$TAG: build_experiment (layers 2-4 on)"
  "$PY" -m sim.build_experiment --design "$B/design.tsv" \
    --events-dir "$EVENTS" --transcript-lengths sim/transcript_lengths.tsv \
    --neighbors "$B/neighbors.tsv" --novel-junctions "$B/novel_junctions.tsv" \
    --out-matrix "$B/count_matrix.tsv" \
    --out-transcripts "$B/transcript_ids.txt" \
    --out-psi-check "$B/psi_check.tsv" \
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
if [[ ! -s $B/labels.tsv ]]; then
  step "$TAG: build_extract_csv"
  "$PY" -m sim.build_extract_csv --design "$B/design.tsv" \
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
