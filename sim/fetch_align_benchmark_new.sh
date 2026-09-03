#!/usr/bin/env bash
# Stage 8 / Track A1: acquire and align the fresh-benchmark source study
# (ENCSR376FGR SRSF3 knockdown vs ENCSR264TUE non-target control, HepG2,
# GSE88229/GSE88148) into c1_new.bam (knockdown) / c2_new.bam (control).
#
# Same STAR invocation as sim/align_reads.sh (--outSAMattributes All,
# --outSAMtype BAM SortedByCoordinate, same genomeDir), adapted for 2 real
# paired-end runs per condition instead of 3 simulated replicates.
#
# Usage: sim/fetch_align_benchmark_new.sh <work_dir> <star_index> <threads>
#
# Stage 12 / Track B reuses this for further KD/control pairs. The four runs
# and the output prefix are overridable from the environment; unset, every
# default reproduces the original SRSF3 invocation byte for byte:
#
#   C1_REP1=SRR3469390 C1_REP2=SRR3469391 \
#   C2_REP1=SRR3469516 C2_REP2=SRR3469517 OUT_PREFIX=ptbp1 \
#   sim/fetch_align_benchmark_new.sh <work_dir> <star_index> 12
set -euo pipefail

WORK_DIR=$1
STAR_INDEX=$2
THREADS=${3:-12}

STAR_BIN=${STAR_BIN:-STAR-avx2}
FASTQ_DIR="$WORK_DIR/fastq"
ALIGN_DIR="$WORK_DIR/align"
mkdir -p "$FASTQ_DIR" "$ALIGN_DIR"
ulimit -n 65536 2>/dev/null || true

# condition 1 = SRSF3 knockdown (ENCSR376FGR), condition 2 = non-target control (ENCSR264TUE)
declare -A RUNS=(
  [cond1_rep1]=${C1_REP1:-SRR4421711}
  [cond1_rep2]=${C1_REP2:-SRR4421712}
  [cond2_rep1]=${C2_REP1:-SRR4421557}
  [cond2_rep2]=${C2_REP2:-SRR4421558}
)
OUT_PREFIX=${OUT_PREFIX:-new}

# fastqs are kept uncompressed -- gzip of a ~40M-read-pair fastq is
# single-threaded and was measured to take ~50 min/file, dominating the
# whole pipeline for no benefit since STAR reads them exactly once and
# they are deleted after alignment anyway (CLAUDE.md: only BAMs/TSVs
# leave pasteur).
for name in "${!RUNS[@]}"; do
  srr=${RUNS[$name]}
  if [[ ! -f "$FASTQ_DIR/${srr}_2.fastq" ]]; then
    echo "fetching $srr ($name)"
    prefetch -O "$FASTQ_DIR" "$srr"
    fasterq-dump --split-files --threads "$THREADS" -O "$FASTQ_DIR" "$FASTQ_DIR/$srr/$srr.sra"
    rm -rf "$FASTQ_DIR/$srr"
  fi
done

for name in "${!RUNS[@]}"; do
  srr=${RUNS[$name]}
  sample_out="$ALIGN_DIR/$name/"
  mkdir -p "$sample_out"
  "$STAR_BIN" \
    --runThreadN "$THREADS" \
    --genomeDir "$STAR_INDEX" \
    --readFilesIn "$FASTQ_DIR/${srr}_1.fastq" "$FASTQ_DIR/${srr}_2.fastq" \
    --outSAMattributes All \
    --outSAMtype BAM SortedByCoordinate \
    --outFileNamePrefix "$sample_out" \
    > "$ALIGN_DIR/${name}.star.log" 2>&1
  samtools index "${sample_out}Aligned.sortedByCoord.out.bam"
done

samtools merge -f "$ALIGN_DIR/c1_${OUT_PREFIX}.bam" \
  "$ALIGN_DIR/cond1_rep1/Aligned.sortedByCoord.out.bam" \
  "$ALIGN_DIR/cond1_rep2/Aligned.sortedByCoord.out.bam"
samtools index "$ALIGN_DIR/c1_${OUT_PREFIX}.bam"

samtools merge -f "$ALIGN_DIR/c2_${OUT_PREFIX}.bam" \
  "$ALIGN_DIR/cond2_rep1/Aligned.sortedByCoord.out.bam" \
  "$ALIGN_DIR/cond2_rep2/Aligned.sortedByCoord.out.bam"
samtools index "$ALIGN_DIR/c2_${OUT_PREFIX}.bam"

echo "wrote $ALIGN_DIR/c1_${OUT_PREFIX}.bam $ALIGN_DIR/c2_${OUT_PREFIX}.bam"
