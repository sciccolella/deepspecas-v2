#!/usr/bin/env bash
# Align simulated per-replicate reads with STAR (same version/index/
# --outSAMattributes All as the real benchmark BAMs), then merge 3
# replicates per condition into c1.bam/c2.bam -- exactly how
# benchmark/c1.bam and benchmark/c2.bam were built. Per
# plans/signal-native-v2.md Phase 2 step 3.1: "Never go simulator -> BAM
# directly."
#
# Usage:
#   sim/align_reads.sh <reads_dir> <star_index> <out_dir> <n_threads> <ext>
#
# <reads_dir> must contain, per sample cond{1,2}_rep{1,2,3}:
#   sample_<NN>_1.<ext>[.gz]  sample_<NN>_2.<ext>[.gz]
# where <NN> is polyester's 1-based sample index in count-matrix column
# order (cond1_rep1=01 ... cond2_rep3=06). <ext> is fasta or fastq,
# whichever simulate_experiment_countmat actually wrote.
set -euo pipefail

READS_DIR=$1
STAR_INDEX=$2
OUT_DIR=$3
THREADS=${4:-24}
EXT=${5:-fasta}

STAR_BIN=${STAR_BIN:-STAR-avx2}
mkdir -p "$OUT_DIR"

# STAR's sorted-BAM output opens threads * outBAMsortingBinsN file
# descriptors; the default ulimit -n 1024 is too low at --runThreadN 24.
ulimit -n 65536 2>/dev/null || true

declare -A SAMPLE_IDX=( [cond1_rep1]=01 [cond1_rep2]=02 [cond1_rep3]=03 [cond2_rep1]=04 [cond2_rep2]=05 [cond2_rep3]=06 )

readFilesCommand=""
r1=""
r2=""
for name in "${!SAMPLE_IDX[@]}"; do
  idx=${SAMPLE_IDX[$name]}
  f1="$READS_DIR/sample_${idx}_1.${EXT}"
  f2="$READS_DIR/sample_${idx}_2.${EXT}"
  [[ -f "${f1}.gz" ]] && f1="${f1}.gz"
  [[ -f "${f2}.gz" ]] && f2="${f2}.gz"
  if [[ ! -f "$f1" || ! -f "$f2" ]]; then
    echo "missing reads for $name: $f1 / $f2" >&2
    exit 1
  fi

  sample_out="$OUT_DIR/$name/"
  mkdir -p "$sample_out"
  cmd=""
  [[ "$f1" == *.gz ]] && cmd="--readFilesCommand zcat"

  "$STAR_BIN" \
    --runThreadN "$THREADS" \
    --genomeDir "$STAR_INDEX" \
    --readFilesIn "$f1" "$f2" \
    $cmd \
    --outSAMattributes All \
    --outSAMtype BAM SortedByCoordinate \
    --outFileNamePrefix "$sample_out" \
    > "$OUT_DIR/${name}.star.log" 2>&1

  samtools index "${sample_out}Aligned.sortedByCoord.out.bam"
done

samtools merge -f "$OUT_DIR/c1.bam" \
  "$OUT_DIR/cond1_rep1/Aligned.sortedByCoord.out.bam" \
  "$OUT_DIR/cond1_rep2/Aligned.sortedByCoord.out.bam" \
  "$OUT_DIR/cond1_rep3/Aligned.sortedByCoord.out.bam"
samtools index "$OUT_DIR/c1.bam"

samtools merge -f "$OUT_DIR/c2.bam" \
  "$OUT_DIR/cond2_rep1/Aligned.sortedByCoord.out.bam" \
  "$OUT_DIR/cond2_rep2/Aligned.sortedByCoord.out.bam" \
  "$OUT_DIR/cond2_rep3/Aligned.sortedByCoord.out.bam"
samtools index "$OUT_DIR/c2.bam"

echo "wrote $OUT_DIR/c1.bam $OUT_DIR/c2.bam"
