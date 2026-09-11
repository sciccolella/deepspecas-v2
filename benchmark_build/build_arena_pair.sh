#!/usr/bin/env bash
# Stage 20.1 -- build one model-blind arena pair end to end on pasteur:
# fetch + STAR (sim/fetch_align_benchmark_new.sh) -> salmon/SUPPA2/rMATS/
# consensus + candidate CSVs (benchmark_build/run_arena.sh) -> the NN half
# (benchmark_build/make_nn_candidates.py --n-total 800 --seed 0).
#
# This is the pasteur-only `aqr_pair2_build.sh` (Stage 14) made generic and
# checked in, so pairs 3/4 (and any later pair) are reproducible from the repo.
# Every stage is resumable: each script skips outputs that already exist.
#
# The four runs must be *single* SRA runs. A replicate that spans several runs
# (pair 2's control rep2) is concatenated into a pseudo-run in a pre-step by
# the caller -- this script does not edit fetch_align_benchmark_new.sh either.
#
# Pasteur discipline (CLAUDE.md): the caller wraps this in `limitram` and
# `nice -n 19`; the four thread-pool vars are exported here; STAR/salmon/rMATS
# are capped at $THREADS (default 12).
#
# Usage (from /data/conda/ciccolella/deepSpecas-v2 on pasteur):
#   C1_REP1=SRR14845398 C1_REP2=SRR14845397 \
#   C2_REP1=SRR14831566 C2_REP2=SRR14831565 \
#   nohup nice -n 19 limitram 48G benchmark_build/build_arena_pair.sh \
#       prpf4_work prpf4_arena prpf4 > prpf4_work.log 2>&1 &
set -euo pipefail

WORK_DIR=$1
OUT_DIR=$2
PREFIX=$3
THREADS=${THREADS:-12}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PATH=/data/conda/ciccolella/envs/DSpoly2v2/bin:$PATH

ROOT=${ROOT:-/data/conda/ciccolella/deepSpecas-v2}
cd "$ROOT"
ulimit -n 65536 2>/dev/null || true

: "${C1_REP1:?set C1_REP1}" "${C1_REP2:?set C1_REP2}" "${C2_REP1:?set C2_REP1}" "${C2_REP2:?set C2_REP2}"
export C1_REP1 C1_REP2 C2_REP1 C2_REP2
export OUT_PREFIX=$PREFIX
export THREADS

echo "=== $(date) loadavg before: $(cat /proc/loadavg) ==="
echo "=== fetch + align ($PREFIX) ==="
if [[ ! -f "$WORK_DIR/align/c2_${PREFIX}.bam.bai" ]]; then
  sim/fetch_align_benchmark_new.sh "$WORK_DIR" star_index "$THREADS"
else
  echo "  merged BAMs exist, skipping"
fi
echo "=== $(date) loadavg after align: $(cat /proc/loadavg) ==="

echo "=== build arena (salmon/SUPPA2/rMATS/consensus) ==="
benchmark_build/run_arena.sh "$WORK_DIR" "$OUT_DIR" "$PREFIX"

echo "=== NN candidates ==="
IOE_DIR=$ROOT/sim/events
if [[ ! -f "$OUT_DIR/candidates/nn_candidates.tsv" ]]; then
  /data/conda/ciccolella/envs/DSpoly2v2/bin/python benchmark_build/make_nn_candidates.py \
    "$OUT_DIR/rmats_out" "$OUT_DIR" "$IOE_DIR" "$OUT_DIR/consensus.tsv" "$OUT_DIR" "$OUT_DIR/candidates" \
    "$WORK_DIR/align/c1_${PREFIX}.bam" "$WORK_DIR/align/c2_${PREFIX}.bam" \
    --n-total 800 --seed 0
fi

echo "=== $(date) loadavg at end: $(cat /proc/loadavg) ==="
echo "PAIR_DONE $OUT_DIR"
wc -l "$OUT_DIR/consensus.tsv" "$OUT_DIR/candidates/nn_candidates.tsv"
