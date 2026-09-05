#!/usr/bin/env bash
# Stage 12 / Track B (B2): build one model-blind arena from an aligned
# KD/control pair -- salmon -> SUPPA2 psi/diffSplice, rMATS, their consensus,
# and the candidate CSVs.
#
# Stage 8 ran this interactively for benchmark_new. That is exactly why Track B
# needs it as a script: an arena that cannot be rebuilt on new accessions is
# not renewable, and "renewable" is the entire point (each frozen arena is
# single-use per model generation, see plans/signal-native-v2-real-supervision.md).
#
# NO CURATION and NO MODEL PREDICTION enters any file this produces. The arena
# is model-blind by construction rather than by attestation.
#
# Every stage is skipped when its output already exists, so a failed run can be
# resumed in place rather than repeating hours of salmon.
#
# Usage:
#   benchmark_build/run_arena.sh <work_dir> <out_dir> <prefix>
# e.g.
#   benchmark_build/run_arena.sh ptbp1_work ptbp1_arena ptbp1
#
# Expects, relative to the deepSpecas-v2 root on pasteur:
#   <work_dir>/fastq/SRR*_{1,2}.fastq        (kept by fetch_align_benchmark_new.sh)
#   <work_dir>/align/c{1,2}_<prefix>.bam     (its merged output)
#   benchmark_build/salmon_index, ref/*.gtf, sim/events/*_strict.ioe
set -euo pipefail

WORK_DIR=$1
OUT_DIR=$2
PREFIX=$3

ROOT=${ROOT:-/data/conda/ciccolella/deepSpecas-v2}
SALMON=${SALMON:-/data/conda/ciccolella/envs/salmon/bin/salmon}
RMATS=${RMATS:-/data/conda/ciccolella/envs/rmats/bin/rmats.py}
# rmatspipeline (a compiled .so) sits in the rMATS install directory, NOT in the
# env's site-packages, and rmats.py's shebang is a bare `#!/usr/bin/env python`
# -- which the env does not even provide (only python3/python3.12). Invoked
# directly it dies instantly with ModuleNotFoundError: rmatspipeline.
RMATS_DIR=${RMATS_DIR:-/data/conda/ciccolella/envs/rmats/rMATS}
RMATS_PYTHON=${RMATS_PYTHON:-/data/conda/ciccolella/envs/rmats/bin/python3.12}
SUPPA=${SUPPA:-/data/conda/ciccolella/envs/DSpoly2v2/bin/suppa.py}
PYTHON=${PYTHON:-/data/conda/ciccolella/envs/DSpoly2v2/bin/python}
SALMON_INDEX=${SALMON_INDEX:-$ROOT/benchmark_build/salmon_index}
GTF=${GTF:-$ROOT/ref/Homo_sapiens.GRCh38.99.gtf}
IOE_DIR=${IOE_DIR:-$ROOT/sim/events}
READ_LENGTH=${READ_LENGTH:-100}
THREADS=${THREADS:-8}

# the four runs, in the same cond/rep order fetch_align_benchmark_new.sh used
C1_REP1=${C1_REP1:?set C1_REP1}
C1_REP2=${C1_REP2:?set C1_REP2}
C2_REP1=${C2_REP1:?set C2_REP1}
C2_REP2=${C2_REP2:?set C2_REP2}

mkdir -p "$OUT_DIR"
FQ="$WORK_DIR/fastq"
cd "$ROOT"

# numpy/MKL size their pools to the core count; pasteur is shared (CLAUDE.md)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== 1/6 salmon quant ==="
quant_one() {
  local name=$1 srr=$2
  if [[ -f "$OUT_DIR/salmon/$name/quant.sf" ]]; then
    echo "  $name: already quantified"; return
  fi
  "$SALMON" quant -i "$SALMON_INDEX" -l A \
    -1 "$FQ/${srr}_1.fastq" -2 "$FQ/${srr}_2.fastq" \
    -p "$THREADS" -o "$OUT_DIR/salmon/$name" > "$OUT_DIR/salmon_$name.log" 2>&1
}
quant_one cond1_rep1 "$C1_REP1"
quant_one cond1_rep2 "$C1_REP2"
quant_one cond2_rep1 "$C2_REP1"
quant_one cond2_rep2 "$C2_REP2"

echo "=== 2/6 TPM matrices ==="
if [[ ! -f "$OUT_DIR/cond2_tpm.tsv" ]]; then
  "$PYTHON" benchmark_build/make_tpm_matrix.py "$OUT_DIR/salmon" "$OUT_DIR"
fi

echo "=== 3/6 SUPPA2 psiPerEvent ==="
for t in SE A3 A5 RI; do
  for c in cond1 cond2; do
    [[ -f "$OUT_DIR/psi_${t}_${c}.psi" ]] && continue
    "$SUPPA" psiPerEvent -i "$IOE_DIR/events_${t}_strict.ioe" \
      -e "$OUT_DIR/${c}_tpm.tsv" -o "$OUT_DIR/psi_${t}_${c}" \
      > "$OUT_DIR/psi_${t}_${c}.log" 2>&1
  done
done

echo "=== 4/6 SUPPA2 diffSplice ==="
# SUPPA2 exits 0 having written `<out>.dpsi.temp.0` and, on this install, never
# renames it to `<out>.dpsi` (benchmark_new's Stage 8 run did get the rename, so
# this is install- or version-dependent, not a failure). The temp file is
# complete -- same row count as the reference run, log says "Done!" -- and
# differs only by carrying an extra leading `Event_id` header cell. consensus.py
# reads the final format with `read_csv(...).reset_index()`, i.e. it *needs* the
# 2-field header where the event id column is unnamed, so strip that one cell
# rather than copying the file as-is.
#
# This runs BEFORE the diffSplice loop, not after: the loop skips on `.dpsi`,
# so recovering first turns a resumed run into a no-op instead of recomputing
# every empirical p-value again.
for t in SE A3 A5 RI; do
  [[ -f "$OUT_DIR/diffsplice_${t}.dpsi" ]] && continue
  temp="$OUT_DIR/diffsplice_${t}.dpsi.temp.0"
  [[ -f "$temp" ]] || continue
  { head -1 "$temp" | sed 's/^Event_id\t//'; tail -n +2 "$temp"; } > "$OUT_DIR/diffsplice_${t}.dpsi"
  echo "  ${t}: recovered .dpsi from .temp.0 ($(wc -l < "$OUT_DIR/diffsplice_${t}.dpsi") lines)"
done

for t in SE A3 A5 RI; do
  [[ -f "$OUT_DIR/diffsplice_${t}.dpsi" ]] && continue
  "$SUPPA" diffSplice -m empirical -i "$IOE_DIR/events_${t}_strict.ioe" \
    -p "$OUT_DIR/psi_${t}_cond1.psi" "$OUT_DIR/psi_${t}_cond2.psi" \
    -e "$OUT_DIR/cond1_tpm.tsv" "$OUT_DIR/cond2_tpm.tsv" \
    -o "$OUT_DIR/diffsplice_${t}" > "$OUT_DIR/diffsplice_${t}.log" 2>&1
done

# fresh run: the loop just produced .temp.0 files, convert them the same way
for t in SE A3 A5 RI; do
  [[ -f "$OUT_DIR/diffsplice_${t}.dpsi" ]] && continue
  temp="$OUT_DIR/diffsplice_${t}.dpsi.temp.0"
  [[ -f "$temp" ]] || { echo "missing both diffsplice_${t}.dpsi and its .temp.0" >&2; exit 1; }
  { head -1 "$temp" | sed 's/^Event_id\t//'; tail -n +2 "$temp"; } > "$OUT_DIR/diffsplice_${t}.dpsi"
  echo "  ${t}: recovered .dpsi from .temp.0 ($(wc -l < "$OUT_DIR/diffsplice_${t}.dpsi") lines)"
done

echo "=== 5/6 rMATS ==="
# one merged BAM per condition, matching how benchmark_new's rMATS run was set up
if [[ ! -f "$OUT_DIR/rmats_out/SE.MATS.JC.txt" ]]; then
  echo "$WORK_DIR/align/c1_${PREFIX}.bam" > "$OUT_DIR/b1.txt"
  echo "$WORK_DIR/align/c2_${PREFIX}.bam" > "$OUT_DIR/b2.txt"
  PYTHONPATH="$RMATS_DIR${PYTHONPATH:+:$PYTHONPATH}" "$RMATS_PYTHON" "$RMATS" \
    --b1 "$OUT_DIR/b1.txt" --b2 "$OUT_DIR/b2.txt" --gtf "$GTF" \
    -t paired --readLength "$READ_LENGTH" --variable-read-length \
    --nthread "$THREADS" --od "$OUT_DIR/rmats_out" --tmp "$OUT_DIR/rmats_tmp" \
    > "$OUT_DIR/rmats.log" 2>&1
fi

echo "=== 6/6 consensus + candidate CSVs ==="
"$PYTHON" benchmark_build/consensus.py \
  "$OUT_DIR/rmats_out" "$OUT_DIR" "$IOE_DIR" "$OUT_DIR/consensus.tsv"
"$PYTHON" benchmark_build/make_candidate_csvs.py \
  "$OUT_DIR/consensus.tsv" "$OUT_DIR/candidates" \
  "$WORK_DIR/align/c1_${PREFIX}.bam" "$WORK_DIR/align/c2_${PREFIX}.bam"

echo "ARENA_DONE $OUT_DIR"
wc -l "$OUT_DIR/consensus.tsv"
