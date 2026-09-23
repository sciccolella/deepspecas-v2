#!/usr/bin/env bash
# Stage 19.5 pilot, on pasteur (plans/signal-native-v2-stage19-junction-tail.md).
#
#   p0      A0 inputs: b12 and b1 regenerated with the tail off, indels off and
#           @RG on (b12 keeps its base reads for p1/p1b)
#   p1      b12 with the tail on (TAIL_RATE_SCALE, default 1.0) and indels on,
#           reusing b12_p0's steps 1-5 (base reads regenerate bit-identically,
#           which is what A0 proves)
#   p1b     the one allowed recalibration: same as p1, new TAIL_RATE_SCALE
#   gates   dsignal.verify on a finished batch's BAMs (needs them, so remote)
#   prune   drop b12_p0's kept base reads once the pilot is concluded
#
# The per-batch inputs no longer exist on pasteur (19.0 inventory); the local
# design files are rsynced to sim/stage19_designs/<batch>/ first. Every unit's
# wall clock is appended to sim/batches/stage19_times.tsv (the 20 h abort rule).
#
# Launch (pasteur discipline -- CLAUDE.md):
#   cat /proc/loadavg
#   nohup limitram 48G nice -n 19 bash sim/stage19_pilot.sh p0 > logs_stage19_p0.log 2>&1 &
#   sleep 60; cat /proc/loadavg
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=${ROOT:-/data/conda/ciccolella/deepSpecas-v2}
ENVS=${ENVS:-/data/conda/ciccolella/envs}
DSRENDER_PY=${DSRENDER_PY:-$ENVS/DSrender/bin/python}
REF=$ROOT/ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa
export THREADS=${THREADS:-8}
INDEL=${INDEL:-3.516e-05}   # sim/tail_model.json indel.del_rate_per_base
cd "$ROOT"

TIMES=sim/batches/stage19_times.tsv
mkdir -p sim/batches
[[ -s $TIMES ]] || printf 'unit\tstart\tend\tseconds\n' > "$TIMES"

timed() {  # timed <unit> <cmd...>
  local unit=$1; shift
  local t0; t0=$(date +%s)
  "$@"
  local t1; t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' "$unit" "$(date -d @"$t0" +%FT%T)" "$(date -d @"$t1" +%FT%T)" $((t1 - t0)) >> "$TIMES"
  echo "[$unit] $((t1 - t0)) s; load $(cut -d' ' -f1-3 /proc/loadavg)" >&2
}

stage_design() {  # stage_design <batch> <tag>
  local src=sim/stage19_designs/$1 dst=sim/batches/$2
  mkdir -p "$dst"
  [[ -e $dst/design.tsv ]] || cp "$src/design.tsv" "$dst/design.tsv"
  if [[ -s $src/design.geomwin.tsv && ! -e $dst/design.geomwin.tsv ]]; then
    cp "$src/design.geomwin.tsv" "$dst/design.geomwin.tsv"
  fi
}

reuse_base() {  # reuse_base <from_tag> <to_tag>: steps 1-5 outputs, reads copied (p1 rewrites them)
  local a=sim/batches/$1 b=sim/batches/$2
  mkdir -p "$b/logs"
  for f in design.tsv neighbors.tsv exon_structure.tsv novel.fa novel_junctions.tsv \
           count_matrix.tsv transcript_ids.txt psi_check.tsv design_final.tsv \
           spliced_ids.txt spliced.gtf spliced.fa premrna.fa transcripts.fa transcripts.ordered.fa; do
    if [[ -e $a/$f && ! -e $b/$f ]]; then cp "$a/$f" "$b/$f"; fi
  done
  if [[ ! -s $b/reads/sample_06_2.fasta ]]; then
    rm -rf "$b/reads"
    cp -r "$a/reads" "$b/reads"
    rm -f "$b/reads/.tail_merged" "$b/reads/.indels_done"
  fi
}

case ${1:?usage: stage19_pilot.sh p0|p1|p1b|gates <tag>|prune} in
  p0)
    stage_design b12 b12_p0
    timed b12_p0 env DESIGN_EXTRACT=design.tsv PRUNE=0 sim/run_batch.sh b12_p0 12
    stage_design b1 b1_p0
    timed b1_p0 env DESIGN_EXTRACT=design.geomwin.tsv PRUNE=1 sim/run_batch.sh b1_p0 1
    ;;
  p1)
    reuse_base b12_p0 b12_p1
    timed b12_p1 env DESIGN_EXTRACT=design.tsv TAIL=1 TAIL_RATE_SCALE="${TAIL_RATE_SCALE:-1.0}" \
      INDEL_RATE="$INDEL" PRUNE=1 sim/run_batch.sh b12_p1 12
    ;;
  p1b)
    : "${TAIL_RATE_SCALE:?p1b needs TAIL_RATE_SCALE (the only knob P1-prime may change)}"
    reuse_base b12_p0 b12_p1b
    timed b12_p1b env DESIGN_EXTRACT=design.tsv TAIL=1 TAIL_RATE_SCALE="$TAIL_RATE_SCALE" \
      INDEL_RATE="$INDEL" PRUNE=1 sim/run_batch.sh b12_p1b 12
    ;;
  gates)
    tag=${2:?gates <tag>}
    B=sim/batches/$tag
    rc=0
    timed "verify_$tag" "$DSRENDER_PY" -m dsignal.verify --csv "$B/extract.csv" --ref "$REF" \
      > "$B/logs/verify.log" 2>&1 || rc=$?
    echo "verify exit $rc" >> "$B/logs/verify.log"
    grep -E '^\[(coverage|junction) parity\]|^verify exit' "$B/logs/verify.log"
    ;;
  prune)
    rm -rf sim/batches/b12_p0/reads sim/batches/b12_p0/aligned/cond[12]_rep[123]/Aligned.sortedByCoord.out.bam*
    ;;
  *) echo "unknown unit $1" >&2; exit 2 ;;
esac
