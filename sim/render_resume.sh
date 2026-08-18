#!/bin/bash
# Finish a sharded render, stepping over events that crash plot2s.py.
#
# Why this exists: a handful of very narrow windows strip to a zero-column
# image, and PIL then raises "height and width must be > 0" from
# Image.resize, killing the whole shard. Six of twelve shards died that way,
# at different points, each losing everything after its first bad event.
# Rather than patch published v1 rendering code, this driver restarts the
# shard past the offending line: plot2s writes its outcsv incrementally, at
# 4 rows (crossfeeds) per event, so the failing event is exactly the one at
# index rows/4 of the remaining input.
#
# Shared-machine rules per CLAUDE.md: limitram, nice, thread caps.
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/data/conda/ciccolella/deepSpecas-v2
PY=/data/conda/ciccolella/envs/DSrender/bin/python
OUT=$ROOT/sim/render_squishstack
cd "$ROOT"

run_shard() {
  local i=$1
  local input="$OUT/shards/$i"
  local remaining="$OUT/$i.remaining"
  local skipped="$OUT/$i.skipped"
  : > "$skipped"
  cp "$input" "$remaining"

  # anything already rendered for this shard counts as done
  local done_rows=0
  [ -f "$OUT/$i.csv" ] && done_rows=$(wc -l < "$OUT/$i.csv")
  local done_events=$(( done_rows / 4 ))
  if [ "$done_events" -gt 0 ]; then
    tail -n +$(( done_events + 1 )) "$input" > "$remaining"
  fi

  for attempt in $(seq 1 200); do
    [ -s "$remaining" ] || break
    local before=0
    [ -f "$OUT/$i.csv" ] && before=$(wc -l < "$OUT/$i.csv")

    nice -n 19 $PY plot2s.py -i "$remaining" -o "$OUT/img" --outcsv "$OUT/$i.csv.part" \
         -m squishstack --zoom -1 --resize >> "$OUT/$i.log" 2>&1
    local rc=$?
    cat "$OUT/$i.csv.part" >> "$OUT/$i.csv" 2>/dev/null
    rm -f "$OUT/$i.csv.part"

    local after=0
    [ -f "$OUT/$i.csv" ] && after=$(wc -l < "$OUT/$i.csv")
    local made=$(( (after - before) / 4 ))

    if [ $rc -eq 0 ]; then
      touch "$OUT/$i.done"
      return 0
    fi
    # crashed: the event after the last completed one is the bad one
    sed -n "$(( made + 1 ))p" "$remaining" >> "$skipped"
    tail -n +$(( made + 2 )) "$remaining" > "$remaining.tmp" && mv "$remaining.tmp" "$remaining"
  done
  # ran out of attempts or input; only claim success if nothing is left
  [ -s "$remaining" ] || touch "$OUT/$i.done"
}

for i in "$@"; do
  [ -f "$OUT/$i.done" ] && { echo "$i already done"; continue; }
  run_shard "$i" &
done
wait
echo "== after resume:"
for f in "$OUT"/shards/part*; do
  i=$(basename "$f")
  printf "%s rows=%s skipped=%s %s\n" "$i" \
    "$(wc -l < "$OUT/$i.csv" 2>/dev/null || echo 0)" \
    "$(wc -l < "$OUT/$i.skipped" 2>/dev/null || echo 0)" \
    "$([ -f "$OUT/$i.done" ] && echo DONE)"
done
