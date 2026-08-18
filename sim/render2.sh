#!/bin/bash
# Stage 2 render, second attempt. Three fixes over the first:
#
#  1. THREAD LIMITS (see below) -- 32 unlimited shards took the load average to
#     230 on this shared 64-core box and locked everyone out of a login shell.
#  2. WIDTH CAP. Render cost scales with window width and the tail is brutal:
#     median 1,916 bp, p90 19 kb, p99 110 kb, max 347 kb. Twelve shards spent
#     245 seconds at 174% CPU each producing *zero* rows, all stuck in the tail.
#     --max-width (default 50 kb) drops 4% of events; the cost is concentrated
#     in CSE (900 -> 786 at 50 kb), which is worth stating with any number that
#     comes out of this.
#  3. ROUND-ROBIN BY WIDTH. `split -n l/N` gives contiguous chunks, so whichever
#     shard owns the giants runs for hours after the others are done. Sorting by
#     width and using `split -n r/N` deals them out like cards instead.
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/data/conda/ciccolella/deepSpecas-v2
PY=/data/conda/ciccolella/envs/DSrender/bin/python
MODE=${1:-squishstack}
NSHARD=${2:-12}
MAXW=${3:-50000}
cd "$ROOT"

OUT=sim/render_$MODE
rm -rf "$OUT" && mkdir -p "$OUT/shards" "$OUT/img"
awk -F, -v m="$MAXW" '{split($1,a,":"); split(a[2],b,"-"); w=b[2]-b[1]; if (w<m) print w"\t"$0}' \
    sim/render_geomwin.csv | sort -k1,1n | cut -f2- > "$OUT/input.csv"
echo "rendering $(wc -l < "$OUT/input.csv") of $(wc -l < sim/render_geomwin.csv) events (width < $MAXW)"
split -n r/$NSHARD -d -a 2 "$OUT/input.csv" "$OUT/shards/part"

for f in "$OUT"/shards/part*; do
  i=$(basename "$f")
  ( nice -n 19 $PY plot2s.py -i "$f" -o "$OUT/img" --outcsv "$OUT/$i.csv" \
        -m "$MODE" --zoom -1 --resize > "$OUT/$i.log" 2>&1 \
    && touch "$OUT/$i.done" ) &
done
wait

ok=$(ls "$OUT"/*.done 2>/dev/null | wc -l)
echo "shards succeeded: $ok / $NSHARD"
cat "$OUT"/part*.csv > "$OUT/all.csv" 2>/dev/null
echo "rendered rows: $(wc -l < "$OUT/all.csv"), images: $(find "$OUT/img" -name '*.png' | wc -l)"
[ "$ok" -eq "$NSHARD" ] && touch "$OUT/RENDER_OK"
