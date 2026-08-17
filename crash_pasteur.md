# Making pasteur unresponsive — exact reproduction

Written 2026-08-17 after locking `pasteur.bias.disco.unimib.it` out of new login
shells for roughly 40 minutes. Everything below is the literal command sequence,
in order, so the incident can be reproduced deliberately and investigated.

**The machine never ran out of memory.** While it was unusable, `free -g`
reported 212 GB free of 251 GB and 27 GB used. What was pathological was the run
queue: **load average 230 on 64 cores, 2,946 threads**. `sshd` accepted TCP
connections on port 22 the whole time (ping 0.6 ms, `nc -z` succeeded) but login
shells took tens of minutes to appear, which is what made it look like a network
or host failure from outside.

---

## 0. Setup that preceded the incident (harmless, included for completeness)

```bash
# a render env, created fresh -- no existing env was modified
ssh pasteur '/opt/miniforge3/bin/mamba create -y -p /data/conda/ciccolella/envs/DSrender \
    -c conda-forge -c bioconda python=3.11 pysam numpy matplotlib pillow rich pytorch-cpu'

# modules shipped from the local checkout
rsync plot2s.py mc_mplp.py DSutils.py pasteur:/data/conda/ciccolella/deepSpecas-v2/

# the render input: 4,800 rows of `region,bam1,bam2,label`, built by joining
# each batch's extract.geomwin.csv with its labels.geomwin.tsv
ssh pasteur 'cd /data/conda/ciccolella/deepSpecas-v2 && \
    /data/conda/ciccolella/envs/DSrender/bin/python mkrender.py'

# single-process smoke test on 3 events -- completed in 2m32s, no problem
ssh pasteur 'cd /data/conda/ciccolella/deepSpecas-v2 && \
    /data/conda/ciccolella/envs/DSrender/bin/python plot2s.py -i /tmp/smoke.csv \
      -o /tmp/smokeimg --outcsv /tmp/smoke.out.csv -m squishstack --zoom -1 --resize'
```

## 1. The command that did it

```bash
ssh pasteur 'cd /data/conda/ciccolella/deepSpecas-v2 && \
    nohup bash sim/render.sh squishstack 32 > sim/render_squishstack.log 2>&1 &'
```

`sim/render.sh` as it existed at that moment — 32 concurrent `plot2s.py`
processes, **no thread limits, no `nice`, no memory limit**:

```bash
#!/bin/bash
set -uo pipefail
ROOT=/data/conda/ciccolella/deepSpecas-v2
PY=/data/conda/ciccolella/envs/DSrender/bin/python
MODE=${1:-squishstack}
NSHARD=${2:-24}
cd "$ROOT"

OUT=sim/render_$MODE
rm -rf "$OUT" && mkdir -p "$OUT/shards" "$OUT/img"
split -n l/$NSHARD -d -a 2 sim/render_geomwin.csv "$OUT/shards/part"

for f in "$OUT"/shards/part*; do
  i=$(basename "$f")
  ( $PY plot2s.py -i "$f" -o "$OUT/img" --outcsv "$OUT/$i.csv" \
        -m "$MODE" --zoom -1 --resize > "$OUT/$i.log" 2>&1 \
    && touch "$OUT/$i.done" ) &
done
wait
```

Note `nohup ... &`: the job survives the ssh session, so killing the client does
nothing. It ran until explicitly killed.

## 2. What was observed

| when | measurement |
|---|---|
| ~90 s after launch | 758 images written; still responsive |
| a few minutes later | `ssh` hangs; `timeout 90 ssh …` returns 124 |
| — | `ping` 0.6 ms, `getent hosts` fine, `nc -z … 22` succeeds — network healthy |
| during | `/proc/loadavg` = `230.34 231.52 222.97 198/2946` on 64 cores |
| during | `free -g` = 251 total, 27 used, **212 free** — memory was never tight |
| after kill | load fell to 18 within minutes |

## 3. Recovery

```bash
ssh pasteur 'pkill -f plot2s.py; pkill -f render.sh'
```

**Careful:** `pkill -f plot2s.py` also matches the ssh command line that carries
the pattern, so it kills your own session mid-command (this returned 255 and
truncated its own output). Use a bracketed pattern to match only the real
processes:

```bash
ssh pasteur 'pkill -f "[p]lot2s"'
ssh pasteur 'pgrep -cf "[p]lot2s"'   # same trick for counting
```

## 4. Hypothesis for why the load was ~7x the process count

32 processes cannot produce load 230 by themselves. The likely mechanism is
per-process thread pools sized to the *machine's* core count:

* `numpy`/BLAS defaults to one thread per core unless `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` are set;
* `plot2s.py` imports `torch` at module level, and torch keeps its own intra-op
  pool (`torch.set_num_threads`) which is **not** fully governed by
  `OMP_NUM_THREADS`.

Supporting evidence from the second attempt: with all four env vars set to 1,
`ps -eo pcpu` still showed each `plot2s.py` process at **~174% CPU**, i.e. about
1.7 cores each. So something in the stack — most plausibly torch — is still
multi-threading past `OMP_NUM_THREADS=1`. 32 × ~2 threads is still only ~64;
reaching 2,946 threads suggests the *unlimited* case was far worse than 2 threads
per process. Worth measuring directly:

```bash
# on one render process, with and without the env vars
ssh pasteur 'ps -o nlwp= -p <PID>'          # thread count
ssh pasteur 'cat /proc/<PID>/status | grep Threads'
```

## 5. A second, independent failure mode found afterwards

Even at 12 well-behaved shards, the render produced **zero rows in 245 seconds**
while all 12 processes burned ~174% CPU. Cause: window width. The render set has

```
n 4800   median 1,916   p75 6,087   p90 19,159   p99 109,953   max 347,345 bp
```

and `split -n l/N` hands out *contiguous* chunks, so a shard that owns part of
the tail stalls for hours. Two fixes, both now in `sim/render2.sh`: cap width
(`--max-width`, default 50 kb, drops 166 of 4,800 events) and deal the events out
round-robin **after sorting by width** (`split -n r/N`), so every shard gets a
comparable mix. With that, throughput went from ~0 to **26,490 events/hour**.

## 5b. A third failure mode: narrow windows crash the shard outright

With width capped and shards balanced, 6 of 12 shards still died — this time on
a real bug, not resources. Very narrow windows strip to a **zero-column image**,
and PIL then raises from `Image.resize`:

```
File "plot2s.py", line 328, in plot
File ".../PIL/Image.py", line 2433, in resize
ValueError: height and width must be > 0
```

Reproducer (180 bp window, `sim/batches/b3`):

```bash
echo '7:129471130-129471310,/data/conda/ciccolella/deepSpecas-v2/sim/batches/b3/aligned/c1.bam,/data/conda/ciccolella/deepSpecas-v2/sim/batches/b3/aligned/c2.bam,NN' > /tmp/bad.csv
ssh pasteur 'cd /data/conda/ciccolella/deepSpecas-v2 && \
  /data/conda/ciccolella/envs/DSrender/bin/python plot2s.py -i /tmp/bad.csv \
    -o /tmp/badimg --outcsv /tmp/bad.out.csv -m squishstack --zoom -1 --resize'
```

It is not purely a width threshold — other windows of the same size render
fine — so it depends on what survives `strip()`. One bad event kills the whole
shard and everything after it, which is why the six survivors looked "done"
while four fifths of the work was missing.

Handled in `sim/render_resume.sh` without touching published v1 rendering code:
plot2s writes its outcsv incrementally at 4 rows per event, so on a crash the
driver knows the failing event is at index `rows/4`, records it in
`partNN.skipped`, and restarts past it. Dropping the narrow tail instead would
have cost 689 events (14%) biased toward exactly the classes whose windows are
narrow (A3/A5/RI), which would have distorted the comparison this render exists
to make.

## 6. What should have been done, and now is

```bash
ssh pasteur 'cd /data/conda/ciccolella/deepSpecas-v2 && \
    nohup limitram 32G bash sim/render2.sh squishstack 12 50000 > sim/render2.log 2>&1 &'
```

with `sim/render2.sh` exporting the four thread variables and running each shard
under `nice -n 19`. See `CLAUDE.md` → "Running jobs on pasteur".

`limitram` (`/usr/bin/limitram`) is
`systemd-run --user --scope -p MemoryMax="$LIMIT" -- "$@"`, so it caps **memory**
only. It would not, on its own, have prevented this particular incident, whose
binding constraint was runnable threads. The CPU analogue, if it is ever wanted,
is the same mechanism with a different property:

```bash
systemd-run --user --scope -p CPUQuota=800% -- <command>
```
