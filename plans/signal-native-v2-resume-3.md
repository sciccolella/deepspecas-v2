# signal-native-v2 — START HERE (2026-08-18)

Resume document. Branch `signal-native-v2`, clean tree, **nothing pushed** (per
repo convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-2.md`, which remains accurate for
everything up to Stage 1 and holds the detail this file compresses. The plan
both were working through is `plans/signal-native-v2-docs-and-model.md`; **Parts
A and B are now complete, Stages 0–3 included.**

---

## The one thing to know

**v2 is already performing at the level of a single v1 representation. The
headline gap this branch has been chasing is mostly *ensembling*.**

The 87.9% v1 number everyone has been measured against is the **six-net
ensemble**. Trained on its own data, a *single* v1 representation
(`squishstack`) scores **75.7%** on the same 140 events. v2's current best is
**75.7% dev / 82.9% test**. There is no representation gap left to explain.

That conclusion comes from Stage 2, which swapped only the training data:

| representation | trained on | dev | test | all 140 |
|---|---|---|---|---|
| v1 images + ResNet50 | **v2's simulation** | 75.7% | 72.9% | 104/140 = 74.3% |
| v1 images + ResNet50 | v1's own simulation | 80.0% | 71.4% | 106/140 = 75.7% |
| v2 signal-native | v2's simulation | 75.7% | **82.9%** | — |

Two events separate the first two rows: **v2's simulation trains v1's
representation as well as v1's own data does.** The simulation is not the
bottleneck and the signal-native representation is not fragile.

## The headline configuration

**58/70 = 82.9% collapsed on the held-out test half**, scored once, every choice
made on dev (which gives 53/70 = 75.7%):

```bash
python -m dsignal.score_benchmark --net runs/geomwin_iv30_refit/fold*.pt \
  --half test --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
  --train-labels sim/experiment/labels_geomwin.tsv
```

Trained with:

```bash
python -m dsignal.train --npz-dir sim/experiment/npz_geomwin \
  --labels sim/experiment/labels_geomwin.tsv --augment all \
  --feature-subset no-geometry --refit --inner-val-frac 0.30 \
  --out-dir runs/geomwin_iv30_refit --json-out runs/geomwin_iv30_refit.json
```

**Do not quote 78.6% any more.** Its epoch was chosen by peeking at the fold it
then reported; the honest re-baseline of that same model is 49/70 on dev.

## Two protocol changes that invalidate older numbers

1. **Epoch selection** (`dsignal/train.py`). Until 2026-08-17 the best epoch was
   chosen on the held-out fold being reported. `--select-on inner` is now the
   default. **Every sim-CV number predating that reads ~0.013 high**;
   `--select-on held-out` reproduces one. Surprisingly this also cost ~10 points
   of transfer, and `--refit` proved it was the *epoch choice*, not the 15% of
   data (49/70 vs 48/70, p = 1.000). A larger inner split recovers part of it.
2. **The benchmark is split** (`benchmark/split.tsv`, committed). 70 dev / 70
   test, stratified by class, cluster-disjoint at 50 kb. Everything tuned is read
   off **dev**; test is scored once per milestone. `--half {dev,test,both}` on
   `score_benchmark` and `prior_sweep`;
   `python -m dsignal.benchmark_split compare A.tsv B.tsv --half dev` does the
   paired McNemar. Cost: n=70 needs an 8–10 event swing where n=140 needed 5–6.

## The scoreboard, all on dev unless stated

| change | set | sim-CV | dev collapsed | vs its control |
|---|---|---|---|---|
| *(control)* honest protocol, 4,800 | 4,800 | 0.853 | 49/70 = 70.0% | — |
| `--inner-val-frac 0.30` | 4,800 | 0.862 | 51/70 = 72.9% | p = 0.727 |
| + condition-swap TTA | — | — | 51/70 = 72.9% | p = 0.625 |
| + pre-registered `nn_weight` 0.1 | — | — | 51/70 = 72.9% | p = 0.500 |
| **all three stacked (headline)** | 4,800 | 0.862 | **53/70 = 75.7%** | p = 0.344 |
| `--feature-subset all` | 4,800 | 0.873 | 52/70 = 74.3% | p = 0.549 |
| `--two-head` | 4,800 | 0.858 | 47/70 = 67.1% | p = 0.727 |
| `--two-head` + TTA + prior | 4,800 | 0.858 | 53/70 = 75.7% | p = 1.000 |
| *(control)* 960 events, same protocol | 960 | 0.756 | 48/70 = 68.6% | — |
| Stage 3 re-simulation | 960 | 0.642 | 42/70 = 60.0% | p = 0.308 |

## What is now settled, and what it means for the next session

**Realism is 0 for 3.** Depth invariance (null), effect-size realism (null), and
now the Stage 3 multi-axis batch — per-class positives floored at twice the
measured null, `nn_hard` nulls at the measured replicate-noise scale, and 6.09
co-occurring events perturbed per window, all together — which came back
*worse* (sim-CV 0.756 → 0.642, dev 68.6% → 60.0%). Three axes moved at once, so
it cannot apportion blame, but that was exactly the design: it was the honest
test of "realism only pays when several axes move together", and the answer is
no.

**Model surgery is 0 for 3 too.** `--feature-subset all`, the two-head
factorization, and (earlier) the depth-normalisation variants. The two-head
model is worth one note: it does not classify better, but it makes the NN
decision a single scalar the prior correction can move, recovering 8 events
where the single head recovers 4.

**So the mechanism the evidence actually supports is ensembling.** v1 gets ~12
points from six representations voting; v2 has never been given an analogue.
Fold-averaging and condition-swap TTA are the two it has, and TTA was worth +2.
A real experiment would train v2 on genuinely different *views* — feature
subsets, window paddings, depth normalisations — and combine them. Unlike the
last six interventions, that has a mechanism behind it the data supports.

Two smaller things worth doing first because they are nearly free:

* `--avg-last-k` (implemented, never run at scale) — tail-averaging the refit
  run's weights, the other cheap answer to the epoch-choice noise Stage 1.2
  exposed;
* the `--inner-val-frac` sweep only has 0.15 and 0.30; 0.4–0.5 is untested and
  0.30 was worth +2 events over 0.15.

**Not recommended:** more scale-up, more noise layers, Phase-4 fine-tuning, more
single-axis realism knobs, or another head architecture.

## Open honesty caveats

1. **The NN operating point — settled.** The deployment argument is written into
   `dsignal/prior_sweep.py`'s docstring *before* any number was read, and
   `nn-weight 0.1` was pre-registered from it, then scored on dev only.
2. **`DPSI_BETA`/`PSI_LEVEL_BETA` are still fitted on all 140 benchmark
   events.** The Stage 3 batch deliberately avoided `--psi-dist beta` entirely
   and used only quantities measured from the simulation itself, so it is clean.
   But if beta is ever used again, it must be refit on **dev only** — and note
   the per-event benchmark dPSI values were never stored, so that refit means
   re-measuring them (the matching logic lives in `sim/psi_recovery_check.py`).

## What is new in the code

| file | what |
|---|---|
| `dsignal/benchmark_split.py` | `make` builds `benchmark/split.tsv`; `compare` is the paired McNemar between two scored per-event TSVs |
| `dsignal/train.py` | `--select-on {inner,held-out}`, `--inner-val-frac`, `--refit`, `--avg-last-k`, `--two-head`, `--diff-lambda` |
| `dsignal/model.py` | two-head factorization + `compose_two_head` (returns 5-class **log probs**, so `softmax(log p) == p` keeps every caller working) |
| `dsignal/dataset.py` | `aux_targets` — `nn_hard` negatives supervise the type head too, since their `event_id` carries the SUPPA type |
| `dsignal/predict.py` | `--tta-condition-swap` (views averaged *within* each fold) |
| `dsignal/score_benchmark.py` | `--half`, `--split`, `--target-prior nn-weight` + `--nn-weight` |
| `dsignal/prior_sweep.py` | `--half`, `--tta-condition-swap`, and the pre-registered deployment argument in its docstring |
| `sim/null_dpsi_scale.py` | measures the null \|dPSI\| from the simulation's own replicate noise (median 0.0505, p90 0.133, n=399) |
| `sim/select_events.py` | `--psi-dist paired`, `--null-dpsi-sigma`, `PAIRED_DPSI_RANGE`, `NULL_DPSI_SIGMA` |
| `sim/abundance.py`, `sim/build_experiment.py` | `--cooc-frac`, `--cooc-dpsi` — the gene's other catalogued events also change between conditions |
| `sim/run_batch.sh` | `NULL_DPSI_SIGMA`, `COOC_FRAC` env knobs, both defaulting to pre-Stage-3 behaviour |
| `sim/render2.sh`, `sim/render_resume.sh`, `sim/build_render_csv.py`, `sim/build_v1_trainset.py` | the Stage 2 render pipeline |
| `crash_pasteur.md` | how pasteur was made unresponsive, exactly, and the three failure modes found |

## Artifacts

| path | what |
|---|---|
| `benchmark/split.tsv` | the committed 70/70 dev/test split |
| `runs/geomwin_iv30_refit/` | **the current best checkpoints** |
| `runs/benchmark_events_stack_{dev,test}.tsv` | headline per-event results |
| `runs/geomwin_{innerval,innerval_refit,allfeat,twohead,b1_refit}/` | the Stage 1 comparisons |
| `runs/stage3/`, `sim/experiment/npz_stage3/`, `labels_stage3.tsv` | the Stage 3 batch (960 events, gene-disjoint from b1–b5) |
| `runs/v1_crossover/` | the crossover: 18,120 images, `train.csv`/`val.csv`, `rs50.pth`, `pred/` |
| `runs/v1_pretrained/pred/` | the matched control, v1's own pretrained `squishstack` net |
| `runs/prior_sweep_{dev,iv30_tta_dev,twohead_dev}.tsv` | the prior curves |
| `runs/noise_audit_sim_b7stage3.tsv` | the Stage 3 acceptance audit |
| pasteur `sim/batches/b7stage3/`, `sim/render_squishstack/` | the batch and the render |

## Running jobs on pasteur — read `CLAUDE.md` first

It is a **shared** machine. On 2026-08-17 an unthrottled 32-way render took the
load average to 230 on 64 cores and locked everyone out of login shells for ~40
minutes. `crash_pasteur.md` is the exact reproduction. The rules now in
`CLAUDE.md`: `limitram <LIMIT> <cmd>` always, all four thread variables set to 1
(numpy *and* torch size pools to the core count — processes still run at ~174%
CPU with `OMP_NUM_THREADS=1`), `nice -n 19`, ≤12 shards, check `/proc/loadavg`
before and after, and sample throughput twice rather than assuming it.

Two more traps from this session:

* **`limitram` jobs die when the ssh session ends.** It is
  `systemd-run --user --scope`, and pasteur has `Linger=no`, so the user manager
  exits with the last session and takes the scope with it — `nohup` does not
  help. Hold the connection, or ask about `loginctl enable-linger`.
* **`pkill -f plot2s.py` kills your own ssh session**, because the pattern is in
  the command line carrying it. Use `pkill -f "[p]lot2s"`.

## Traps hit this session — do not repeat

1. **A null drawn twice.** `_null_dpsi` was called once inside the `psi1`
   expression and again inside `psi2`, so the realised effect was two
   independent draws — median 0.036 instead of the measured 0.0505. Caught by
   inspecting `design.tsv` *before* polyester and STAR ran, which is the cheap
   place to catch it. **Check a design against its intent before spending hours
   on it.**
2. **`val_ds` referenced `test_ds` before assignment**, breaking
   `--select-on held-out` — the flag whose only job is reproducing old numbers.
   The smoke test in that same commit ran before the line that broke it.
3. **Contiguous shards + a fat tail = a stalled render.** Window widths run
   median 1.9 kb to max 347 kb; `split -n l/N` gave one shard all the giants.
   Sort by width and `split -n r/N`.
4. **Narrow windows crash `plot2s`** (`Image.resize`, zero-column image), and one
   bad event kills the shard and everything queued behind it.
   `sim/render_resume.sh` steps over them rather than dropping the narrow tail,
   which would have cost 689 events biased toward A3/A5/RI.

## Conventions

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar; 4–6 events is noise at n=140, 8–10
  at n=70.
- Never delete a measured number from a docstring or plan doc — retractions go
  next to the number they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/rendering on pasteur.
