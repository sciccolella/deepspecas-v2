# signal-native-v2 — START HERE (2026-08-21)

Resume document. Branch `signal-native-v2`, clean tree (after this commit),
**nothing pushed** (per repo convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-4.md` as the entry point.
`plans/signal-native-v2-scale-union2.md` (the plan) and
`plans/signal-native-v2-scale-union2-results.md` (the full stage-by-stage
record) are both **fully implemented** — read the results doc for detail
this file compresses. resume-4 and `plans/signal-native-v2-ensemble-results.md`
still hold everything about the ensemble plan (Stages 0–6) and the v1
comparison; that work is unchanged by this session.

---

## The one thing to know

**Scale, on its own, has a ceiling — and this session found it. View
diversity does not share that ceiling, and the two levers stack.**

resume-4 closed with "scale `npz_union`-style data further" as the clearest
untried lever, off one Stage-3 diagnostic run at 78.6% dev. This session
measured it properly — a three-point curve, not one point — and the honest
answer is smaller than hoped on the scale axis alone, but the follow-up
ensemble check landed the best number this branch has produced on any half:

| n (events) | sim macro-F1 | dev, collapsed |
|---|---|---|
| 4,800 | 0.868 | 51/70 = 72.9% |
| 8,640 | 0.896 | 57/70 = 81.4% |
| 12,480 | 0.906 | 56/70 = 80.0% |
| **12,480, 6-member ensemble (`e4_union2`)** | 0.904–0.914 per member | **61/70 = 87.1%** |

Single-model transfer to the curated benchmark **plateaus past ~8,640
events** even though in-sim macro-F1 keeps climbing — the ceiling is
specific to zero-shot transfer. Ensembling six diverse views of the largest
pool (Stage 2's exact family list, no new axes) reached 87.1% dev collapsed,
+7 events over the best single run, unanimous direction, but **not
significant at n=70** (McNemar p=0.13, 7 discordant, need 8–10).

## No v1-vs-v2 claim — still true, deliberately

Test was spent in the ensemble plan (Stage 6: 60/70 dev, 63/70 test for v1;
E3's 54/70 dev, 60/70 test for v2; McNemar p=0.549, no parity, no win). This
session's plan **explicitly scoped out any further test spend or v1
comparison** — everything above is read against this plan's own dev-only
baseline, not against v1's numbers, even where they land close together
(87.1% dev collapsed vs v1's own 85.7% dev collapsed — noted, not claimed).
If e4_union2 is pursued further, the next step is a **fresh held-out
benchmark half**, not the existing test 70, which stays spent.

## What is now settled, and what it means for the next session

**Single-model scale on this simulation design tops out around 56–57/70
dev.** Three adjacent-pair McNemar tests, none individually significant, but
the shape (sharp rise then flat) is consistent and the pre-registered read
(56/70 falls in the 54–56 "flat-to-marginal" band) fired as designed before
any number was seen. Training more single models on more `npz_union`-style
gene-disjoint batches is not expected to move this further without a design
change — see the RI exhaustion note below, which forecloses that path
mechanically regardless.

**View diversity and scale do not compete for the same headroom.** Stage 2's
ensemble lift (npz_geomwin, +1 dev event, free) not only survived at 1.44×
the data, it grew (+4-5 events at n=12,480 vs the best single run) — though
this session's n=70 sample can't prove the growth is real rather than noise.
This is the first evidence in either plan that the two axes are not simply
trading off against each other.

**The RI gene pool is nearly exhausted.** 908 unused RI genes going into this
session, 720 consumed by the four new batches, leaving **~188**. A fifth
gene-disjoint scale-up batch under the current one-gene-per-event design is
not available — this forecloses "scale further" as a lever without either
relaxing that invariant (multiple events per gene) or accepting an
RI-starved batch that can't hit 180/class.

**What this suggests for a next plan, if there is one:**

* If e4_union2's 87.1% dev is worth chasing to a real test: curate a fresh
  held-out benchmark half (the existing 140-event set's test half stays
  spent) and pre-register the parity/win criterion the way the ensemble plan
  did, before reading any number.
* Do **not** scale `npz_union2` further with more gene-disjoint batches —
  the RI pool forecloses it, and the curve itself shows it wasn't paying
  past 8,640 anyway.
* View-diversity ensembling at larger scale is the one axis this session
  found still moving. If pursued, it should be measured with a larger
  member count or a genuinely new view family (not reused from Stage 2) to
  see if the growth continues or was itself a small-n artifact.
* Relaxing the one-gene-per-event design (multiple simulated events per
  gene, with within-gene leakage handled some other way) is the only path
  back to more RI data if scale is ever revisited.

## Open honesty caveats (carried forward, still open)

1. **The NN operating point — settled**, unchanged this session
   (`nn_weight=0.1`, pre-registered in `dsignal/prior_sweep.py`'s docstring).
2. **`DPSI_BETA`/`PSI_LEVEL_BETA` are still fitted on all 140 benchmark
   events.** `b12`–`b15` used `PSI_DIST=uniform` throughout, so they stay
   clean; unchanged from resume-4.
3. **e4_union2's 87.1% dev has had no exposure to test.** If it or any
   successor is trained again or scaled up, it becomes eligible for a
   Stage-6-style one-shot test spend on a **fresh** held-out half only — the
   existing test 70 is spent and must not be re-read.
4. **The RI gene pool sits at ~188 genes**, down from 908 at the start of
   this session. Any future batch plan must check this budget before
   committing to a batch count.

## What is new in the code

| file | what |
|---|---|
| `sim/merge_batches.py` | `--subsample-events N` (repeatable), reusing `_drop_genes_to`; `_write_labels_subset` factored out of `resample_to_nn_frac` |

No other `dsignal/`/`sim/` module changed this session — `train.py`,
`score_benchmark.py`, `ensemble.py`, `benchmark_split.py`,
`baseline_gbm.py`'s shared harness, `run_batch.sh` all did everything this
plan needed unmodified.

## Artifacts

| path | what |
|---|---|
| `plans/signal-native-v2-scale-union2-results.md` | the full stage-by-stage record |
| `runs/ensembles/e4_union2.json` | the Stage 7.4 ensemble spec, committed and reproducible |
| `sim/experiment/batches_union2/{b12,b13,b14,b15}/` | the four new gene-disjoint batches (`npz/`, `labels.tsv`, `design.tsv`, `psi_recovery.tsv`, `logs/` only) |
| `sim/experiment/npz_union2/`, `labels_union2{,.n4800,.n8640,.n12480,.n12480.nn15}.tsv` | the merged pool and nested scale-curve subsamples |
| `runs/scale_union2_n{4800,8640,12480}/` | the three scale-curve checkpoints |
| `runs/ens_union2_{allfeat,twohead,depthscale,augsub,nn15}/` | the five additional e4_union2 members (baseline is `scale_union2_n12480`) |
| `runs/benchmark_events_{scale_n4800,scale_n8640,scale_n12480,e4_union2}_dev.tsv` | per-event scoring, dev only |
| `runs/noise_audit_sim_b{12,13,14,15}.tsv` | per-batch noise-calibration measurements |

## Running jobs on pasteur — read `CLAUDE.md` first

Unchanged from resume-4: `limitram`, all four thread env vars at 1, `nice
-19`, hold the ssh session (a backgrounded Bash tool call that keeps the
local ssh process alive; `nohup cmd &` inside the ssh command does not
reliably survive `Linger=no`). This session's four batches ran at pasteur
load 0.8–1.6/64 throughout, no throttling needed. One operational note for
next time: **`chmod +x` on shell scripts does not always survive an rsync
push** — verify executable bits on `sim/*.sh` after syncing `sim/` to
pasteur, not just file content.

## Conventions

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar; 4–6 events is noise at n=140,
  8–10 at n=70.
- Never delete a measured number from a docstring or plan doc — retractions
  go next to the number they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/rendering on
  pasteur.
