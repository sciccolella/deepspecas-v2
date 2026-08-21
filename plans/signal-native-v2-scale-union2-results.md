# signal-native-v2 — Stage 7 (`npz_union2`) results

*Implements `plans/signal-native-v2-scale-union2.md`. Written 2026-08-21, same
session that ran it end to end (Stages 7.0–7.4). Dev only throughout — test is
spent (Stage 6, `plans/signal-native-v2-ensemble-results.md`) and this plan
makes no v1-vs-v2 claim.*

---

## Headline

**Scale alone saturates past ~8,640 events; view diversity does not.** The
three-point curve (4,800 / 8,640 / 12,480, nested gene-disjoint subsamples of
one nine-batch-plus-four-new pool) rose sharply then flattened: 51/70 (72.9%)
→ 57/70 (81.4%) → 56/70 (80.0%) dev collapsed. The pre-registered read fires
the middle band — **flat-to-marginal, scale is saturating** — not the ≥57/70
"still paying" case that would have triggered a fresh-benchmark follow-up
plan. In-sim macro-F1 kept climbing monotonically the whole time (0.868 →
0.896 → 0.906), so the ceiling is specific to zero-shot transfer, not to the
model's ability to fit more data.

**View diversity, reapplied at 1.44× the data, did not saturate.** Six
members from Stage 2's exact proven-diverse family list, trained on the full
12,480-event pool and ensembled equal-weight/`mean_prob`, reached **60/70 =
85.7% (5-class) / 61/70 = 87.1% (collapsed) dev** — 4–5 events above the best
single Stage 7.3 run, 0 losses in either direction, but not significant at
n=70 (McNemar p=0.29 5-class, p=0.13 collapsed; 7–8 discordant, need 8–10).
The two levers do not appear to compete for the same headroom.

## Stage-by-stage

**Stage 7.0 (utility, local).** `sim/merge_batches.py --subsample-events N`
added, reusing the existing `_drop_genes_to` gene-preserving downsample and
the `--from-labels` path already built for `--nn-frac`. Verified against
`labels_union.tsv`: repeated calls at the same `--seed` produce exactly
nested gene sets (the shuffle order depends only on the seed and the full
gene set, not on `keep_n`), class fractions within ~1pp.

**Stage 7.1 (four new batches, pasteur).** `b12`–`b15`, seeds 12–15, same
`PSI_DIST=uniform` Stage-3 settings, each excluding every gene from
`batches_geomwin`/`batches` b1–5, `batches_beta/b6beta`, `batches_bag2`
b8–11, and the earlier members of this batch's own chain (9,024 → 11,904
genes excluded across the run). Every batch's design was inspected before
any polyester/STAR time was spent: 0 gene overlap in every `select_events`
log, all four positive classes hit 180, `nn_hard` rows `psi1==psi2` exactly.
One operational hiccup — `sim/align_reads.sh` lost its executable bit in the
rsync from local to pasteur (`chmod +x` was run before this session's rsync
of `dsignal/`+`sim/*.py`+`.sh` but the mode bit didn't survive); b12 failed
at the STAR step with `Permission denied`, `chmod +x` fixed it, and
`run_batch.sh`'s per-step existence guards meant the retry skipped straight
back to STAR rather than re-doing the completed steps.

Post-simulation gates, every batch: `sim.psi_recovery_check` (run inline by
`run_batch.sh`), `sim.width_leak_check` (0.296–0.356 acc / 0.276–0.334
macro-F1, consistent with the accepted `bag2` batches' 0.330/0.287),
`dsignal.audit_noise compare` (same off-by-2x quantity set and directions as
`bag2` in every batch — `intron_cov_frac`, `se_minor_count`, `cov_mean`,
`a3_minor_count`, `top_junction_count`, `dup_rate`, `antisense_frac` all
over-inject; `novel_junction_rate`/`novel_junction_read_frac` under-inject;
no new divergence introduced). 3,840 genes consumed, 960/batch; the RI gene
pool (908 unused going in) is now at **~188 genes remaining** — this lever
cannot be pulled a third time without changing the one-gene-per-event
design (see the honesty ledger).

**Stage 7.2 (merge, local).** 13 batches (`batches_geomwin` b1–5,
`batches_bag2` b8–11, `batches_union2` b12–15) merged into `npz_union2` /
`labels_union2.tsv`: **12,480 records, 12,480 distinct genes, 0 `ix`
collisions** — the one-gene-per-event invariant holds across the full merge,
verified directly rather than assumed. `--subsample-events 4800 8640 12480`
at `--seed 0` produced the three nested pools; nesting (n4800 ⊂ n8640 ⊂
n12480, by gene set) and class-fraction stability (worst deviation ~1.2pp,
n4800's CSE at 17.6% vs the pool's 18.8%) were both checked, not assumed.

**Stage 7.3 (the scale curve, local `DSnet`).** Three identical-config runs
(`--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30`),
differing only in `--labels`:

| n | sim macro-F1 | dev, 5-class = collapsed |
|---|---|---|
| 4,800 | 0.868 | 51/70 = 72.9% |
| 8,640 | 0.896 | 57/70 = 81.4% |
| 12,480 | 0.906 | 56/70 = 80.0% |

McNemar on every adjacent pair (n=70, 8–10 discordant needed):

| pair | discordant (base-only / new-only) | p |
|---|---|---|
| n4800 → n8640 | 1 / 7 | 0.0703 |
| n8640 → n12480 | 5 / 4 | 1.0000 |
| n4800 → n12480 | 3 / 8 | 0.2266 |

The 4800→8640 jump trends positive but isn't itself significant; 8640→12480
shows no signal at all. **Pre-registered read: n12480 = 56/70 falls in the
54–56/70 band → flat-to-marginal, scale is saturating.** Not the ≥57/70 case
(no fresh-benchmark follow-up triggered) and not the ≤53/70 case either — the
`npz_union` diagnostic's 78.6% dev (resume-4) stands as a real, if partial
and now-plateaued, effect. Corroborating points from earlier sessions,
different gene sets so context rather than curve: `geomwin_iv30_refit` 53/70
at 4.8k, `ens_bag_union` 55/70 at 8.6k — consistent with this curve's shape.

**Stage 7.4 (conditional ensemble, local `DSnet`).** Triggered because 7.3
was not the ≤53/70 "noise" case. Six members on `labels_union2.n12480.tsv`,
reusing Stage 2's exact family list — no new axes, per Stage 4's breadth
null: baseline (= `scale_union2_n12480`), `--feature-subset all`,
`--two-head`, `--depth-norm scale`, `--augment flip condition_swap
subsample`, and a 15%-NN labels variant. All six cluster tightly in sim
macro-F1 (0.904–0.914) — a genuinely diverse-view ensemble, no member
dominating. Equal weight, `mean_prob` (`runs/ensembles/e4_union2.json`,
committed):

| | dev, 5-class | dev, collapsed |
|---|---|---|
| best single 7.3 run (n12480) | 56/70 = 80.0% | 56/70 = 80.0% |
| e4_union2 (6-member ensemble) | **60/70 = 85.7%** | **61/70 = 87.1%** |

McNemar vs the best single run: 5-class 8 discordant (2 base-only / 6
new-only) p=0.2891; collapsed 7 discordant (1/6) p=0.1250. Unanimous
direction, 0 losses either metric, not significant at this n. **The
view-diversity lift from Stage 2 (npz_geomwin, +1 dev event, free) survives
at 1.44× the data and does not appear to have been absorbed by scale — if
anything it looks larger here, though this session cannot prove that
statistically.** No v1-vs-v2 claim is made from this number; test is spent
and this plan is dev-only throughout.

## What this changes about the target

Before this plan: `npz_union`'s 78.6% dev (resume-4) was a single Stage-3
diagnostic run, unreplicated, with no signal on whether it would hold at
larger scale or whether it was itself noise. After it: the scale axis has
been measured properly (a three-point curve, not one point) and the honest
answer is that it **plateaus**, not that it keeps paying — 56–57/70 is where
single-model scale on this simulation design tops out for now. The genuinely
new finding is that **ensembling and scale stack**: 87.1% dev collapsed is
the best number this branch has produced on any half, dev or test, single-
model or ensemble, image-free or hybrid-ceiling — but it rests on n=70 with
discordant counts just under the significance threshold, so it is reported
as a real, unproven-at-this-n effect, not a settled result.

## Honesty ledger

* `DPSI_BETA`/`PSI_LEVEL_BETA` remain fitted on all 140 benchmark events;
  `b12`–`b15` used `PSI_DIST=uniform` throughout, so they stay clean —
  unchanged from every prior session.
* The NN operating point (`nn_weight=0.1`) was reused unchanged, not
  re-tuned.
* The scale-curve read bands and the Stage 7.4 trigger condition were
  pre-registered in the plan before any dev number was read.
* **The RI gene pool is now ~188 genes from exhausted** (908 unused before
  this plan, 720 consumed by `b12`–`b15`). A fifth gene-disjoint scale-up
  batch is not available under the current one-gene-per-event design without
  either relaxing that invariant or accepting a smaller, RI-only batch.
* e4_union2's 87.1% dev is **not** eligible for a test spend on the current
  benchmark split — test was spent once already (Stage 6) and this plan's
  explicit scope is dev-only. If this number is pursued further, it needs a
  fresh held-out half, not the existing test 70.
* This plan makes no v1-vs-v2 claim anywhere, including in this document —
  87.1% dev collapsed is compared only against this plan's own single-run
  baseline (56/70 = 80.0%), never against v1's dev figure, even though the
  two numbers happen to be close.

## Critical files touched

`sim/merge_batches.py` (`--subsample-events`, refactored `_write_labels_subset`
helper), `sim/experiment/batches_union2/{b12,b13,b14,b15}` (new, `npz/` +
`labels.tsv` + `design.tsv` + `psi_recovery.tsv` + `logs/` only — no BAMs or
FASTA), `sim/experiment/npz_union2/`, `labels_union2{,.n4800,.n8640,.n12480,
.n12480.nn15}.tsv` (new), `runs/scale_union2_n{4800,8640,12480}{,.json,.log}`,
`runs/ens_union2_{allfeat,twohead,depthscale,augsub,nn15}{,.json,.log}`,
`runs/ensembles/e4_union2.json`, `runs/benchmark_events_{scale_n4800,
scale_n8640,scale_n12480,e4_union2}_dev.tsv`, `runs/noise_audit_sim_b{12,13,
14,15}.tsv`.

No changes to `dsignal/` — `train.py`, `score_benchmark.py`, `ensemble.py`,
`benchmark_split.py` and `baseline_gbm.py`'s shared harness already did
everything this plan needed. `dsignal.verify_features` passed as a
regression check both on pasteur (post-rsync) and locally.
