# signal-native-v2 — Stage 10 results: `benchmark_new` dev scored for the first time

*Results, 2026-09-02. Plan: `plans/have-a-look-of-abundant-spring.md` (kept
under `plans/` as `plans/signal-native-v2-bnew-dev-scoring.md`, see note at
bottom). Predecessor: `plans/signal-native-v2-trash-results.md` (Stage 9).*

---

## Summary

`benchmark_new` dev (242 events, 33% NN truth) has now been scored properly
for the first time — with a real predictor, prior correction, and TTA,
rather than the review-aid `v2_pred`/`v1_pred` columns Stage 9 quoted. Three
findings:

1. **v2 clearly beats v1 on this set.** `e4_union2` (6-member ensemble)
   58.7% 5-class / 58.7% collapsed vs v1's ensemble 41.3% / 48.8%
   (McNemar p<0.0001 5-class, p=0.0032 collapsed). This is **descriptive, not
   a pre-registered arena** — both `v1_pred` and `v2_pred` were used as
   review aids while curating `benchmark_new`, so neither model is scored
   here on data it had zero influence over. The pre-registered comparison
   remains `benchmark_new`'s test half, still unspent.
2. **NN recall is the dominant weakness for every checkpoint set measured**,
   v1 included: at the currently-committed `nn_weight=0.1`, v2's NN recall on
   dev ranges 8.9%–13.9% depending on ensemble. This is the first time NN
   recall has been a measurable number on this branch at all.
3. **`nn_weight=0.1` is measurably too aggressive for this benchmark.** The
   sweep's argmax sits at 0.5 (`geomwin`, 52.5% vs 51.2% uncorrected) and
   0.25 (`union2`, 56.6% vs 55.0% uncorrected) — both push overall accuracy
   *up* by suppressing NN less, the opposite direction from what the old
   no-NN benchmark would show. Neither move clears McNemar significance
   (p=0.30 / p=0.50 at their respective argmax vs `nn_weight=0.1`), so this is
   a measured direction, not a re-committed constant — see Stage 3 below for
   why `nn_weight=0.1` stays the deployed value regardless.

No model, checkpoint, feature vector, or training code changed. One code
change: `dsignal/v1_reference.py` gained a `--curation` mode (Stage 2 below).

---

## Stage 0 — prep and split-key sanity

* `python -m dsignal.verify` on all five `benchmark_new/*.csv`: **970/970**
  regions match (unchanged from Stage 9).
* `benchmark_new/split.tsv` keys `(csv, region)` for all 485 rows match the
  five frozen CSVs exactly (33/23/213/57/159 by class), split 242 dev / 243
  test.
* Regression check: `runs/ensembles/e3_bagged.json` scored on the *old*
  `benchmark/` dev half reproduces **54/70 = 77.1%** exactly, matching the
  historical figure in `CLAUDE.md`/`runs/benchmark_*.log`. Confirms nothing
  in this stage perturbed the existing scoring path.

---

## Stage 1 — v2 scored on `benchmark_new` dev

All three at `nn_weight=0.1` (the currently-deployed value), `--tta-condition-swap`,
`--target-prior nn-weight`:

| predictor | 5-class | collapsed | NN recall | wall clock |
|---|---:|---:|---:|---|
| `e4_union2` (6-member, current dev-best on `benchmark/`) | **142/242 = 58.7%** | 58.7% | 11/79 = 13.9% | 0.145 s/event |
| `e3_bagged` (16-member, test-validated headline on `benchmark/`) | 120/242 = 49.6% | 121/242 = 50.0% | 7/79 = 8.9% | 0.255 s/event |
| `geomwin_iv30_refit` (single model, base recipe) | 117/242 = 48.3% | 48.3% | 7/79 = 8.9% | 0.048 s/event |

Non-NN per-class accuracy is uniformly much stronger than NN across all
three: `e4_union2`'s A3/A5/CSE/IR run 67.9%–87.5%. **NN is the whole gap**
between "58.7% overall" and something closer to the ~75–85% these models hit
on classes they were trained to recognize.

`e4_union2` (dev-tuned for `benchmark/`) clearly outperforms `e3_bagged`
(test-validated on `benchmark/`) here — 58.7% vs 50.0% collapsed, McNemar
p=0.0002. Read this as evidence the two benchmarks reward different things,
not as a reason to prefer `e4_union2` in general: `e3_bagged` is the one with
test-half validation on the *other* benchmark, and `benchmark_new`'s own test
half hasn't been touched by anything.

Per-event TSVs: `runs/bnew_dev_e4_union2.tsv`, `runs/bnew_dev_e3_bagged.tsv`,
`runs/bnew_dev_geomwin.tsv`.

---

## Stage 2 — v1 reference on `benchmark_new` dev

`dsignal/v1_reference.py` gained `--curation <path>` (kept alongside the
existing `--glob` mode, mutually exclusive). It reads truth and `v1_pred`
straight from `benchmark_new/curation.tsv`, maps `truth` back to the
frozen-CSV stem via the inverse of `score_benchmark.CSV_LABEL`
(`CSE -> ES`, everything else identity), drops the 88 `ambiguous` rows, and
keys every row by `(stem, region)` — never region alone, since the branch
has already retracted one number (62/69) for exactly that mistake.

Verification before trusting it:
* Output row count is **exactly 242**, matching the dev half.
* Cross-check against the actual v1 prediction files
  (`runs/v1_uncurated_new/pred_*.csv`): 33/33 sampled rows agree between
  `curation.tsv`'s `v1_pred` and the raw prediction CSVs.

v1 result: 100/242 = 41.3% 5-class, 118/242 = 48.8% collapsed. (v1 predicts
only `A`, never `A3`/`A5`, so the 5-class number understates it by
construction — collapsed is the comparable figure, as always for v1.)

**Contamination caveat, repeated**: `v1_pred`/`v2_pred` were both curation
aids for `benchmark_new`, so this comparison is descriptive. Table:

| comparison | 5-class | collapsed | McNemar p (5-class / collapsed) |
|---|---|---|---|
| v1 vs `e4_union2` | 41.3% vs 58.7% | 48.8% vs 58.7% | 0.0000 / 0.0032 |
| v1 vs `e3_bagged` | 41.3% vs 49.6% | 48.8% vs 50.0% | 0.0225 / 0.7982 |

Per-event TSV: `runs/bnew_dev_v1_ensemble.tsv`.

---

## Stage 3 — `nn_weight` sensitivity curve on `benchmark_new` dev

Ran `dsignal.prior_sweep` (unmodified) against two independent checkpoint
sets, both with `--tta-condition-swap`, `--half dev`:

**`geomwin_iv30_refit`** (trained on `sim/experiment/labels_geomwin.tsv`, 25% NN):

| setting | acc | collapsed | NN pred count | McNemar p vs uncorrected |
|---|---:|---:|---:|---:|
| uncorrected | 51.2% | 51.2% | 55 | 1.000 |
| uniform / nn_weight=1 | 50.4% | 50.4% | 48 | 0.625 |
| nn_weight=0.5 | **52.5%** | **52.5%** | 37 | 0.549 |
| nn_weight=0.25 | 52.1% | 52.1% | 27 | 0.804 |
| **nn_weight=0.1 (current)** | 48.3% | 48.3% | 11 | 0.296 |
| nn_weight=0.05 | 47.9% | 47.9% | 8 | 0.243 |
| nn_weight=0 (oracle) | 45.9% | 45.9% | 0 | 0.060 |

**`scale_union2_n12480`** (trained on `labels_union2.n12480.tsv`, 25% NN):

| setting | acc | collapsed | NN pred count | McNemar p vs uncorrected |
|---|---:|---:|---:|---:|
| uncorrected | 55.0% | 55.0% | 37 | 1.000 |
| uniform / nn_weight=1 | 55.0% | 55.0% | 34 | 1.000 |
| nn_weight=0.5 | 56.2% | 56.2% | 21 | 0.607 |
| nn_weight=0.25 | **56.6%** | **56.6%** | 15 | 0.503 |
| **nn_weight=0.1 (current)** | 53.7% | 53.7% | 8 | 0.690 |
| nn_weight=0.05 | 52.9% | 52.9% | 6 | 0.442 |
| nn_weight=0 (oracle) | 50.8% | 50.8% | 0 | 0.110 |

**Reading, deliberately not a re-commit:**

* Both curves' argmax (0.5 and 0.25 respectively) sits **above**
  `nn_weight=0.1`, in the direction of predicting NN *more* often — the
  opposite of what the old, all-negative `benchmark/` would ever have shown.
  That is the expected shape once a benchmark actually contains NN truth: the
  old sweep's `nn_weight=0` was an oracle bound precisely because there was
  no NN cost to trade against; here there genuinely is one, on both sides.
* **Neither move is significant** at n=242, dev (McNemar p=0.30 / 0.50
  against the current setting) — this is a measured direction on a
  sensitivity curve, not grounds to change the deployed constant from a
  single dev sweep.
* The deployment argument in `prior_sweep.py`'s docstring is unaffected by
  this: deepSpecas classifies candidates from an upstream differential-splicing
  caller, and the training NN fraction (25%) was a design choice for class
  balance, not a deployment estimate — that argument doesn't depend on which
  benchmark happens to be used to read off the curve. **`nn_weight=0.1`
  therefore stays the deployed value.** What's new is that the curve is now
  visible with real NN cost on both sides, and it's a specific, citable
  number the next time this is revisited (e.g. alongside test-half spend).
* This does **not** retro-apply to any `benchmark/`-derived figure (all
  no-NN, all fitted assuming a different regime).

Per-setting TSVs: `runs/bnew_prior_sweep_geomwin_dev.tsv`,
`runs/bnew_prior_sweep_union2_dev.tsv`.

**Honesty caveat #1 (carried from `resume-6`) status: measured, not
resolved.** The constant was settled on a benchmark with zero NN events;
it has now been sensitivity-tested on one that has 79 NN events on dev
alone, and the deployment argument for keeping it still holds. A future
session choosing to move it should do so from this table plus the test-half
spend, not from dev alone.

---

## Stage 4 — clean vs TRASH breakdown, properly scored

New script `benchmark_build/trash_breakdown.py`, joining any set of
already-scored per-event TSVs to `curation.tsv`'s `trash` column on
`(csv, region)` (never region alone). Replaces Stage 9's `v2_pred`-column
figures with real prior-corrected, TTA'd predictions:

| predictor | clean n=172 (5-class / collapsed) | TRASH n=63 (5-class / collapsed) |
|---|---:|---:|
| `e4_union2` | 68.6% / 68.6% | 28.6% / 28.6% |
| `e3_bagged` | 57.0% / 57.6% | 27.0% / 27.0% |
| `geomwin_iv30_refit` | 55.2% / 55.2% | 25.4% / 25.4% |
| v1 ensemble | 51.2% / 60.5% | 17.5% / 17.5% |

(7 of 242 dev events have `trash=NA` and are dropped, matching Stage 9's
172+63=235 dev denominator exactly.)

These numbers are lower across the board than Stage 9's `v2_pred`-derived
70.4%/24.6% and v1's 58.0%/18.3% (those were computed over the full 485-event
unambiguous pool, not dev alone, and from the raw consensus-review prediction
column rather than a prior-corrected, TTA'd scoring run) — but the **shape is
the same**: every predictor does far worse on TRASH than on clean, by roughly
2.0–2.4x depending on model. This corroborates Stage 9's central finding
(TRASH is a real, large effect on accuracy) with a properly-scored number
rather than a review artifact, on the dev half specifically.

Per-model summary TSV: `runs/bnew_dev_trash_breakdown.tsv`.

---

## What is new in the code

| file | what |
|---|---|
| `dsignal/v1_reference.py` | new `--curation` mode + `build_rows_from_curation`; `--glob` mode unchanged (byte-identical output for existing callers) |
| `benchmark_build/trash_breakdown.py` | (new) clean-vs-TRASH accuracy join for any set of already-scored per-event TSVs, keyed by `(csv, region)` |

No `dsignal/features.py`, `dsignal/dataset.py`, `dsignal/model.py`,
`dsignal/predict.py`, `dsignal/train.py`, `dsignal/score_benchmark.py`, or
`dsignal/prior_sweep.py` changed — every number in this stage came from
running existing tools against `benchmark_new`.

---

## Open honesty caveats (carried forward, none newly resolved)

1. **`nn_weight=0.1`**: sensitivity-tested this stage (Stage 3), not
   re-derived or changed. Still the deployed value; the deployment argument
   for it is unaffected by having NN truth to read a curve against.
2. **`DPSI_BETA`/`PSI_LEVEL_BETA`** still fitted on `benchmark/`'s 140 events.
   Untouched.
3. **`benchmark_new`'s test half (243 events) is completely unspent.**
   Nothing in this stage read it. `benchmark/`'s own test-half spend
   (Stage 8 Track C) remains separately owed and untouched.
4. **Stage 2's sim-vs-real TRASH gap** (`intron_cov_frac` 3.3x,
   `novel_junction_rate` 2.3x short of `real-TRASH`) is unchanged — this
   stage did no re-simulation, per its own explicit non-goal.
5. **New**: every v1-vs-v2 number in this stage is a descriptive comparison
   on a benchmark whose curation used both models' predictions as review
   aids, not a pre-registered arena. Treat it as suggestive (and consistent
   with `benchmark/`'s own numbers in direction), not as a fresh head-to-head
   result. The pre-registered `benchmark_new` test-half spend is still owed
   and would be the honest one-shot comparison.

---

## Artifacts

| path | what |
|---|---|
| `runs/bnew_dev_{e4_union2,e3_bagged,geomwin,v1_ensemble}.tsv` | per-event scored predictions, dev half |
| `runs/bnew_dev_{e4_union2,e3_bagged,geomwin}.log` | `score_benchmark` full output |
| `runs/bnew_prior_sweep_{geomwin,union2}_dev.tsv` | full `nn_weight` sensitivity tables |
| `runs/bnew_dev_trash_breakdown.tsv` | clean vs TRASH accuracy, all four predictors |
| `runs/bench_old_e3_dev_regression.log` | Stage 0 regression check (54/70 reproduced) |
| `benchmark_build/trash_breakdown.py` | new clean-vs-TRASH join script |
| `dsignal/v1_reference.py` | `--curation` mode added |

## Note on the plan file name

The plan behind this stage was written and approved under the session's
auto-generated path `~/.claude/plans/have-a-look-of-abundant-spring.md`
(outside the repo). Its content is reproduced in spirit here; no separate
`plans/signal-native-v2-bnew-dev-scoring.md` plan file was committed to the
repo since the plan itself needed no code beyond what's listed above.
