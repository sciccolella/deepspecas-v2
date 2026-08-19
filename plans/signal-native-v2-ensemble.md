# signal-native-v2 — Stage 4: build the v2 ensemble

*Plan, agreed 2026-08-19. Per `CLAUDE.md`, a separate session implements it.*
*Successor to `plans/signal-native-v2-resume-3.md`, whose closing
recommendation this executes.*

---

## Context

`plans/signal-native-v2-resume-3.md` closes with one recommendation and the
evidence for it: realism is 0 for 3, model surgery is 0 for 3, and the only
mechanism the data still supports is **ensembling** — v1 gets its headline
number from six representations voting, and v2 has never been given an
analogue. This plan builds that analogue.

**First, the target was wrong, and the corrected one is much closer.** Every
document on this branch compares against "v1 = 123/140 = 87.9%". That figure is
over all 140 events, while v2 is now scored on the 70/70 dev/test split. Scoring
the stored v1 six-net ensemble predictions (`runs/v1/pred_{A3,A5,ES,IR}.csv`,
produced by `runs/v1_repro.log`) against `benchmark/split.tsv` gives the numbers
that are actually comparable:

| model | dev | test | all |
|---|---|---|---|
| **v1 six-net ensemble** | **60/70 = 85.7%** | **62/69 = 89.9%** | 122/139 = 87.8% |
| v1 `squishstack` pretrained, single net | 56/70 = 80.0% | 49/69 = 71.0% | 105/139 = 75.5% |
| **v2 headline** (`geomwin_iv30_refit` + TTA + `nn_weight` 0.1) | 53/70 = 75.7% | 58/70 = 82.9% | 110/139 = 79.1% |

(139 not 140: region `14:24115727-24116971` appears in both `A3.csv` and
`A5.csv`, and the v1 prediction files are keyed by region alone. The v2 TSVs
key on `(csv, region)` and keep both, which is why v2 shows 70 on test and v1
shows 69. `dsignal/benchmark_split.compare` already keys on `(csv, region)` —
the reference file built in Stage 0.1 must do the same.)

**So the gap is 7 events on dev and 4 on test, not 17.**

### What dev/test means here, and which comparison is the real one

The split is about **selection exposure, not training exposure**. Neither model
ever trains on benchmark events. What `benchmark/split.tsv` protects against is
that essentially every v2 design choice on this branch — the window sampler,
`nn_weight`, `--inner-val-frac`, the augmentation set, the epoch-selection
protocol — was made by *reading* benchmark accuracy. Reading a 140-event set
dozens of times fits it even with no gradient touching it. The split freezes 70
events no choice has been read off.

Two consequences that decide how this plan is scored:

1. **For v1 the split is meaningless.** No v1 hyperparameter was chosen by
   reading these events, so v1's unbiased figure is the whole benchmark,
   **122/139 = 87.8%**. Its 85.7% dev / 89.9% test breakdown is sampling noise
   across 70-event halves, not two different quantities.
2. **For v2 only the test half is unbiased.** 53/70 dev has had many choices
   read off it; 58/70 test has had one.

So the honest head-to-head is **paired and restricted to test**:

| comparison | v1-only-right | v2-only-right | McNemar p |
|---|---|---|---|
| dev (contaminated on v2's side — do not headline) | 12 | 5 | 0.143 |
| **test (the real one)** | **9** | **4** | **0.267** |
| pooled 139 (contaminated on v2's side) | 21 | 9 | 0.043 |

v2 is ~4–5 events behind on the only half that counts, and that gap is not
significant at n=70 — but "not significant" at this size is weak evidence, not
parity. Beating v1 outright needs a real margin, and the only way to claim it is
a **pre-registered ensemble scored once on test**. Every stage below therefore
iterates on dev and spends test exactly once, in Stage 6.

Two further measurements motivate the design:

* **Complementarity is large.** Oracle union of v2 with the v1 ensemble is
  **131/139**; with the single v1 `squishstack` net, 127/139. v2 gets 9 events
  right that the whole v1 ensemble misses. The two views fail on different events,
  which is exactly the precondition for ensembling to pay.
* **One error mode dominates v2.** On dev, 8 of v2's 17 errors are curated
  exon-skipping predicted as A3/A5 (`CSE→A`). Full dev confusion:
  `CSE→A 8, A→CSE 3, RI→A 3, A→RI 2, A→NN 1`. The v1 ensemble makes only 5
  `CSE→A` over all 139. This is the single largest recoverable bucket, and
  `dsignal/diagnose_cse_a3.py` already exists to reason about it.

### Decisions taken with the user

1. **The deliverable ensemble is image-free** (signal-native members only). The
   hybrid with v1 image nets is measured and reported as a **documented upper
   bound**, not shipped.
2. **New gene-disjoint simulation batches are in scope** — used for *bagging*
   (members trained on disjoint data), not for scale-up. Framed and
   pre-registered as such, since scale-up per se is 0-for-3.
3. **A CSE-vs-A targeted member is in scope**, entering the ensemble like any
   other member and judged by McNemar on dev.

---

## What the code already does, and the three holes

From reading `dsignal/predict.py`, `train.py`, `score_benchmark.py`:

* `Predictor(checkpoint_paths, ...)` (`predict.py:150`) already holds a
  **per-checkpoint model + scaler**, so `--net runA/fold*.pt runB/fold*.pt`
  mechanically works today. Averaging is over **probabilities**
  (`fold_probs`, `predict.py:202`; `combine`, `:255`), and `PriorCorrection` is
  applied **per fold before averaging** (`:264`) — the correct estimator, and
  deliberately *not* v1's `sum(logits)` rule (`dsEnsemble.py:141-143`, a known
  issue in `report.md`).
* `--seed` (`train.py:398`) perturbs init, augmentation RNG and the inner val
  split, but **not** the outer `GroupKFold` partition — so fold *k* of every
  seed is trained and evaluated on the same gene-disjoint data. Exactly the
  property a multi-seed ensemble needs.
* `prior_sweep.cache_fold_probs` / `evaluate` (`prior_sweep.py:108-141`) already
  demonstrate the "extract once, evaluate many combinations analytically"
  pattern. Reuse it rather than re-inventing.
* `score_benchmark --out` → `benchmark_split compare A.tsv B.tsv --half dev` is
  already ensemble-agnostic. Keep the TSV schema byte-identical and every
  comparison stays McNemar-able for free.

Three holes block a *correct* multi-run ensemble:

| # | hole | where |
|---|---|---|
| H1 | Only checkpoint 1's `FeatureConfig` is kept; checkpoints 2..N have theirs silently discarded — mixing `--flank` or `--depth-norm` across runs loads without complaint and is then **wrong** | `predict.py:176-183` |
| H2 | One global `PriorCorrection` for all members, built from a single `--train-labels`; but each checkpoint already stores `train_class_counts` (`train.py:385`), which `Predictor` ignores entirely | `predict.py:160,264`; `score_benchmark.py:239` |
| H3 | `feature_subset` equality across *all* checkpoints is asserted, which forbids the single cheapest diversity axis | `predict.py:180` |

---

## Stages

Build → verify → commit each stage before starting the next. Everything is read
off **dev**; **test is scored exactly once, in Stage 6.**

### Stage 0 — plumbing and the reference (no training) · *strong model*

**0.1 A committed v1-ensemble reference in score_benchmark's own format.**
New `dsignal/v1_reference.py` (or a `make-v1-ref` subcommand on
`benchmark_split.py`) that reads `runs/v1/pred_*.csv` + `benchmark/split.tsv`
and writes `runs/benchmark_events_v1_ensemble.tsv` with the exact columns
`score_benchmark` writes (`csv region width truth pred correct
correct_collapsed p_* extract_s infer_s total_s truth_c pred_c`; probabilities
absent → leave empty, `compare` only needs `csv/region/truth/pred`). Do the same
for `runs/v1_pretrained/pred/` → `..._v1_squishstack.tsv`.
*Why:* it turns "beat the v1 ensemble" into
`benchmark_split compare runs/benchmark_events_v1_ensemble.tsv <new>.tsv --half dev`,
one command, same test as every other comparison on this branch.
**Verify:** the reference TSV reproduces 60/70 dev, 62/69 test, 122/139.
*Tier: cheap model — mechanical reformatting.*

**0.2 Fix H1–H3 in `Predictor`.** `predict.py`:
* validate **every** checkpoint's `feature_config` against the first, raising on
  mismatch (H1) — this is a correctness fix that stands alone;
* accept an optional per-checkpoint `PriorCorrection`, defaulting to one built
  from that checkpoint's own `train_class_counts` when `--train-labels` is not
  given (H2). Keep the existing global path working unchanged so every stored
  number stays reproducible;
* move the `feature_subset` check from "all checkpoints agree" to "all
  checkpoints *within one run* agree" (H3), with the per-run grouping owned by
  Stage 0.3.
**Verify:** `python -m dsignal.score_benchmark --net runs/geomwin_iv30_refit/fold*.pt
--half dev --tta-condition-swap --target-prior nn-weight --nn-weight 0.1
--train-labels sim/experiment/labels_geomwin.tsv` still returns **53/70**, and
`benchmark_split compare` against `runs/benchmark_events_stack_dev.tsv` gives
b=c=0. Any deviation means the refactor moved the estimator.
*Tier: strong model — silent-correctness territory.*

**0.3 `dsignal/ensemble.py` — the multi-run ensemble and its posterior cache.**
* `EnsembleSpec`: a small JSON/TSV listing members as
  `{run_dir, labels (optional, for that run's prior), weight (default 1.0)}`.
  Committed under `runs/ensembles/<name>.json` so every reported ensemble is
  reproducible from a file, not from a shell glob.
* `MultiRunPredictor`: one `Predictor` per run (each internally homogeneous),
  each with its own `PriorCorrection`. **Extract each event once** via
  `dsignal.extract`, then fan the record out to every run's
  `fold_probs(record)` — extraction dominates wall clock (`predict.py:278-283`),
  so never call `predict_region` per run.
* `combine(per_run_probs, rule, weights)` with three rules:
  `mean_prob` (default; the estimator v2 already uses),
  `mean_logprob` (geometric mean), and
  `sum_logit` (v1's rule, included only so the comparison is on the record).
  Prior correction stays **per fold, before any averaging**.
* A cache: `--cache-out runs/ens_cache_<half>.npz` storing per-run per-fold
  posteriors keyed by `(csv, region)`, plus a `--cache-in` path that skips
  extraction entirely. Model it directly on `prior_sweep.cache_fold_probs`.
  This is what makes Stages 4–5 cost minutes instead of hours.
* Wire into `score_benchmark.py` as `--ensemble <spec.json>` (mutually exclusive
  with `--net`), **reusing `score` and `report` unchanged** so the per-event TSV
  schema, the collapsed metric and the width bins stay identical.
**Verify:** a spec containing only `geomwin_iv30_refit` reproduces 53/70 dev
exactly, and `compare` against the stored TSV gives b=c=0.
*Tier: strong model — this is the core new component.*

### Stage 1 — the free members: seeds and the two untried knobs · *cheap model to run, strong to read*

All local, `DSnet`, on `sim/experiment/npz_geomwin` + `labels_geomwin.tsv`
(4,800 events; ~19 min/run per `runs/geomwin_iv30_refit.json`).

1. **Seeds.** Four more runs of the exact headline command with `--seed 1..4` →
   `runs/ens_seed{1,2,3,4}/`. With the existing seed-0 run that is 5 runs × 5
   folds = 25 members, all on the same gene-disjoint outer partition.
2. **`--avg-last-k 5`** (implemented, never run at scale; only active with
   `--refit`, `train.py:338-349`) → `runs/ens_avglastk5/`.
3. **`--inner-val-frac 0.45`** (only 0.15 and 0.30 have been tried; 0.30 was
   worth +2 over 0.15) → `runs/ens_iv45/`.

**Pre-registered ensemble E1 = the five seeds, equal weight, `mean_prob`, TTA,
`nn_weight 0.1`.** Score dev once. This is the clean "is the gap just variance?"
experiment and it costs ~80 min of GPU.
**Report:** E1 vs `runs/benchmark_events_stack_dev.tsv` (McNemar), and E1 vs
`runs/benchmark_events_v1_ensemble.tsv` (McNemar).
*Tier: cheap model can launch and tabulate; a strong model reads the result.*

### Stage 2 — view diversity · *strong model for the CSE member, cheap for the rest*

One run per view, all on `npz_geomwin`, all with the Stage-1 protocol
(`--refit --inner-val-frac 0.30 --augment all`):

| member | flag |
|---|---|
| `ens_allfeat` | `--feature-subset all` (0.873 sim-CV, its own best) |
| `ens_twohead` | `--two-head` (recovers 8 NN events under prior correction where the single head recovers 4) |
| `ens_depthscale` | `--depth-norm scale` |
| `ens_augsub` | `--augment flip condition_swap subsample` (a strictly weaker augmentation view) |
| `ens_nn15` | `--labels sim/experiment/labels_geomwin.nn15.tsv` (same records, 15% NN — a class-prior view) |

**Plus the CSE-vs-A member, `ens_cse`.** The mechanism to test first is the one
`diagnose_cse_a3.py` already established: a curated ES window often *also*
contains a genuine donor-sharing junction pair, so both answers are structurally
present and the model picks the wrong one. Two candidate implementations, in
order of preference:

1. **Motif features as an extra view.** Promote the three structural motif
   scores from `diagnose_cse_a3.py` (`a3`, `a5`, `se`, each with its
   differential strength) into event-level features and train a member on
   `--feature-subset all` plus them. This adds features → new
   `N_EVENT_FEATURES`, which is safe for loading (config comes from the
   checkpoint) but means this member must be its own run, which the multi-run
   ensemble now supports. **Note H1's fix makes this member load-safe;** without
   Stage 0.2 it would silently ride on another run's feature config.
2. If (1) is too invasive, a simpler class-weighted member: raise the CSE class
   weight so the A/CSE boundary moves, and let the ensemble average it against
   the unweighted members.

**Pre-registered ensemble E2 = E1's five seeds + one member per view family
above**, equal weight. Score dev once; McNemar vs E1 and vs the v1 reference.
*Tier: strong model for `ens_cse` (feature plumbing touches three call sites —
`baseline_gbm.apply_feature_subset`, `dataset.apply_feature_subset`,
`dataset.apply_feature_subset_meta`); cheap model for the other five runs.*

### Stage 3 — data bagging: new gene-disjoint batches · *strong model to launch, on pasteur*

**Read `CLAUDE.md`'s pasteur section and `crash_pasteur.md` before launching
anything.** `limitram`, all four thread variables at 1, `nice -n 19`, ≤12
shards, `/proc/loadavg` before and after, and hold the ssh session (scopes die
with it, `Linger=no`).

Four new batches `b8..b11`, driven by `sim/run_batch.sh <tag> <seed> <prior
design.tsv…>` with **all Stage-3-era knobs at their pre-Stage-3 defaults**
(`PSI_DIST=uniform`, `NULL_DPSI_SIGMA=0`, `COOC_FRAC=0`) so they are
`geomwin`-equivalent, and with every existing `design.tsv` from
`batches_geomwin/b1..b5` and the stage3 batch passed as `--exclude-genes` priors
so the new genes are disjoint from everything already used.

Then `sim/merge_batches.py` into `npz_bag2` / `labels_bag2.tsv` (3,840 events).
Acceptance gate before any training, per `CLAUDE.md`:
`python -m dsignal.audit_noise compare` and `python -m sim.psi_recovery_check`.
Also **inspect `design.tsv` against its intent before polyester runs** — the
`_null_dpsi` double-draw of the last session was caught exactly there.

Members: two runs trained on `npz_geomwin` alone and two on `npz_bag2` alone
(disjoint genes ⇒ genuinely decorrelated members), plus one on the union as a
scale control. **Pre-register the framing:** this is a bagging experiment; the
union run is the control that separates "bagging helped" from "more data
helped". If the union run alone matches the bagged pair, the finding is scale,
not bagging, and it must be reported that way.

**E3 = E2 + the bagged members.** Score dev once; McNemar vs E2.
*Tier: strong model — remote, shared machine, and a design that must be checked
before it is spent.*

### Stage 4 — combination rule and weighting · *cheap model, all from cache*

From the cached posteriors only, no re-extraction. Pre-register **exactly three**
rules — `mean_prob`, `mean_logprob`, `sum_logit` — and **two** weightings —
equal, and per-run weight ∝ that run's out-of-fold sim macro-F1 (available in
each run's `--json-out`). Six cells, evaluated on dev.
**Do not run greedy forward member selection.** At n=70 it would fit the dev
half and the gain would not survive to test; the value of the cache is speed,
not a licence to search. Record this restriction in the results doc.
*Tier: cheap model — arithmetic over a cache.*

### Stage 5 — the hybrid upper bound (reported, not shipped) · *cheap model*

Add the v1 `squishstack` net trained on v2's simulation
(`runs/v1_crossover/rs50.pth`) as an extra member, and separately the six
pretrained v1 nets via `dsEnsemble.py`. Report on dev:
* the image-free ensemble (the deliverable),
* image-free + v1 crossover net,
* the oracle union (already measured: 64/70 dev with the single v1 net,
  65/70 with the full v1 ensemble).
Label the whole section explicitly as a ceiling, not a deliverable. It answers
"how much complementary signal is left in the image view", which is the
scientific question behind the hybrid.
*Tier: cheap model — reuses `dsEnsemble.py` and the reference TSVs from 0.1.*

### Stage 6 — one test scoring, then write it up · *strong model*

Pick **one** ensemble — whichever of E1/E2/E3 wins on dev under the Stage-4
rule — and score `--half test` **once**. Then:
* `benchmark_split compare` it against `runs/benchmark_events_stack_test.tsv`
  (the current 58/70) and against `runs/benchmark_events_v1_ensemble.tsv` on
  test;
* write `plans/signal-native-v2-ensemble-results.md` and a new
  `plans/signal-native-v2-resume-4.md`; update `plans/README.md`, and update
  `CLAUDE.md`'s "Which v1 number to compare against" paragraph with the
  **corrected split-wise v1 figures** (85.7% dev / 89.9% test), keeping 87.9%
  next to them as the all-140 figure it actually is.
* Per branch convention: **never delete a measured number** — the corrected
  target sits next to the old one, with the reason.
* Commit. **Do not push.**

---

## Critical files

| file | change |
|---|---|
| `dsignal/predict.py` | H1/H2/H3 fixes in `Predictor.__init__`/`combine`; per-checkpoint `PriorCorrection` |
| `dsignal/ensemble.py` | **new** — `EnsembleSpec`, `MultiRunPredictor`, `combine` rules, posterior cache |
| `dsignal/score_benchmark.py` | `--ensemble <spec.json>`; reuse `score`/`report` unchanged |
| `dsignal/v1_reference.py` | **new** — v1 pred CSVs → score_benchmark-format TSVs |
| `dsignal/features.py`, `dataset.py`, `baseline_gbm.py` | Stage 2's motif features (three call sites, see `CLAUDE.md`) |
| `runs/ensembles/*.json` | **new** — the committed ensemble specs |
| `sim/run_batch.sh`, `sim/merge_batches.py` | unchanged; driven for `b8..b11` |

## Reuse, don't reimplement

`prior_sweep.cache_fold_probs`/`evaluate` (the cache pattern) ·
`baseline_gbm.grouped_folds`/`report`/`report_collapsed`/`print_vs_reference`
(the shared harness) · `score_benchmark.score`/`report` (the TSV schema) ·
`benchmark_split.load_split`/`filter_rows`/`compare` and
`prior_sweep.mcnemar` (every significance test) · `predict.PriorCorrection`,
`class_counts_from_labels`, `make_target_prior`.

## Verification

Gates, in order, at the stage each applies to:

```bash
# Stage 0 — the refactor must be an identity on the stored headline
python -m dsignal.score_benchmark --net runs/geomwin_iv30_refit/fold*.pt \
  --half dev --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
  --train-labels sim/experiment/labels_geomwin.tsv --out /tmp/reproduce_dev.tsv
python -m dsignal.benchmark_split compare \
  runs/benchmark_events_stack_dev.tsv /tmp/reproduce_dev.tsv --half dev   # expect b=c=0

# Stage 0.3 — single-run spec == the --net path
python -m dsignal.score_benchmark --ensemble runs/ensembles/single_iv30.json \
  --half dev --out /tmp/spec_dev.tsv                                      # expect 53/70

# Stage 2 — feature plumbing invariants (mandatory after any feature change)
python -m dsignal.verify_features
python -m dsignal.verify

# Stage 3 — simulation acceptance, before training on the new batches
python -m sim.psi_recovery_check
python -m sim.width_leak_check
python -m dsignal.audit_noise compare

# every stage — the comparison that decides anything
python -m dsignal.benchmark_split compare \
  runs/benchmark_events_v1_ensemble.tsv runs/benchmark_events_<E>_dev.tsv --half dev
```

**The success criterion, stated before any number is read.** Scored on the
**test** half only, paired against `runs/benchmark_events_v1_ensemble.tsv`:

* **parity** — the ensemble's test accuracy is ≥ v1's (currently 62/69 there)
  *and* the paired discordant counts do not favour v1;
* **a win** — McNemar p < 0.05 in v2's favour.

Dev cannot settle this: dev is where every v2 choice was read off, so a dev win
is not evidence. And the current test p = 0.267 is *not* parity — it is 4–5
events behind with too little power to prove it, which is exactly the ambiguity
this plan has to remove. Test is scored once, in Stage 6; if it is spent more
than once the criterion above is void and the next milestone needs a fresh
held-out half.

## Honesty ledger to carry forward

* `DPSI_BETA`/`PSI_LEVEL_BETA` remain fitted on all 140 benchmark events. Stage 3
  uses `PSI_DIST=uniform`, so it stays clean; if beta is ever used, refit on dev
  only.
* The NN operating point (`nn_weight 0.1`) is settled and pre-registered — reuse
  it, do not re-tune it.
* E1/E2/E3 compositions and the Stage-4 rule grid are pre-registered in this
  document. No greedy member selection on dev.
* The hybrid (Stage 5) is a ceiling measurement and must never be quoted as the
  v2 result.
