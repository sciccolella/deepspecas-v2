# Results

Every number below can be claimed, with its caveat, and cites the committed file
it comes from. `python scripts/check_results.py` recomputes all of the tables in
sections 1–3 from the per-event TSVs (read-only: it scores nothing). "Collapsed"
means A3+A5 → A, so the five v2 classes are comparable to v1's four. "Test" and
"dev" are the halves of `benchmark/split.tsv` and `benchmark_new/split.tsv`.

**Read first:** [§5 Caveats](#5-caveats) and [§6 Retracted figures](#6-retracted-figures).
No held-out curated half remains (§5.1), so everything in this file is final; any
new model-vs-model claim on curated data is exploratory.

## 1. Headline: `benchmark_new` test (the pre-registered, one-shot measurement)

243 events, `benchmark_new/{A3,A5,ES,IR,NN}.csv`, curated truth as frozen
(convention A). Model: `e4_union2` (6 members × 5 folds, simulation-only training,
no fine-tuning) with the settings frozen before the spend: condition-swap TTA,
`--target-prior nn-weight --nn-weight 0.1`. Baseline: the v1 six-net ensemble.

| | correct / 243 | collapsed accuracy |
|---|---|---|
| **v2 `e4_union2`** | 143 | **58.8%** |
| v1 six-net ensemble | 110 | 45.3% |

Paired exact McNemar: 40 events only v2 gets right, 7 only v1, **p = 1.1e-6**
(printed `0.0000` in the logs); the pre-registered PASS criterion (v2 ahead and
p < 0.05, one test, α = 0.05, no correction) is met. 5-class v2 accuracy:
142/243 = 58.4% (v1's 5-class number is meaningless: it never emits A3/A5).
Convention B (TRASH counted as NN, n = 236): 125/236 = 53.0% vs 96/236 = 40.7%,
32 vs 3.

Sources: `runs/bnew_test_e4_union2.tsv`/`.log`, `runs/bnew_test_v1_ensemble.tsv`,
`runs/bnew_test_compare_primary.log`, `runs/bnew_test_compare_primary_convB.log`;
pre-registration and verdict in `plans/signal-native-v2-final-test-spend.md`.
Dev for the same models (n = 242): v2 142 (58.7%), v1 118 (48.8%)
(`runs/bnew_dev_e4_union2.tsv`, `runs/bnew_dev_v1_ensemble.tsv`) — v2 is stable
dev→test (+1), v1 drops 8.

### Per-class recall (test, collapsed, v1 six-net vs `e4_union2`)

| truth class | n | v1 | v2 | v2-only / v1-only | p |
|---|---|---|---|---|---|
| A3 | 17 | 10 (58.8%) | 13 (76.5%) | 3 / 0 | 0.25 |
| A5 | 11 | 5 (45.5%) | 11 (100%) | 6 / 0 | 0.031 |
| CSE | 106 | 69 (65.1%) | 91 (85.8%) | 27 / 5 | 0.0001 |
| **NN** | **80** | **0 (0%)** | **3 (3.8%)** | 3 / 0 | 0.25 |
| RI | 29 | 26 (89.7%) | 25 (86.2%) | 1 / 2 | 1.0 |

Per-class tests are not part of the pre-registered hypothesis and carry no
multiplicity correction. Source: `runs/bnew_test_compare_primary.log`.

### The NN weakness, stated plainly

**v2 finds 3 of 80 curated no-event (NN) regions (3.8%).** 77 of its 100 test
errors are NN-truth events, called CSE (29), A3 (24), A5 (13) or RI (11). On dev
NN recall is 11/79 (13.9%), so even that figure is optimistic about the
operating point. Excluding NN truth, v2 gets 140/163 = 85.9% and v1 110/163 =
67.5%; the margin over v1 comes from CSE (+22) and A5 (+6), not from NN. Curated
NN was found not to be expressible from the current features (Stage 18, rule 3 in
`plans/signal-native-v2-stage18-domain-probe.md`). 5-class confusion matrix (test,
rows = truth, columns A3/A5/CSE/NN/RI): A3 `13 0 1 1 2`,
A5 `1 10 0 0 0`, CSE `10 4 91 1 0`, NN `24 13 29 3 11`, RI `3 0 1 0 25`.

## 2. `benchmark/` (140 curated events, no NN class)

Split 70 dev / 70 test (`benchmark/split.tsv`, cluster-disjoint at 50 kb).
`benchmark/` contains **no NN events**, so every NN prediction there is a false
positive and NN recall is unmeasurable.

| model | dev (n=70) | test (n=70) | source |
|---|---|---|---|
| v2 16-member ensemble `e3_bagged` | 54 (77.1%) | **60 (85.7%)** | `runs/benchmark_events_e3_{dev,test}.tsv` |
| v2 single model (`geomwin_iv30_refit`) | 53 (75.7%) | 58 (82.9%) | `runs/benchmark_events_stack_{dev,test}.tsv`, `runs/benchmark_stack_{dev,test}.log` |
| v1 six-net ensemble | 60 (85.7%) | **63 (90.0%)** | `runs/benchmark_events_v1_ensemble.tsv` + `benchmark/split.tsv` |
| v1 single net (`squishstack`, v1-trained) | 56 (80.0%) | 50 (71.4%) | `runs/benchmark_events_v1_squishstack.tsv` + split |

On `benchmark/` test the pre-registered parity/win criterion for v2 **was not
met**: E3 vs the v1 ensemble, 4 v2-only / 7 v1-only, McNemar **p = 0.549**
(3 events behind, not significant). Source: `plans/signal-native-v2-ensemble-results.md`
(Stage 6). v1 six-net over all 140 = 123/140 = 87.9%; a single v1 net trained on
v1's own data = 106/140 = 75.7%, and trained on v2's simulated data 74.3%
(`runs/v1_crossover/`, `runs/v1_pretrained/` — not in this repository; see
`DATA.md`). A single v2 model should be measured against ~76%, not 87.9%.
The two benchmarks therefore disagree in direction on the ensemble comparison
(v1 ahead on `benchmark/` test, n.s.; v2 ahead on `benchmark_new` test,
p < 0.0001); §5.5 lists what differs between them.

## 3. Cost

| | per event | basis |
|---|---|---|
| v2 `e4_union2` (6 members, TTA) | **0.106 s** (0.025 extract + 0.081 inference) | `runs/bnew_test_e4_union2.log`, 243 events in 25.8 s, local machine |
| v1 six-net ensemble | **~17 s**, image rendering alone | design-time estimate carried in `plans/signal-native-v2.md` (l. 180) and printed by `score_benchmark`; from v1's `report.md` (v1 repository). **Not re-measured here**, and not the same hardware as the v2 figure |

Treat the ~160× ratio as an order-of-magnitude statement, not a benchmark.
CPU-only inference is checked: `dsignal.predict` on 6 events took 0.13 s/event
with CUDA hidden (this repository, step 6 of the release; not a benchmark).

## 4. Exploratory secondaries on `benchmark_new` test

Scored on the same spend, descriptive only, never promoted. Both are
**real-gene-flagged**, so read their numbers as possibly optimistic.

| model | test (n=243) | vs v1 | vs `e4_union2` |
|---|---|---|---|
| Stage 20 arm A, 3 seeds pooled (1 of 2,258 real training genes in the test set) | 144 (59.3%) | 44 vs 10, p<0.0001 | 12 vs 11, p=1.0 |
| SSL `both`, 3 seeds pooled (42 of 9,911 real training genes in the test set) | 141 (58.0%) | 44 vs 13, p<0.0001 | 13 vs 15, p=0.85 |

Arm A's dev lead (150 vs 142) did not reproduce on test. Sources:
`runs/bnew_test_{stage20_A,ssl_ft_both}_pooled.tsv`,
`runs/bnew_test_compare_{A,ssl}_{v1,e4}.log`, `runs/bnew_test_eligibility.log`.

## 5. Caveats

1. **No held-out curated half remains.** `benchmark/` test was spent once
   (Stage 6, 2026-08-19) and `benchmark_new` test once (2026-10-05). The project
   decision is no further manual curation. Any further model-vs-model comparison
   on curated data is exploratory and must not be reported as confirmatory.
2. **Genes shared between the simulation pool and the test set.** The simulated
   training pool was drawn from the whole annotation and never excluded against
   `benchmark_new` genes: 162 of 222 symbol-matched (194 of 315 span-overlapping)
   test genes appear among its 12,480 genes. The pre-registered eligibility rule
   was therefore relaxed, by user decision, to real-derived rows and PTBP1 only
   (`dsignal/check_test_eligibility.py`, `runs/bnew_test_eligibility.log`). The
   simulated reads are not the test reads, but the gene sequence and exon
   structure may have been seen.
3. **Hyper-parameters fitted on benchmark events.** `DPSI_BETA`/`PSI_LEVEL_BETA`
   in `sim/select_events.py` were fitted on all 140 `benchmark/` events (not dev
   only), so anything simulated with `--psi-dist beta` is domain-calibrated along
   that axis; refit on dev only if reused. The NN operating point
   (`nn_weight = 0.1`) was pre-registered from the deployment argument in
   `dsignal/prior_sweep.py`, then scored on dev; it was not tuned on test.
4. **`e4_union2` was selected on `benchmark/` dev**, not `benchmark_new` dev: it
   is dev-selected and test-clean for `benchmark_new`, but its membership carries
   the `benchmark/` dev exposure (87.1% dev collapsed on `benchmark/`, a
   dev-tuned number — `runs/ensembles/e4_union2.json`,
   `plans/signal-native-v2-scale-union2-results.md`).
5. **The two benchmarks differ.** `benchmark/` has no NN class and was curated
   from rMATS+SUPPA consensus events; `benchmark_new` has 80/243 NN events on
   test and was curated later, with v1/v2 prediction columns visible in the
   review sheets. The curator states those columns did not inform the manual
   assessment (attestation 2026-09-02, `benchmark_new/README.md`); the earlier
   "descriptive only" caveat was retracted on that basis (§6), but it is a
   statement, not something the data can show.
6. **Loader deviation.** `Predictor` loads pre-Stage-22b checkpoints with
   `strict=False`, tolerating only the pretext-head keys (`mask_token`,
   `recon_head.*`, `proj_head.*`). Dev outputs were verified identical before
   test was scored (max |Δp| = 0.0).
7. **Reproducibility of the packaged model.** On `benchmark_new` dev the
   packaged `models/e4_union2` reproduces `runs/bnew_dev_e4_union2.tsv` exactly
   in the lab env (DSnet, 0 prediction differences, max |Δp| = 0) and with 0
   prediction differences and max |Δp| = 1.3e-7 in `envs/inference.yml` (torch
   2.13 CPU). Test was not re-scored for either.

## 6. Retracted figures

Kept on record next to what they retract; do not quote them.

| figure | status | why |
|---|---|---|
| v1 ensemble 62/69 on `benchmark/` test (122/139 overall) | **retracted** | deduplicated by region alone instead of `(csv, region)`, dropping one of the two rows of the region that appears in both `A3.csv` and `A5.csv`; correct value 63/70 (`plans/signal-native-v2-ensemble-results.md`, Stage 0.1) |
| 110/140 = 78.6% for a single v2 model (older headline) | **retracted** | its training epoch was selected on the fold it then reported (`dsignal/score_benchmark.py` docstring) |
| "v2 is 17 events behind v1" | **retracted belief** | compared a single model with the six-net ensemble on the wrong denominator; real test gap on `benchmark/` was 3 events (`plans/signal-native-v2-ensemble-results.md`) |
| `benchmark_new` comparisons are "descriptive, not pre-registered" | **retracted 2026-09-02** | curator attestation (see §5.5) |
| two headline findings of `plans/signal-native-v2-resume.md` (2026-08-16) | **retracted** | marked superseded in `plans/README.md` |
| 78.6% dev for the union-alone single run | **not retracted, not a result** | best dev number in the ensemble plan, but it was a pre-registered diagnostic (scale vs bagging), never eligible for test (`plans/signal-native-v2-ensemble-results.md`); unrelated to the 78.6% above |
| Stage 5 hybrid (v2 + one v1 net) 87.1% dev; oracle unions 91.4–94.3% | **ceiling only** | not reachable by the image-free model; must not be quoted as the v2 result |
