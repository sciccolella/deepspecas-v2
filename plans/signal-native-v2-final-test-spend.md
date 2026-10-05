# signal-native-v2 — final measurement: the one `benchmark_new` test spend

**Status:** approved 2026-10-05. Its commit is the pre-registration
timestamp. **Nothing in this file may change after any test row is read.**
Implement in a fresh session; this session (Stage 19's) stopped at the
handoff.

## Why now

The Stage 18–22 accuracy programme
(`plans/signal-native-v2-stage18-22-accuracy-plan.md`, "Final measurement")
ends with one spend of `benchmark_new` test (243 events). Every gated
intervention has now been run and none passed:

| stage | outcome |
|---|---|
| 20.3 real fine-tune | all three arms FAIL |
| 22b SSL pretraining | no variant passes |
| 19 junction-tail re-sim | Gate A FAIL; stopped before the full re-sim, closed 2026-10-05 (commit `24b9f54c2`) |

Stage 19's "Gate A fails twice" branch therefore applies: *"the remaining
work is the final test-spend plan with the best dev-selected existing model."*

## Dev evidence the choice was made from

Computed 2026-10-05 from committed TSVs. All rows are scored with
`--tta-condition-swap --target-prior nn-weight --nn-weight 0.1`. Pooled means
the mean of `p_*` over 3 seeds' TSVs, then argmax
(`dsignal.benchmark_split pool`).

| model | dev collapsed (n=242) | NN recall | vs `e4_union2`, McNemar |
|---|---|---|---|
| Stage 20 arm A, 3 seeds pooled | 150 (62.0%) | 3/79 (3.8%) | +8 (20 vs 12), p=0.22; NN 9 vs 1, p=0.02 *worse* |
| **`e4_union2`** (6 members, sim-only) | **142 (58.7%)** | 11/79 (13.9%) | — |
| SSL `both`, 3 seeds pooled | 142 (58.7%) | 6/79 (7.6%) | 17 vs 17, p=1.0 |
| SSL `mfm`, 3 seeds pooled | 138 (57.0%) | 4/79 | p=0.57 |
| `union2_base`, 3 seeds pooled | 127 (52.5%) | 7/79 | −15, p=0.0015 |
| v1 six-net ensemble | 118 (48.8%) | — | `e4_union2` beats it, 43 vs 19, p=0.0032 |

**Eligibility.** No candidate's training pool contains a `ptbp1_arena` /
`real_*_ptbp1` row. That holds for `npz_union2` (sim), for `npz_union2_real3`
(aqr/prpf4/u2af2) and for the SSL pool, which never contains ptbp1 by
construction. Step 2 re-asserts it mechanically.

**User decision (2026-10-05):** `e4_union2` is the primary. Arm A has the
best dev point estimate but failed its own pre-registered gate on curated NN,
and its +8 is not significant. Promoting it would override that gate. Arm A
and SSL `both` are scored as **descriptive secondaries** on the same spend.

## Pre-registered measurement

- **Data:** `benchmark_new` test half per `benchmark_new/split.tsv` (243
  events), CSVs `benchmark_new/{A3,A5,ES,IR,NN}.csv`. Convention A: `truth`
  as frozen. TRASH rows count as their curated truth.
- **Primary hypothesis:** `e4_union2` collapsed accuracy > v1 six-net
  ensemble collapsed accuracy, two-sided paired McNemar, **α = 0.05**.
  Exactly one test; no correction.
  - **PASS:** v2 point estimate > v1 and p < 0.05.
  - **Not significant:** p ≥ 0.05 with v2 ahead.
  - **FAIL:** v2 ≤ v1.
  - Reported verbatim whichever happens.
- **Frozen inference settings** (identical to dev):
  - `--ensemble runs/ensembles/e4_union2.json --tta-condition-swap --target-prior nn-weight --nn-weight 0.1`;
  - no `--train-labels` (each member uses its stored `train_class_counts`, as in `runs/bnew_dev_e4_union2.log`);
  - no `--min-confidence`.
- **v1:** predictions already exist in `benchmark_new/curation.tsv`
  (`v1_pred`, from the Stage 8 consensus run, keyed by `(stem, region)`). Get
  them with
  `dsignal.v1_reference --curation benchmark_new/curation.tsv --split benchmark_new/split.tsv --half test`.
  No v1 re-run is needed: these are the same predictions the dev v1 number
  used. Note that the column was visible to the curator but not consulted
  (README attestation).
- **Descriptive secondaries** (no hypothesis, no verdict, never promoted by
  this spend):
  - Stage 20 arm A, 3 seeds pooled: `runs/stage20_A_s{0,1,2}/fold*.pt`, `--train-labels sim/experiment/labels_union2_real3.tsv`.
  - SSL `both`, 3 seeds pooled: `runs/ssl_ft_both_s{0,1,2}/fold*.pt`, with the same `--train-labels` its dev TSVs used (see `plans/signal-native-v2-stage22b-ssl-pretrain.md`; train prior printed in `runs/bnew_dev_ssl_ft_both_s0.log`).
  - Each gets a McNemar against v1 and against `e4_union2`, labelled exploratory.
- **Reported, not gated:**
  - 5-class accuracy;
  - per-class recall with per-class McNemar (`--by-class`);
  - the TRASH-as-NN convention B (`--trash-as-nn benchmark_new/curation.tsv`);
  - dev–test difference for each model;
  - wall clock.
- **Nothing else touches test.** No re-runs with other settings, no prior
  sweep, no threshold choice on test. If a scoring bug is found after test
  rows are read, fix it and rescore with identical settings, and record both
  runs.

## Steps

| # | step | model tier |
|---|---|---|
| 1 | **Reproducibility guard (dev only, before any test read).** Re-score `e4_union2`, arm A s0 and SSL-both s0 on `--half dev` with the exact commands above, into scratch paths. Require `pred` + `correct_collapsed` to be identical to the committed `runs/bnew_dev_*.tsv` (p_* to ~1e-6). If any differs, stop and report: the checkpoints or code have drifted. | Sonnet |
| 2 | **Eligibility assertion.** Script check: the gene set of each candidate's training labels (`labels_union2.n12480.tsv`, `.nn15.tsv`, `labels_union2_real3.tsv`, SSL pool manifest) ∩ genes of `benchmark_new` test rows = ∅, and no `ptbp1` kind/row. Use the same gene mapping the Stage 14/16/20.1 exclusion builds used. A hit makes that model ineligible; if the primary is hit, stop. | Sonnet |
| 3 | **Score test once.** Primary `e4_union2` → `runs/bnew_test_e4_union2.tsv`. v1 → `runs/bnew_test_v1_ensemble.tsv`. Secondaries: per seed `runs/bnew_test_{stage20_A,ssl_ft_both}_s{0,1,2}.tsv`, then `benchmark_split pool` → `runs/bnew_test_{stage20_A,ssl_ft_both}_pooled.tsv`. Keep every `.log`. | Haiku/Sonnet (mechanical) |
| 4 | **Compare.** `dsignal.benchmark_split compare runs/bnew_test_v1_ensemble.tsv runs/bnew_test_e4_union2.tsv --half test --split benchmark_new/split.tsv --by-class` (primary), plus convention B, plus the secondaries' comparisons. | Haiku/Sonnet |
| 5 | **Write-up.** Results section appended to this file (verdict first, then tables). Add `plans/signal-native-v2-resume-11.md` as the new entry point, the `plans/README.md` row, and update CLAUDE.md's headline paragraph: `benchmark_new` test is then **spent**, the same as `benchmark/` test. Commit, no push. | Opus |

Local only (GPU, `DSnet` env). About 1 h. **No pasteur.**

## Honesty caveats to carry into the results

- `nn_weight=0.1` and `DPSI_BETA`/`PSI_LEVEL_BETA` (fit on all 140
  `benchmark/` events) are unchanged from resume-10's caveats.
- `e4_union2`'s membership was chosen on `benchmark/` dev, not
  `benchmark_new` dev. It is still dev-selected and test-clean.
- Curated NN is 79/242 of dev, and v2 NN recall is ~14%. Expect NN to
  dominate the error profile on test too; report it, do not explain it away.
- This is the programme's last spend. After it, no held-out curated half
  remains on this branch.

---

# Results (appended 2026-10-05; everything above this line is the frozen pre-registration)

## Verdict

**PRIMARY: PASS.** `e4_union2` collapsed **143/243 = 58.8%** vs the v1
six-net ensemble **110/243 = 45.3%**; paired McNemar discordant **40
v2-only / 7 v1-only, p < 0.0001** (printed `0.0000`), two-sided, α = 0.05,
one test, no correction. v2 ahead and p < 0.05 → the pre-registered PASS
criterion is met. Convention B (TRASH-as-NN, n=236): 125/236 = 53.0% vs
96/236 = 40.7%, 32 vs 3. 5-class: 142/243 = 58.4% (v1's 5-class column is
meaningless; it never emits A3/A5).

This is the programme's last held-out half. `benchmark_new` test is now
**spent**, the same as `benchmark/` test.

## Deviations from the frozen plan (both recorded, neither touches the measurement)

1. **Loader fix before step 1.** Stage 22b.3 added `mask_token`/`recon_head.*`/
   `proj_head.*` to `JunctionSetTransformer`; `Predictor` used a strict
   `load_state_dict`, so `e4_union2` and Stage 20 arm A (pre-22b checkpoints)
   would not load. `dsignal/predict.py` now loads `strict=False` and tolerates
   only those keys. Step 1 was rerun with the fix: `pred`, `correct_collapsed`
   and `p_*` are identical to the committed `runs/bnew_dev_*.tsv` (max |Δp| =
   0.0) for `e4_union2`, arm A s0 and SSL-both s0.
2. **Eligibility criterion (step 2) relaxed by user decision.** Taken
   literally (training-gene set ∩ test-gene set = ∅) it **fails for every
   candidate**: the simulated pool was drawn from the whole annotation and
   was never excluded against `benchmark_new` genes (162 of 222 symbol-matched
   / 194 of 315 span-overlapping test genes appear among its 12,480 genes).
   The Stage 14/16/20.1 exclusions only ever guarded the *real* arena rows.
   The user approved applying the criterion to **real-derived rows and ptbp1
   only**. Under it (`dsignal/check_test_eligibility.py`,
   `runs/bnew_test_eligibility.log`):

   | candidate | real-gene overlap with test | ptbp1 rows | status |
   |---|---|---|---|
   | `e4_union2` (primary) | 0 (no real rows) | 0 | **eligible** |
   | Stage 20 arm A | 1 of 2,258 | 0 | **flagged** |
   | SSL `both` | 42 of 9,911 (19 by symbol) | 0 | **flagged** |

   Arm A and SSL `both` were scored on the same spend (no extra test
   exposure) as descriptive secondaries, but a flagged model has trained on
   real reads from a test gene; **read their test numbers as possibly
   optimistic**. They gate nothing.

## Primary and secondaries (test, n=243, collapsed, convention A)

| model | test | dev (n=242) | dev→test | vs v1 (McNemar) | vs `e4_union2` |
|---|---|---|---|---|---|
| v1 six-net ensemble | 110 (45.3%) | 118 (48.8%) | −8 | — | — |
| **`e4_union2`** (primary, eligible) | **143 (58.8%)** | 142 (58.7%) | +1 | 40 vs 7, p<0.0001 | — |
| Stage 20 arm A, 3 seeds pooled (flagged) | 144 (59.3%) | 150 (62.0%) | −6 | 44 vs 10, p<0.0001 | 12 vs 11, p=1.0 |
| SSL `both`, 3 seeds pooled (flagged) | 141 (58.0%) | 142 (58.7%) | −1 | 44 vs 13, p<0.0001 | 13 vs 15, p=0.85 |

Exploratory only. The three v2 variants are indistinguishable on test; arm A's
dev lead (+8) did not reproduce (+1, p=1.0), consistent with the dev decision
not to promote it.

## Per-class recall (primary vs v1, collapsed, per-class McNemar)

| class | n | v1 | `e4_union2` | v2-only / v1-only | p |
|---|---|---|---|---|---|
| A (A3) | 17 | 10 (58.8%) | 13 (76.5%) | 3 / 0 | 0.25 |
| A (A5) | 11 | 5 (45.5%) | 11 (100%) | 6 / 0 | 0.031 |
| CSE | 106 | 69 (65.1%) | 91 (85.8%) | 27 / 5 | 0.0001 |
| NN | 80 | 0 (0%) | 3 (3.8%) | 3 / 0 | 0.25 |
| RI | 29 | 26 (89.7%) | 25 (86.2%) | 1 / 2 | 1.0 |

Macro-recall 70.5% vs 51.8%. (Per-class tests are not part of the
pre-registered hypothesis; no correction applied.)

## What the PASS does and does not say

- **NN dominates the error and is not explained away.** NN recall is 3/80
  (3.8%); of the 100 v2 errors, 77 are NN-truth events, mostly called CSE
  (29), A3 (24), A5 (13) or RI (11). Confusion matrix (rows=truth,
  A3/A5/CSE/NN/RI): A3 `13 0 1 1 2`, A5 `1 10 0 0 0`, CSE `10 4 91 1 0`,
  NN `24 13 29 3 11`, RI `3 0 1 0 25`. Excluding NN truth, v2 is 140/163 =
  85.9% and v1 110/163 = 67.5%. v1 gets 0/80 on NN.
- The margin over v1 comes mainly from CSE (+22) and A5 (+6), not from NN.
- `nn_weight=0.1` and `DPSI_BETA`/`PSI_LEVEL_BETA` (fit on all 140
  `benchmark/` events) are unchanged resume-10 caveats. `e4_union2`'s
  membership was chosen on `benchmark/` dev, not `benchmark_new` dev; it is
  dev-selected and test-clean, but the sim pool shares genes with the test
  set (deviation 2).
- Dev–test: v2 is stable (+1, −1 events), v1 drops 8; flagged arm A drops 6.
- Wall clock: 0.106 s/event (0.025 extract + 0.081 inference), 25.8 s for
  243 events, six-member ensemble with TTA, vs v1's ~17 s/event on rendering.
- Test was scored once per model with the frozen settings; no re-runs.

## Artifacts

`runs/bnew_test_{e4_union2,v1_ensemble}.tsv`, `runs/bnew_test_{stage20_A,ssl_ft_both}_{s0,s1,s2,pooled}.tsv`,
every `.log`, `runs/bnew_test_compare_*.log` (primary with `--by-class`,
convention B, and each secondary vs v1 and vs `e4_union2`),
`runs/bnew_test_eligibility.log`.
