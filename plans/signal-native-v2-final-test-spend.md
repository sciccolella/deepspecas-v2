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
