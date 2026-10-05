# signal-native-v2 — START HERE (2026-10-05, after the final `benchmark_new` test spend)

Supersedes `plans/signal-native-v2-resume-10.md` as the entry point; that
file still holds the Stage 18/20 state and the real-data pool rules. Branch
`signal-native-v2`, **nothing pushed**.

## State in one paragraph

Every gated intervention of the Stage 18–22 programme failed its gate
(20.3 fine-tune, 22b SSL, 19 junction-tail re-sim). The programme then spent
`benchmark_new` test once, pre-registered in `plans/signal-native-v2-final-test-spend.md`
(commit `261c41fce`), on the dev-selected `e4_union2` ensemble versus the v1
six-net ensemble. **Result: PASS.** 143/243 (58.8%) vs 110/243 (45.3%)
collapsed, 40 v2-only vs 7 v1-only, McNemar p < 0.0001. Convention B 53.0%
vs 40.7%. Stage 20 arm A (144) and SSL `both` (141) are statistically
indistinguishable from `e4_union2` on test (p=1.0, 0.85) — and are
real-gene-flagged (see below). Full tables, per-class recall and caveats are
in the Results section of that plan file.

## What to carry forward

- **No held-out curated half remains.** `benchmark/` test is spent (Stage 6)
  and `benchmark_new` test is spent (this spend). Any further v1-vs-v2 or
  model-vs-model claim on curated data needs new, independently curated
  events. The memory note "no new manual curation" still stands, so treat
  further curated-accuracy claims as exploratory.
- **NN is still the weak class**: 3/80 test recall (3.8%), 11/79 on dev.
  77 of the 100 v2 test errors are NN-truth events. Curated NN was shown
  non-expressible from the current features (Stage 18 rule 3).
- **Deviation 1 — loader.** `dsignal/predict.py` loads checkpoints
  `strict=False`, tolerating only the Stage 22b pretext keys
  (`mask_token`, `recon_head.*`, `proj_head.*`). Pre-22b checkpoints load
  again; dev outputs are bit-identical.
- **Deviation 2 — eligibility.** The literal "training genes ∩ test genes = ∅"
  fails for every candidate because the sim pool covers the annotation
  (162–194 shared genes). The approved criterion is **real-derived rows and
  ptbp1 only**: `e4_union2` eligible; arm A (1 gene) and SSL `both`
  (42 genes) flagged and read as possibly optimistic. Checker:
  `python -m dsignal.check_test_eligibility`.
- `nn_weight=0.1` and `DPSI_BETA`/`PSI_LEVEL_BETA` caveats from resume-10
  are unchanged.

## Where things are

| what | where |
|---|---|
| pre-registration + results | `plans/signal-native-v2-final-test-spend.md` |
| test per-event TSVs / logs / comparisons | `runs/bnew_test_*` |
| eligibility log | `runs/bnew_test_eligibility.log` |
| the model | `runs/ensembles/e4_union2.json` (6 members, `npz_union2`, 12,480 events) |
| scoring | `python -m dsignal.score_benchmark --ensemble runs/ensembles/e4_union2.json --half test --split benchmark_new/split.tsv --csv benchmark_new/{A3,A5,ES,IR,NN}.csv --tta-condition-swap --target-prior nn-weight --nn-weight 0.1` — **do not use for further selection; test is spent** |
