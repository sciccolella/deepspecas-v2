# signal-native-v2 — START HERE (2026-09-02, Stage 10)

Resume document. Branch `signal-native-v2`, **nothing pushed** (per repo
convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-6.md` as the entry point.
`plans/signal-native-v2-bnew-dev-results.md` is the full stage-by-stage
record of this session. `resume-6` and `plans/signal-native-v2-trash-results.md`
(Stage 9) still hold everything about `benchmark_new`'s freeze/split and the
TRASH-recoverability gate; nothing about how that set was built changed this
session — only how it's been *used*.

---

## The one thing to know

**`benchmark_new` dev has been scored for the first time**, with real
predictors (prior correction, TTA), not the `v1_pred`/`v2_pred` review-aid
columns Stage 9 quoted. Three things came out of it:

1. v2 beats v1 on this set (`e4_union2` 58.7% vs v1 41.3% 5-class,
   p<0.0001; 58.7% vs 48.8% collapsed, p=0.0032) — but this is a
   **descriptive** comparison, not the pre-registered arena: both models'
   predictions were used as curation aids while building `benchmark_new`.
   **Corrected 2026-09-02 (Stage 12)**: the curator states the prediction
   columns were visible but did not inform the assessment — this is a real
   result, and `benchmark_new` test is a legitimate pre-registered arena.
2. **NN recall is now a measurable, and clearly the dominant, weakness** —
   8.9%–13.9% across the three v2 checkpoint sets measured, at the currently
   deployed `nn_weight=0.1`. This is the first time this branch has ever had
   a real NN-recall number.
3. `nn_weight=0.1`'s sensitivity curve on real NN-containing data has an
   argmax *above* 0.1 (0.25–0.5 depending on checkpoint set), the opposite
   direction the old all-negative benchmark would ever show — but neither
   move clears McNemar significance at n=242 dev, so `nn_weight=0.1` stays
   the deployed value. The deployment argument for it (candidates come from
   an upstream caller; the 25% training NN fraction was a class-balance
   choice, not a deploy-prior estimate) is unaffected either way.

Also confirmed, properly scored rather than review-column-derived: TRASH
events are ~2.0–2.4x worse across every predictor than clean events on this
set (`e4_union2` 68.6% clean vs 28.6% TRASH), corroborating Stage 9's
central finding with real numbers.

## No v1-vs-v2 claim — still true, on both benchmarks

`benchmark/`'s test half (Stage 6, resume-4/5) remains spent and unchanged:
v1 60/70 dev + 63/70 test, v2 (E3) 54/70 dev + 60/70 test, McNemar p=0.549,
no parity, no win. `benchmark_new`'s test half (243 events) is **still
completely unspent** — this session scored dev only. Stage 8 Track C's
one-shot v1-vs-v2 spend against `benchmark/`'s test-half remaining potential
is still owed, separately from `benchmark_new`'s own future one-shot test
spend.

## What is now settled, and what it means for the next session

* **v2's NN behavior on real data is measurable for the first time**, and it
  is weak in an absolute sense (~9-14% recall) even though v2 clearly beats
  v1 in aggregate on this set. If NN recall matters for a deployment
  decision, this is the number to improve, not overall accuracy.
* **`nn_weight=0.1` was sensitivity-tested, not moved.** A future session
  changing it should use this session's curve (`runs/bnew_prior_sweep_*.tsv`)
  plus eventually a test-half read, not dev alone — the moves measured here
  are not statistically significant at n=242.
* **The v1-vs-v2 comparisons in this session are not evidence for a
  pre-registered claim.** `benchmark_new`'s curation process used both
  models' predictions as review aids, so a model that curation happened to
  agree with more often will look better here by construction to some
  unknown degree. Treat the direction as suggestive (and consistent with
  `benchmark/`'s own numbers), not dispositive.
  **Retracted 2026-09-02 (Stage 12)** — the curator states the columns were
  visible in the sheets but were not consulted, so no such by-construction
  advantage exists. See `plans/signal-native-v2-real-supervision.md`.
* **No re-simulation, no retraining, no feature or model changes** happened.
  One code addition: `dsignal/v1_reference.py --curation` mode, needed
  because `benchmark_new`'s curation moved rows between candidate classes
  (so the old glob-by-candidate-CSV-stem approach doesn't map to truth).

## Open honesty caveats (carried forward, none newly resolved)

1. **`nn_weight=0.1`** — sensitivity-tested this session (see the curve),
   still not changed. Unchanged status from `resume-6`.
2. **`DPSI_BETA`/`PSI_LEVEL_BETA`** still fitted on `benchmark/`'s 140 events.
   Untouched.
3. **`benchmark_new`'s test half (243 events) is completely unspent.**
   `benchmark/`'s own test-half spend (Stage 8 Track C) is separately still
   owed.
4. **Stage 2's (Stage 9's) sim-vs-real TRASH gap** — `intron_cov_frac` 3.3x,
   `novel_junction_rate` 2.3x short of `real-TRASH` — unchanged, no
   re-simulation happened this session either.
5. **New**: every v1-vs-v2 number this session produced is descriptive, not
   pre-registered — see above. Do not quote these numbers as a fresh
   head-to-head result without that caveat attached.
   **Retracted 2026-09-02 (Stage 12)**: curator attestation, see above. The
   numbers may be quoted as a real result on the dev half.

## What is new in the code

| file | what |
|---|---|
| `dsignal/v1_reference.py` | new `--curation` mode (`build_rows_from_curation`) for benchmarks where curation moved rows across candidate classes; existing `--glob` mode unchanged |
| `benchmark_build/trash_breakdown.py` | (new) clean-vs-TRASH accuracy join for any set of already-scored per-event TSVs |

No `dsignal/features.py`, `dsignal/dataset.py`, `dsignal/model.py`,
`dsignal/predict.py`, `dsignal/train.py`, `dsignal/score_benchmark.py`, or
`dsignal/prior_sweep.py` changed this session.

## Artifacts

| path | what |
|---|---|
| `plans/signal-native-v2-bnew-dev-results.md` | the full stage-by-stage record |
| `runs/bnew_dev_{e4_union2,e3_bagged,geomwin,v1_ensemble}.tsv` | per-event scored predictions, dev half |
| `runs/bnew_prior_sweep_{geomwin,union2}_dev.tsv` | `nn_weight` sensitivity tables |
| `runs/bnew_dev_trash_breakdown.tsv` | clean vs TRASH accuracy, all four predictors |

## Conventions

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar; 4-6 events is noise at n=140,
  8-10 at n=70; `benchmark_new` dev is n=242, roughly 10-14 discordant
  events for significance.
- Never delete a measured number from a docstring or plan doc — retractions
  go next to the number they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/rendering on
  pasteur.
