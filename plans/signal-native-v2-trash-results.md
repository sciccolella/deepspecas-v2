# signal-native-v2 — Stage 9 results: `benchmark_new`, and TRASH-iness as signal

*Results, 2026-09-02. Plan: `plans/signal-native-v2-trash-signal.md`.*

---

## Summary

`benchmark_new/` is frozen, split, and includes real `NN` truth for the
first time. TRASH-iness (the user's spuriousness annotation) is **not**
cleanly recoverable from window-local `audit_noise`-style signal at the
~0.70 AUC bar the plan pre-registered as its decision gate — pooled
cluster-grouped logistic regression AUC came in at **0.612**. Per the plan's
discipline, **Stages 3 and 4 (spuriousness event features, conditioned NN
prior) did not run.** Stage 2's sim-vs-real comparison did run regardless,
and found a real, actionable gap: the simulator's noise knobs already
reproduce `real-clean`'s distribution closely, but never reach
`real-TRASH`'s — a cleaner explanation for part of the zero-shot plateau
than "scale is saturated."

**`benchmark_new`'s test half remains completely unspent.** Stage 8 Track
C's one-shot v1-vs-v2 test spend is still owed, on the *old* `benchmark/`
split, and is untouched by this stage.

---

## Stage 0 — the curation ledger, and the frozen set

573 candidate events (Stage 8 Track A3's consensus pool) went through
manual review. 88 are ambiguous (multi-label `Review`, e.g. `CSE/NN`, or one
blank row) and are recorded in `curation.tsv` but never scored anywhere.
The 485 unambiguous rows became `benchmark_new/`:

| truth | count | of which TRASH |
|---|---:|---:|
| A3 | 33 | 1 |
| A5 | 23 | 0 |
| CSE | 213 | 32 |
| RI | 57 | 9 |
| **NN** | 159 | 84 |
| **total** | **485** | **126** |

`trash` is three-valued (`1`/`0`/`NA`): 14 of the 485 scored rows are `NA`
(the unannotated prefix of `review_A5.tsv`, rows 1-17 -- the TRASH marker
itself wasn't adopted yet when those were reviewed, not that they were
skipped; the other four flags parsed from `Notes` are unaffected). Across
all 573 rows (scored + ambiguous) the tally is `1=160 / 0=396 / NA=17` --
this supersedes the plan doc's own `160/380/17`, which does not sum to 573
and was a documentation slip; `ingest_curation.py`'s output is internally
consistent (`126+345+14=485`, `396+160+17=573`) and is what this stage's
numbers are computed from.

**This is the first curated v2 benchmark with real NN truth.** The original
`benchmark/` has none, so NN recall has never been measurable there.

Verification: `dsignal.verify` 970/970 regions match (485 events x 2
conditions). `audit_noise padding` reproduces the old set's window geometry
exactly for the two most populous classes -- ES (CSE) median pad
`+100/+100` bp, IR (RI) median pad `+0/+0` -- with wider spread on A3/A5
(median pad `0/58` and `49/0`) explained by curation moving rows in from
other candidate classes (e.g. 6 of A3's 33 truth rows were originally IR
candidates) without re-padding; window geometry is a property of the
*candidate* class the row was extracted under, not its curated truth.
`benchmark_split make --trash-tsv` produced a 242/243 split, balanced per
`(truth, trash)` stratum to within one event.

---

## Stage 1 — is TRASH-iness recoverable from the signal? Gate: **no**

`benchmark_build/trash_probe.py`, dev half only (n=235 after dropping 7
`trash=NA` rows), joined `audit_noise measure`'s per-window metrics
(`runs/noise_audit_bnew.tsv`) to `curation.tsv`'s `trash` column.

**Pooled cluster-grouped 5-fold logistic regression AUC: 0.612** (n=235,
212 clusters), over `{intron_cov_frac, background_frac, antisense_frac,
dup_rate, unsupported_junction_frac, struct_span_frac, cov_cv}`. Below the
plan's pre-registered ~0.70 gate.

Per-metric pooled AUCs, none clearing 0.70:

| metric | pooled AUC | direction |
|---|---:|---|
| intron_cov_frac | 0.667 | higher = TRASH |
| novel_junction_rate (diagnostic) | 0.597 | higher = TRASH |
| struct_span_frac | 0.544 | higher = TRASH |
| dup_rate | 0.528 | higher = TRASH |
| antisense_frac | 0.522 | higher = TRASH |
| unsupported_junction_frac | 0.496 | lower = TRASH |
| background_frac | 0.396 | lower = TRASH |
| cov_cv | 0.390 | lower = TRASH |

`intron_cov_frac` is the strongest single signal and is the strongest
within CSE specifically (0.703). `cov_cv` and `unsupported_junction_frac`
do reasonably within RI alone (0.736, 0.700) but that does not survive
pooling across classes -- the per-class breakdown (full table in
`runs/trash_probe_dev.log`) is close to noise everywhere else, which is
exactly the check the plan asked for: no metric is simply re-encoding "this
is an NN row" in one direction, but none is a clean spuriousness detector
across classes either.

**The correlation with actual model error is real, even though the linear
window-level signal is weak.** Using `curation.tsv`'s already-recorded
`v2_pred` column (from the original consensus review, no new inference
run): mean(trash) = 0.132 among v2's correct dev predictions vs 0.455 among
its wrong ones. This is the same 70.4%/24.6% clean/TRASH split the plan's
context section measured on the full unambiguous pool, now confirmed
specifically on dev -- but it is a population-level association, not
something a 7-feature window-local linear probe can extract per-event at a
useful AUC.

**Decision, per the plan's own discipline:** Stages 3 and 4 do not run.
No spuriousness block was added to the event feature vector, no
spuriousness-conditioned NN prior was built, and `nn_weight=0.1` was not
re-derived on this set.

---

## Stage 2 — does the simulation reach the TRASH regime? Yes, and it doesn't need to be told to.

`audit_noise compare` gained `--group-tsv`/`--group-col` (split the real
side into named groups by a `curation.tsv` column) and `--half`/`--split`
(restrict the real side to dev first). Sim side: `runs/noise_audit_sim_b{12,
13,14,15}.tsv`, 800 windows -- the four newest batches in `npz_union2`'s
pool. (`batches_union2`'s working directories, including the BAMs, no
longer exist on pasteur; only `npz`/`labels.tsv`/`design.tsv` were kept.
Re-simulating to get a fresh sim-side measurement was out of scope by the
user's 2026-09-02 decision recorded in the plan, so this is the best
available subsample rather than a purpose-drawn one.)

Three-way table, dev half, `trash=NA` dropped (172 clean / 63 TRASH real
events; full table in `runs/noise_audit_bnew_grouped_dev.log`):

| metric | sim | real-clean | real-TRASH | sim -> TRASH multiplier |
|---|---:|---:|---:|---:|
| intron_cov_frac | 0.0343 | 0.0356 | **0.1127** | **3.29x** |
| novel_junction_rate | 0.0732 | 0.0973 | 0.1673 | 2.28x |
| cov_median | 3.5 | 2.5 | 8 | 2.29x |
| dup_rate | 0.1332 | 0.1453 | 0.1623 | 1.22x |
| background_frac (lower bound) | 0.0780 | 0.1474 | 0.0859 | 1.10x |

**The hypothesis holds.** Sim's `intron_cov_frac` already matches
`real-clean` almost exactly (0.0343 vs 0.0356) -- the simulator was
validated against, and reproduces, ordinary windows well. But it falls
3.3x short of `real-TRASH`'s intronic-coverage level, and 2.3x short on
both novel-junction rate and median depth. **A meaningful slice of the real
distribution -- the dev TRASH fraction, 63/242 = 26% -- was never
simulated at all.** This is a cleaner, more specific explanation for part
of the zero-shot transfer plateau than "scale is saturated" (the reading
`plans/signal-native-v2-scale-union2-results.md` settled on for single-model
scale specifically): it comes with a target multiplier per knob rather than
a guess, so if it is acted on it is a well-targeted re-simulation plan, not
another blind widen-and-see.

**No re-simulation happened in this stage** (user decision, 2026-09-02, per
the plan).

---

## What this does and doesn't change

* `benchmark_new/` is now the frozen home for the second curated real
  benchmark: 5 classes including NN, split, ready for use. `benchmark/`
  (the original 140-event set) is untouched.
* `dsignal.score_benchmark`'s `CSV_LABEL`/`report()` generalize to any CSV
  set including NN, so scoring `benchmark_new/` works unmodified; scoring
  the old `benchmark/` produces byte-identical output to before (`DEFAULT_CSVS`
  unchanged, `has_nn_truth` false for that set so the old wording prints).
* `dsignal.benchmark_split make` gained `--trash-tsv` for TRASH-stratified
  splitting; unused, it behaves exactly as before.
* `dsignal.audit_noise measure` gained `cov_cv`; existing stored TSVs
  (`runs/noise_audit_sim_b*.tsv` etc.) simply lack that column and are
  otherwise unaffected. `compare` gained `--group-tsv`/`--group-col`/`--half`;
  unused, it behaves exactly as before.
* **No model, checkpoint, feature vector, or prior-correction code
  changed.** The event feature vector is still 32 features
  (`N_EVENT_FEATURES` untouched); every existing checkpoint's stored scaler
  is unaffected because nothing in `dsignal/features.py`,
  `dsignal/dataset.py`, or `dsignal/predict.py` was touched this stage.
* `nn_weight=0.1`'s "settled" status is **unchanged and still open** as a
  caveat: it was settled against a benchmark with no NN truth at all. Stage
  1's gate failure means this stage does not resolve that caveat either --
  a spuriousness-conditioned prior was the mechanism that would have acted
  on it, and it wasn't built.

---

## Open honesty caveats (carried forward, one new)

1. **The NN operating point (`nn_weight=0.1`) is still not re-derived on any
   NN-containing benchmark.** `benchmark_new` now has 159 NN events and a
   dev/test split, so `dsignal/prior_sweep.py` *could* re-derive it -- this
   stage did not, because Stage 1's gate blocked the conditioned-prior work
   the re-derivation was bundled with. A plain re-derivation of the global
   constant on `benchmark_new` dev is a small, separately-schedulable task,
   not blocked by anything in this stage.
2. **`DPSI_BETA`/`PSI_LEVEL_BETA` are still fitted on the *old* benchmark's
   140 events**, unrelated to and unaffected by this stage.
3. **`benchmark_new`'s test half is completely unspent** -- 243 events,
   never read by any tool in this stage. Stage 8 Track C's one-shot
   v1-vs-v2 spend against the *old* `benchmark/split.tsv` test half is
   separately still owed and untouched.
4. **New**: the Stage 2 sim-vs-real gap (`intron_cov_frac` 3.3x,
   `novel_junction_rate` 2.3x short of `real-TRASH`) is a measured,
   specific target for a future noise-layer widening plan, not yet acted
   on. It was deliberately not acted on in this stage (no re-simulation, per
   the user's 2026-09-02 decision).

---

## Artifacts

| path | what |
|---|---|
| `benchmark_new/{A3,A5,ES,IR,NN}.csv`, `events.bed`, `c1.bam`/`c2.bam`, `README.md` | the frozen, curated, 5-class benchmark |
| `benchmark_new/curation.tsv` | all 573 reviewed rows, full ledger |
| `benchmark_new/split.tsv` | dev/test split, stratified by `(truth, trash)` |
| `benchmark_build/ingest_curation.py` | reviews/\*.tsv -> curation.tsv |
| `benchmark_build/make_curated_csvs.py` | curation.tsv -> benchmark_new/\*.csv |
| `benchmark_build/trash_probe.py` | Stage 1's AUC/logreg/vs-correctness probe |
| `runs/noise_audit_bnew.tsv` | `audit_noise measure` over all 485 `benchmark_new` events |
| `runs/trash_probe_dev.log` | Stage 1's full output |
| `runs/noise_audit_bnew_grouped_dev.log` | Stage 2's full 3-way table |
