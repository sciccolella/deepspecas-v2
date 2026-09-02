# signal-native-v2 — Stage 11: simulate the TRASH regime, retrain, and treat borderline as NN

*Plan, 2026-09-02. Predecessors: `plans/signal-native-v2-bnew-dev-results.md`
(Stage 10), `plans/signal-native-v2-trash-results.md` (Stage 9).*

## Context

Stage 10 measured, for the first time, what v2 does with real NN events. The
result localizes the problem precisely:

* **NN recall is 9–14%** across every checkpoint set, and the failure is
  *representational, not calibration*: for true-NN dev events, NN ranks **last
  (5th) for 27/79 and 4th for 16/79**. Multiplying `p_NN` by 50 only reaches
  53% recall while dropping overall accuracy to 52% — no prior setting fixes
  a ranking this wrong.
* **`p_NN` as a real-NN detector has AUC 0.639**, essentially identical to
  Stage 9's TRASH-recoverability AUC of **0.612** from seven hand-built noise
  metrics. The model's NN axis and the TRASH axis are about equally weak.
* Stage 9 already measured why: sim reproduces `real-clean` almost exactly
  (`intron_cov_frac` 0.0343 vs 0.0356) but is **3.29× short of `real-TRASH`**
  (0.1127) and **2.28× short on `novel_junction_rate`** (0.0732 vs 0.1673).
* And real NN *is* the TRASH regime: **54.5% of real NN events are TRASH**,
  against 3.0% (A3), 0% (A5), 15.1% (CSE), 16.7% (RI).

So: simulated NN is a clean window with no event; real NN is a noisy, spurious
window that a differential caller flagged and a curator rejected. The
simulator has never generated that regime, which is why the model has no
representation for it. **This stage builds it, retrains, and measures.**

Additionally, per the user's 2026-09-02 point: **a TRASH event carrying a
positive label is so borderline that calling it NN is the defensible answer.**
Measured on dev, adopting that convention *costs* accuracy under every current
model (e4_union2 57.9% → 51.5%, v1 42.1% → 38.3%) — it is a harder, not a
flattering, target, which is what makes it worth adopting. It also states what
the simulator should produce: a window with TRASH-grade noise is NN regardless
of what was planted in it.

## Scope decisions (settled with the user)

* **Pilot before committing** to a full re-simulation — one cheap batch, gated
  on the measured `real-TRASH` targets, before spending ~10h of pasteur.
* **Include a real-supervision upper-bound probe**, dev-only, as a diagnostic
  ceiling. It never produces a deployed model.
* **`benchmark_new`'s test half stays unspent.** Everything below is dev.
  `benchmark/`'s owed Stage 8 Track C spend is also untouched.

---

## Stage 0 — lock the two scoring conventions  *(cheap tier)*

Both are reported side by side for every model from here on; neither replaces
the other.

* **Convention A (primary, unchanged):** `curation.tsv`'s `truth` as-is.
  Keeps continuity with every Stage 10 number.
* **Convention B (co-primary, new): TRASH ⇒ NN.** Any event with `trash == "1"`
  has truth NN regardless of its curated label. Drops `trash == "NA"` rows
  (7 on dev), so n=235 and NN truth = 98/235 = 42%.

Implementation: add `--trash-as-nn <curation.tsv>` to
`dsignal/score_benchmark.py`, applied in `score()` right after truth
assignment, and mirror it in `dsignal/benchmark_split.py compare` so McNemar
can be run under either convention. Key by `(csv, region)` via the
`CSV_LABEL` inverse, exactly as `dsignal/v1_reference.py --curation` and
`benchmark_build/trash_breakdown.py` already do.

**Pre-registered baselines to beat** (Stage 10, dev, both conventions, all
already stored in `runs/bnew_dev_*.tsv`):

| model | A: 5-class | B: TRASH-as-NN | NN recall (B) |
|---|---:|---:|---:|
| `e4_union2` | 58.7% (57.9% on the n=235 subset) | 51.5% | 9.2% |
| `e3_bagged` | 49.6% | 43.0% | 6.1% |
| `geomwin` single | 48.3% | 41.3% | 5.1% |
| v1 ensemble | 41.3% | 38.3% | 5.1% |

## Stage 1 — real-supervision upper bound, dev only  *(strong tier — diagnostic design)*

Before spending days of simulation, bound how much of the NN gap is closable
at all. Train on `benchmark_new` **dev only**, with folds grouped by
`split.tsv`'s existing `cluster` column (50 kb clusters — already computed,
reuse it rather than regrouping by gene, which `curation.tsv` does not carry
for all rows).

* Reuse `dsignal/train.py` unchanged where possible: it needs `--npz-dir` +
  `--labels`, so the work is a small adapter that extracts dev events to npz
  (`dsignal.extract` on the five `benchmark_new/*.csv`, dev-filtered) and
  emits a `labels.tsv` in `ix,event_id,label,kind` form, with `event_id`
  carrying the cluster id so grouping works.
* Report under both conventions. Also report NN recall specifically.
* **This is a ceiling, not a model.** It is trained on the evaluation set's
  dev half; it must never be scored on test, never ensembled, and never
  quoted as a deliverable. Write that next to the number.

Read: if the ceiling is also poor, the NN class is intrinsically hard here and
the simulation work should be re-scoped. If it is much higher, the gap is a
sim/real representation gap and Stages 2–4 are the right fix.

## Stage 2 — expose the knobs, add a TRASH layer, run ONE pilot batch  *(strong tier — the design step; pasteur ~2h)*

### 2a. Pass-through plumbing (`sim/run_batch.sh`)

`run_batch.sh` invokes `build_experiment` with only `--cooc-frac --seed`, so
every noise knob currently runs at its Python default. Add env-var
pass-through for `--premrna-frac-low/high`, `--novel-frac-low/high`,
`--novel-nn-frac-low/high`, `--neighbor-depth-low/high`, `--gene-mean-depth`,
and split `N_NN` into `N_NN_HARD` / `N_NN_PLAIN` / `N_NN_NOVEL`. Every new
var defaults to today's value so re-running any existing batch tag reproduces
it bit-identically.

### 2b. The TRASH layer (`sim/build_experiment.py`, `sim/select_events.py`)

A per-event trashiness flag, drawn **class-conditionally to match the measured
real rates** (NN 54.5%, CSE 15.1%, RI 16.7%, A3 3.0%, A5 0%). Trashy events
get their `premrna_frac` and novel-junction fraction drawn from an elevated
range chosen to hit the Stage 9 targets (`intron_cov_frac` ≈ 0.113,
`novel_junction_rate` ≈ 0.167), rather than the background range.

Two hard requirements:

1. **A new, independent RNG stream.** `build_experiment` currently does
   `SeedSequence(seed).spawn(4)`. Go to `spawn(5)` and use the *fifth* stream
   for the trash draw — `spawn(5)[:4]` is identical to `spawn(4)`, so every
   existing layer stays bit-identical and old batches still reproduce. This
   is the same isolation discipline the four existing streams already follow.
2. **The class-conditional rate must be argued, not assumed.** Making noise
   correlate with NN is legitimate *only* because the deployment distribution
   has the same coupling (a caller flags noisy regions; a curator rejects
   them). Record that argument next to the knob, and carry a
   **label-independent control arm** (uniform trash rate across classes) so
   the claim is measured rather than asserted — this is the same discipline
   `sim/window.py` v3 applied to the width coupling.

### 2c. Labeling policy knob

`TRASH_LABEL_POLICY = keep | nn`. Under `nn`, a trashy *positive* is emitted
with label `NN` (kind `nn_trash`), directly implementing the user's
borderline-is-NN semantics in training data. Under `keep`, it retains its
planted label. Both arms get trained in Stage 3; neither is assumed correct.

### 2d. Pilot and the gate

Run **one** batch on pasteur (`sim/run_batch.sh trash_pilot <seed> <all prior design.tsv>`),
~960 events, ~1.5–2h. **Run `audit_noise measure` on pasteur before the working
directory is pruned** — it needs `extract.csv` + BAMs, which do not come back
(this is exactly why Stage 9 had to reuse an old sim subsample):

```bash
python -m dsignal.audit_noise measure --csv $B/extract.csv --labels $B/labels.tsv \
  --subsample 300 --seed 0 --out runs/noise_audit_sim_trash_pilot.tsv
```

Then locally, the branch's stated acceptance test:

```bash
python -m dsignal.audit_noise compare --real runs/noise_audit_bnew.tsv \
  --sim runs/noise_audit_sim_trash_pilot.tsv \
  --half dev --split benchmark_new/split.tsv \
  --group-tsv benchmark_new/curation.tsv --group-col trash
```

**Pre-registered gate:** the pilot's trashy subgroup must bring
`intron_cov_frac` and `novel_junction_rate` to within **1.3×** of
`real-TRASH` (0.1127 / 0.1673), while the non-trashy subgroup stays within
1.3× of `real-clean` (0.0356 / 0.0973). If the gate fails, iterate on the
knob ranges using the pilot only — **do not proceed to full batches.**

Also run the standing gates on the pilot: `sim.psi_recovery_check` mean abs
error per label in the 0.13–0.20 band, and `sim.width_leak_check` (expect no
change; the window sampler is untouched, so an unexplained rise is a bug).

## Stage 3 — full batches and retraining  *(medium tier to run, strong tier to read; pasteur ~8–10h)*

4–5 batches, gene-disjoint from **all** prior batches via repeated
`--exclude-genes` (pass every existing `design.tsv`; `select_events.py`
asserts zero overlap at the end). Measure `audit_noise` on each before
pruning. Merge with `sim/merge_batches.py` into a new pool alongside
`npz_union2`.

Training arms, all with the established base recipe
(`--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30`,
5-fold `GroupKFold` by gene):

| arm | pool | labeling |
|---|---|---|
| A | new TRASH batches only | `keep` |
| B | union of `npz_union2` + new batches | `keep` |
| C | union pool | `nn` (trashy positives → NN) |
| D (control) | union pool, label-independent trash rate | `keep` |

Arm B is the expected headline; A isolates whether the new regime alone
suffices; C tests the borderline-is-NN semantics; D is the leak control for
2b's class-conditional coupling.

**pasteur rules apply** (`CLAUDE.md`, `crash_pasteur.md`): wrap in
`limitram 32G`, export all four thread-pool caps
(`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`),
`nice -n 19`, check `/proc/loadavg` before and a minute after launch, and
never `pkill -f` a pattern that also matches the ssh command carrying it.

## Stage 4 — evaluate, dev only  *(strong tier)*

For every arm, and against the Stage 0 baselines:

1. `score_benchmark` on `benchmark_new` **dev**, under **both conventions**,
   `--tta-condition-swap --target-prior nn-weight --nn-weight 0.1`.
2. **Regression guard**: score the old `benchmark/` **dev** half too. A new
   pool that fixes NN but collapses the positive classes is not a win — the
   old benchmark has no NN truth, so it is a clean read on positive-class
   damage.
3. `prior_sweep` on the best arm over `benchmark_new` dev; the whole
   `nn_weight` curve, both conventions. This is where a genuinely different
   operating point may finally be justified — but per Stage 10, changing the
   deployed constant needs more than one dev sweep.
4. McNemar via `benchmark_split compare` against `runs/bnew_dev_e4_union2.tsv`
   for every claim. n=242 (A) / 235 (B) needs ~10–14 discordant events.
5. Re-run the Stage 10 diagnostics that localized the problem: NN rank
   distribution among true-NN events, and `p_NN` AUC. **These are the real
   success criteria** — the goal is moving AUC 0.639 and the rank profile,
   not just aggregate accuracy.

## Stage 5 — ensemble if warranted, write up, commit  *(medium tier)*

If a single arm clears the baselines, reapply the view-diversity ensemble
recipe (`dsignal/ensemble.py`, `EnsembleSpec` JSON; members may have
different label files, as `e4_union2` already does). Then:

* `plans/signal-native-v2-trash-sim-results.md` — full stage-by-stage record,
  house style, retractions kept beside what they retract.
* `plans/signal-native-v2-resume-8.md` — new entry point; update
  `plans/README.md` and the `CLAUDE.md` pointer.
* Commit per convention (build → verify → commit per step, never push, never
  branch unprompted).

---

## Critical files

| file | change |
|---|---|
| `sim/run_batch.sh` | env-var pass-through for every noise knob; split `N_NN` three ways; `TRASH_*` vars |
| `sim/build_experiment.py` | the TRASH layer; `spawn(5)` with the new stream appended last |
| `sim/select_events.py` | class-conditional trash flag on design rows; `nn_trash` kind; label policy |
| `dsignal/score_benchmark.py` | `--trash-as-nn` convention B |
| `dsignal/benchmark_split.py` | same convention in `compare`, so McNemar works under B |
| `sim/merge_batches.py` | reuse as-is (`--batch TAG=DIR`, `--nn-frac`, `--subsample-events`) |
| `dsignal/train.py` | reuse as-is; `--class-weight-boost NN <f>` available if arms need it |
| `benchmark_new/*`, `benchmark/*` | **read-only, never regenerate** |

## Verification

1. **Reproducibility**: re-running an existing batch tag with all new env vars
   at their defaults produces a bit-identical `design.tsv` and
   `count_matrix.tsv`. This is the check that the `spawn(5)` change and the
   pass-through did not perturb the existing layers.
2. **Stage 2 gate**: `audit_noise compare` 3-way table shows the pilot's
   trashy subgroup within 1.3× of `real-TRASH` on `intron_cov_frac` and
   `novel_junction_rate`, non-trashy within 1.3× of `real-clean`.
3. **Standing sim gates**: `sim.psi_recovery_check` 0.13–0.20 mean abs error
   per label; `sim.width_leak_check` unchanged vs `npz_geomwin`'s 0.349/0.293.
4. **Extractor gates**: `python -m dsignal.verify` and
   `python -m dsignal.verify_features` still pass.
5. **Regression guard**: old `benchmark/` dev does not collapse; `e3_bagged`
   still reproduces 54/70 there.
6. **Every comparison carries a McNemar p-value**, under both conventions.
7. `git status` shows no modification under `benchmark/` or `benchmark_new/`.
8. **Test halves untouched**: no command in this stage passes `--half test`.
