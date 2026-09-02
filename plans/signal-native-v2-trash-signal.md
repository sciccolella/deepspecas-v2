# signal-native-v2 — Stage 9: curated `benchmark_new`, and TRASH-iness as signal

*Plan, 2026-09-02. Successor to `plans/signal-native-v2-fresh-benchmark.md` (Stage 8).
Entry context: `plans/signal-native-v2-resume-5.md`.*

---

## Context

Stage 8 Track A stopped at A3: a 573-event uncurated consensus pool in
`benchmark_new_uncurated/`, waiting on manual curation (A4) before it could be
frozen, split, and used for the one-shot v1-vs-v2 spend the branch has owed since
Stage 6. **A4 is now done.** `benchmark_new_uncurated/reviews/review_{A3,A5,ES,IR}.tsv`
carry two new columns: `Review` (the curated truth class) and `Notes` (free-text
annotations, of which `TRASH` is the interesting one).

Two things came out of reading those 573 rows, and both change what this stage
should be.

**1. The curated pool is not the pool the CSVs claimed.** 130 of 485 unambiguously
reviewed rows were relabelled into a class other than their source CSV's, and 159
rows are `NN` — no real event in the window. Truth is derived from the CSV
*basename* everywhere in this branch (`CSV_LABEL`, `dsignal/score_benchmark.py:76`;
`dsignal/benchmark_split.py:119,154`), so freezing this set means re-partitioning
rows by `Review`, and NN becomes a scored truth class for the first time.

| Review | clean | TRASH | unannotated | total |
|---|---|---|---|---|
| CSE | 180 | 32 | 1 | 213 |
| NN | 70 | **84** | 5 | 159 |
| RI | 45 | 9 | 3 | 57 |
| A3 | 32 | 1 | 0 | 33 |
| A5 | 18 | 0 | 5 | 23 |
| *(multi-label, e.g. `CSE/NN`)* | 50 | 34 | 3 | 87 |
| *(blank)* | | | | 1 |

**The `TRASH` note has partial coverage, and blank ≠ clean everywhere.** The user
curated `review_A5.tsv` first and only started using the note partway through it:
its first `TRASH` row is index 18 of 51, and **A5 rows 1–17 are unannotated**, not
clean. Every other file annotates throughout (`review_A3.tsv` first TRASH at 33/70,
`review_ES.tsv` at 5/372, `review_IR.tsv` at 15/76 — all with clean rows before
them, so those blanks are real negatives). The 17 unannotated rows are the
"unannotated" column above; excluding them is what the accuracy table below does.
The user may annotate them later, which would upgrade them in place.

**2. TRASH is where the whole error budget lives.** The user's `TRASH` note marks
windows with heavy spurious (usually intronic) alignment. Measured on the 485
unambiguous rows, collapsed accuracy splits almost cleanly along it:

| | clean (n=345) | TRASH (n=126) |
|---|---|---|
| v1 six-net ensemble | 200 = **58.0%** | 23 = **18.3%** |
| v2 (`v2_pred` column) | 243 = **70.4%** | 31 = **24.6%** |

v2's clean-subset 70.4% is consistent with everything it has ever scored. Nearly
all of the excess error is in TRASH windows, and 67% of TRASH truth is NN. The
failure mode is specific and stateable: **the model calls a real event in a
spurious region.** That is exactly the error `nn_weight` was introduced to fight —
and `nn_weight=0.1` was tuned against a benchmark containing *no* NN truth, so the
settled operating point is not settled on this set.

The user's framing is the right one: TRASH is not a reason to discard events, it is
a label to learn from. And the branch already has the machinery to measure it —
`dsignal/audit_noise.py measure` emits `intron_cov_frac`, `background_frac`,
`antisense_frac`, `dup_rate`, `n_junctions` vs `n_supported`, `struct_span_frac`,
all **annotation-free and window-local**, on the identical code path for real and
simulated BAMs.

**Intended outcome:** a frozen, split, 5-class 485-event `benchmark_new/`; a
measured answer to "is TRASH-iness recoverable from the signal alone, and does the
simulation reach that regime at all"; and, if it is, a spuriousness block in the
event-level feature vector plus a spuriousness-conditioned NN prior — the two
cheapest ways to act on it without re-simulating.

---

## Discipline (read before Stage 1)

* **The new test half is not touched by this plan.** Stage 8 Track C's one-shot
  v1-vs-v2 spend is still owed and still unspent. Everything here is read off the
  new **dev** half. Split first (Stage 0), then analyse.
* **The TRASH labels must not become fitted parameters.** They live in the
  benchmark; fitting a detector on them and then scoring the benchmark is test-set
  fitting of the kind two retracted numbers on this branch came from. The design
  that avoids it: the model consumes **raw, unfitted** `audit_noise`-style
  quantities as features and learns to use them **from simulation**; the TRASH
  labels are used only to *validate on dev* that those quantities are informative.
  No coefficient anywhere is fit against a `Notes` column.
* Per `CLAUDE.md`: never delete a measured number; retractions sit next to what
  they retract. `nn_weight=0.1`'s provenance gets a caveat added, not removed.
* Build → verify → commit each stage. No `git push`, no new branches.

---

## Stage 0 — ingest curation, freeze and split `benchmark_new/` *(cheap model, one strong review)*

### 0.1 `benchmark_build/ingest_curation.py` *(new)*

Reads the four `reviews/*.tsv`, writes `benchmark_new/curation.tsv` — one row per
reviewed event, **all 573 retained**, columns:

`csv_source, region, gene_symbol, rmats_pvalue, rmats_incleveldiff, suppa_dpsi,
suppa_pvalue, v1_pred, v2_pred, review_raw, truth, ambiguous, notes_raw,
trash, borderline, low_signal, complex, similar_signal`

Normalisation rules, all of which have real instances in the data:

* `Notes` is upper-cased and split on `,`; **`.` is normalised to `,` first** —
  `BORDERLINE.TRASH` is one row that would otherwise be dropped. Flags are set by
  substring on the resulting tokens (`TRASH`, `BORDERLINE`, `LOW SIGNAL`,
  `COMPLEX`, `SIMILAR SIGNAL` / `SAME SIGNAL`). Unrecognised tokens (`SUS REGION`,
  `NO SIGNAL`, `VERY FEW READS`, a stray `REGION chr2:...`) are preserved in
  `notes_raw` and not silently dropped.
* `ambiguous = 1` when `Review` contains `/` or `+` (87 rows), or is blank (1 row).
  `truth` is written only for unambiguous rows.
* **`trash` is three-valued: `1` / `0` / `NA`.** `NA` for the 17 rows in the
  unannotated prefix of `review_A5.tsv` (rows 1–17, i.e. everything before its
  first `TRASH` row); blank `Notes` anywhere else is a real `0`. The prefix length
  is a **written-down constant with a comment**, not re-derived by scanning for the
  first TRASH row — that heuristic would silently mislabel a file whose genuinely
  clean prefix happens to be long. The other three files' first-TRASH indices
  (33/70, 5/372, 15/76) are recorded in the module docstring as the evidence that
  their blanks are negatives. If the user later annotates the 17, flipping the
  constant to 0 is the whole change.
* Sanity assertions, all currently true and all worth having fail loudly:
  573 rows in, 573 out; every `region` present in the matching
  `benchmark_new_uncurated/<stem>.csv`; no region appearing in two review files;
  every non-ambiguous `truth` in `("A3","A5","CSE","RI","NN")`.

### 0.2 Re-partition into `benchmark_new/`

`benchmark_build/make_curated_csvs.py` *(new)* emits, from the **485 unambiguous**
rows, headerless `region,bam1,bam2` files partitioned by `truth`, using the old
naming convention so `CSV_LABEL` keeps working:

`A3.csv` (33), `A5.csv` (23), `ES.csv` (213), `IR.csv` (57), **`NN.csv` (159)**

plus `events.bed` (interval-identical to the region column, per
`benchmark/README.md:38-42`) and a `README.md` recording the study
(`ENCSR376FGR` / `ENCSR264TUE`, SRSF3 KD vs control, HepG2), the consensus counts
(573), the curation counts above, and the TRASH tally.

BAM paths point at `benchmark_new/c1.bam`/`c2.bam`, regenerated by
`samtools view -L benchmark_new/events.bed` from the merged BAMs **on pasteur**
(the full merges are 8.4/8.8 GB and stay there; only the subset comes back).
The old README's caveat carries over: off-window quantities measured on the subset
are lower bounds.

The 87 ambiguous rows and the 1 blank stay in `curation.tsv` only — recorded, not
scored. This keeps every accuracy number a single-truth measurement, matching the
rest of the branch.

### 0.3 Teach the scorer that NN is a class

Three touch points, all small:

* `dsignal/score_benchmark.py:76` — `CSV_LABEL` gains `"NN": "NN"`; `DEFAULT_CSVS`
  stays pointed at `benchmark/` (the old set) so nothing existing changes.
* `dsignal/score_benchmark.py:129-186` `report()` — the hardcoded `"all 140"`, the
  fixed `("A3","A5","ES","IR")` loop, the `v1's 123/140 = 87.9%` note and the
  **"the benchmark contains no NN events"** block all read from the loaded rows
  instead. Stage 8 Track C already flagged the first of these as owed.
* `dsignal/benchmark_split.py` — `assign()` stratifies on `CSV_LABEL[stem]`, so it
  picks NN up for free. **Add TRASH to the stratification key**: `--trash-tsv
  benchmark_new/curation.tsv` makes the per-class quota a per-`(class, trash)`
  quota. Without it the halves can differ by enough TRASH rows to make dev and
  test non-comparable — 26% of the set carries the flag, and it moves accuracy by
  40 points. `trash = NA` is its **own** stratification value, not folded into
  `0`; 14 of the 17 are unambiguous and are scored benchmark events like any other.

Then:

```bash
python -m dsignal.benchmark_split make --csv benchmark_new/*.csv \
  --trash-tsv benchmark_new/curation.tsv --out benchmark_new/split.tsv
```

Verify the printed no-straddle assertion, ~242/243 halves, and per-`(class,trash)`
balance. **Commit the split before running any model against the set.**

---

## Stage 1 — is TRASH-iness recoverable from the signal? *(strong model; local)*

Pure measurement, no model change, dev half only.

1. `python -m dsignal.audit_noise measure --csv benchmark_new/*.csv --out
   runs/noise_audit_bnew.tsv`. Works unchanged — `measure` takes any
   `region,bam1,bam2` CSV.
2. `benchmark_build/trash_probe.py` *(new)* joins that TSV to `curation.tsv` on
   `region`, **drops `trash = NA` rows** (they are unlabelled along the axis being
   probed, and treating them as negatives would inject label noise into the
   negative class), and reports, **on the dev half**:
   * per-metric AUC of `trash` (`intron_cov_frac`, `background_frac`,
     `antisense_frac`, `dup_rate`, `1 - n_supported/n_junctions`,
     `struct_span_frac`, `cov_cv`), with the same table split by `truth` class —
     the check that a metric is not simply re-encoding "this is an NN row";
   * a 5-fold cluster-grouped (reuse `benchmark_split.cluster`) logistic-regression
     AUC over the whole metric block, as an upper bound on how much of the
     annotation is signal-recoverable;
   * the same metrics tabulated against v2's existing per-event correctness, to
     confirm the 70.8/24.6 split is explained by measurable spuriousness rather
     than by something invisible.

**Decision gate.** If the pooled dev AUC is below ~0.70, TRASH is not
signal-recoverable at window level and Stages 3–4 do not run — the finding is
written up and the stage ends at Stage 2. Record the number either way.

Note on scope: `novel_junction_rate` is deliberately **excluded** from the feature
work. `dsignal/extract.py` masks STAR's `+20` annotated bit (`motif_raw % 20`), so
it is not recoverable from stored `.npz` and needs its own BAM pass
(`audit_noise._raw_bam_stats`). Making it a model feature would mean re-extracting
all 12,480 simulated events on pasteur. It is still reported in Stage 1 and Stage
2 as a diagnostic; if it turns out to be the strongest single discriminator, that
is the finding that justifies a re-extraction plan.

---

## Stage 2 — does the simulation reach the TRASH regime? *(strong model; local)*

`audit_noise compare` already does sim-vs-real. Extend `compare` to accept a
`--group-tsv`/`--group-col` so the real side can be split into **TRASH vs clean**
(the plumbing exists — `_by_group` at `audit_noise.py:510` groups rows by a key).

Compare three distributions: `sim` (a subsample of `npz_union2`'s batches),
`real-clean`, `real-TRASH` (`trash = NA` excluded, as in Stage 1).

The hypothesis worth stating before looking: the simulator's noise knobs
(`--premrna-frac-{low,high}` 0.01–0.10, `--novel-frac-{low,high}` 0.005–0.03,
`sim/build_experiment.py:145,163`) were each validated only by showing they
*degraded* the baseline, never by matching a measured real amount. If sim matches
`real-clean` and stops short of the `real-TRASH` tail, then a quarter of the real
distribution was never simulated, and that is a cleaner explanation of the
zero-shot plateau than "scale is saturated" — which would make widening those
knobs a well-targeted next plan with a number to hit, rather than a guess.

Deliverable: the three-way table, plus the multiplier per metric needed to move
sim's distribution onto `real-TRASH`. **No re-simulation in this plan** (user
decision, 2026-09-02).

---

## Stage 3 — spuriousness features in the event vector *(strong model; local)*

Only if Stage 1's gate passes.

### 3.1 `dsignal/features.py`

Add a spuriousness block to `_event_feature_names()` (currently 32 names,
`features.py:121-153`). Every one must be computable from the stored `.npz`, so
the existing `npz_union2` pool and the benchmark both work with **no
re-extraction**:

| feature | source |
|---|---|
| `intron_cov_frac_{1,2}` | port `audit_noise._intron_cov_frac` (`:273`) |
| `background_frac` | port `audit_noise._background_frac` (`:306`) |
| `antisense_frac` | port `audit_noise._antisense_frac` (`:328`) |
| `unsupported_junction_frac` | `1 - n_supported / n_junctions` |
| `struct_span_frac` | port `audit_noise._flanks` (`:347`) |

Port, do not duplicate: the audit helpers take `(norm, idx)` from
`features.normalize_record`, so the clean move is to lift them into `features.py`
and have `audit_noise.py` import them — keeping the "one source of truth for the
representation" property that makes the GBM and the Transformer comparable.

**These are not geometry.** Leave `_GEOMETRY_EVENT_FEATURES`
(`baseline_gbm.py:101`) untouched so the default `--feature-subset no-geometry`
keeps them.

### 3.2 The three call sites, and the checkpoint contract

`CLAUDE.md` names them: `baseline_gbm.apply_feature_subset`,
`dataset.apply_feature_subset`, `dataset.apply_feature_subset_meta`. This change
touches only event features, so the token-count logic in the last two is
unaffected — but `N_EVENT_FEATURES` moves from 32 to 37, which **mismatches every
stored scaler**. Old checkpoints keep *loading* (config comes from the checkpoint,
`predict.py:178`) but must be re-scored to confirm they are unchanged.

### 3.3 Train and score

Two runs, the single-model recipe from `CLAUDE.md` on `npz_union2`, differing only
in the feature block, so the comparison is clean:

```bash
python -m dsignal.train --npz-dir sim/experiment/npz_union2 \
  --labels sim/experiment/labels_union2.n12480.tsv \
  --augment all --feature-subset no-geometry --refit --inner-val-frac 0.30 \
  --out-dir runs/union2_spur --json-out runs/union2_spur.json
```

Score on **new dev** against a same-config control without the block, and on
**old dev** as a regression check. Report overall (all 485, `NA` included — it is a
truth-class metric and the TRASH annotation does not enter it) and split by `trash`
into `1`/`0`/`NA` — the whole point is the TRASH subset, where the baseline sits at
24.6%. McNemar via `dsignal.benchmark_split compare` on both.

---

## Stage 4 — spuriousness-conditioned NN prior *(strong model; local)*

`nn_weight=0.1` is a global constant fixed on a benchmark with zero NN truth. On a
set that is 33% NN, and where NN concentrates in TRASH windows, the deployable
version of that argument is a prior that **varies with the window**: spurious
windows should carry a higher NN prior, clean ones a lower one.

* `predict.make_target_prior` (`:121`) gains a `nn-weight-spurious` kind taking
  `(nn_weight_clean, nn_weight_spurious)` and a scalar spuriousness score in
  `[0,1]`, interpolating in log space. The score is the **unfitted** composite of
  Stage 1's metrics (a fixed, written-down formula — not regression coefficients
  fit on the benchmark).
* `PriorCorrection` (`:49`) becomes per-event: `weights` is currently precomputed
  in `__init__` (`:88`); allow `apply()` to take an event-specific target. The
  **per-fold-before-averaging** invariant (`:57-59`) is load-bearing and must be
  preserved — a per-event prior does not change where the correction is applied.
* Re-derive the operating point on **new dev** with `dsignal/prior_sweep.py`, which
  already evaluates N prior settings analytically from cached per-fold posteriors
  (one extraction pass). Two endpoints, pre-registered before reading: the
  best global `nn_weight` on new dev, and the best `(clean, spurious)` pair. The
  spurious-conditioned version has to beat the re-derived *global* one, not the
  stale 0.1, or it is not a finding.

Record the caveat: `nn_weight=0.1`'s "settled" status (`prior_sweep.py` docstring,
`CLAUDE.md`) was settled against a set with no NN truth. That statement is not
deleted — it gets the new-set measurement written next to it.

---

## Stage 5 — write up *(strong model)*

`plans/signal-native-v2-trash-results.md` and `plans/signal-native-v2-resume-6.md`.
Must carry forward: the 573→485 curation ledger, the clean/TRASH accuracy split for
both v1 and v2, Stage 1's AUC, Stage 2's sim-vs-real three-way table, and the
explicit statement that **`benchmark_new`'s test half remains unspent** and Stage 8
Track C is still owed.

---

## Verification

* **Stage 0 ingest:** the four assertions in 0.1; `wc -l` of the five emitted CSVs
  sums to 485; the `trash` column tallies **1=126 / 0=345 / NA=14** over
  unambiguous rows and **160 / 380 / 17** over all 573 (these are the counts this
  plan's tables were computed from — a mismatch means the note parser changed
  behaviour); `python -m dsignal.verify` on the new `events.bed`/BAM subset (the
  uncurated pool passed 1146/1146 — the curated subset must too).
* **Stage 0 geometry parity:** `python -m dsignal.audit_noise padding` on
  `benchmark_new/events.bed` must reproduce the old set's padding distribution
  (CSE +100 bp/edge, RI/A3/A5 +0). This is the check that the windows match the
  geometry the model was trained for; A3 already established it for the uncurated
  pool, but curation changed *which* class each row belongs to, so re-run it.
* **Stage 0 split:** the printed no-straddle assertion, plus per-`(class, trash)`
  dev/test balance.
* **Stage 3 feature change:** `python -m dsignal.verify_features` (canonicalization
  involution, exact A3↔A5 flip, coverage off-by-one) and `python -m dsignal.verify`
  must both pass — the new features are canonicalization-sensitive
  (`intron_cov_frac` is per condition, and conditions swap under `flip`).
* **Stage 3 backwards compatibility:** re-score `runs/geomwin_iv30_refit` on **old**
  dev and confirm it reproduces **53/70** exactly. A different number means the
  `N_EVENT_FEATURES` change broke a stored scaler.
* **Stage 4:** `prior_sweep` with the new kind at `(0.1, 0.1)` must reproduce the
  global `nn_weight=0.1` result bit-for-bit — the degenerate case of the new code
  path is the old code path.
* **Keying, everywhere:** every per-event TSV keyed `(csv, region)`, never region
  alone. That is the bug behind one retracted v1 figure.

---

## Build order and model tiers

| stage | where | tier | blocks |
|---|---|---|---|
| 0.1 ingest curation | local | cheap | 0.2 |
| 0.2 curated CSVs + BAM subset | local + pasteur | cheap | 0.3 |
| 0.3 NN class + TRASH-stratified split | local | **strong** | 1 |
| 1 trash probe / AUC gate | local | **strong** | 3, 4 |
| 2 sim-vs-real TRASH gap | local | **strong** | — |
| 3 spuriousness features + retrain | local GPU | **strong** | 4 |
| 4 conditioned NN prior | local | **strong** | 5 |
| 5 results + resume-6 | local | **strong** | — |

Commit after each stage. Never `git push`; never create or switch branches.
