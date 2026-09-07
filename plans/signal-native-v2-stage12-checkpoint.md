# signal-native-v2 — Stage 12 checkpoint (2026-09-02 .. 2026-09-07)

> **Resuming? Read `plans/signal-native-v2-resume-9.md` first** for direction
> and open caveats. **This file is the chronological build log** — what was
> built, how it was verified, what broke and why. Read it for mechanism.
>
> Companion documents: `plans/signal-native-v2-real-supervision.md` (the plan),
> `plans/signal-native-v2-real-supervision-results.md` (the results record).
> Nothing in this session has been pushed; 15 commits sit on `signal-native-v2`.

---

## What this session was

The user asked to plan next steps with one constraint: **no new manual
curation** — it is extremely labour-intensive and partly interpretive, so it is
off the table as a source of data. A four-track plan was written, approved, and
then (unusually for this repo's convention) implemented in the same session at
the user's explicit instruction.

Two premises the user corrected mid-session, both load-bearing:

1. **The `benchmark_new` "curation contamination" caveat is wrong.** The
   `v1_pred`/`v2_pred` columns were visible in the review sheets but did not
   inform the manual assessment. The caveat is retracted (kept in place with
   the retraction beside it, per convention) — so `benchmark_new` dev's
   v1-vs-v2 result is real and its test half is a legitimate pre-registered
   arena.
2. **`benchmark/` is probably the *cleaner* set, not merely the older one.**
   Every one of its 140 events is a curator-affirmed positive with IGV evidence
   retained; `benchmark_new` keeps the entire 573-candidate pool (26% TRASH,
   159 NN-because-the-candidate-failed). They trade off in opposite directions.

## Outcome in one paragraph

Track A (real supervision without curation) **died at its pre-registered
gate**, for a reason worth more than a pass would have been. Track C's
close-outs are **done**, including the mechanistic question Stage 11 left open.
Track D's coverage branch was built and measured across three seeds and is **a
reallocation across classes, not an improvement** — not adopted. Track B
produced **a renewable, model-blind, 2,422-event arena** with both positives
and NN, fully scripted; its single scoring pass (B3) has **not** been run.
Threading through all of it: **baseline seed spread alone is 5 events at n=70
and 10 at n=242**, the same magnitude as most comparisons this branch has ever
made.

---

## Track A — annotation-derived labels instead of curation

**Idea.** Curators were judging *differential usage*; the model's five classes
are *event types*, which `sim/events/*_strict.ioe` already defines
geometrically. So labels might need no curation — only a decision about which
catalogued event a window carries and whether it is present.

**A0 — `sim/real_label.py`** (new). SUPPA `event_id` → the two competing forms
of the event as intron sets (SE inclusion pair vs exclusion junction; A3/A5 the
two junctions sharing a donor/acceptor; RI the spliced junction vs intronic
coverage, since the retained form has no junction). A `CatalogueIndex` over
92,178 catalogued events queried by window overlap. Presence = **both**
competing forms carry BAM evidence, i.e. *alternatively spliced in this sample*,
not merely annotated or expressed.

Two design points worth keeping:

* **The annotation→observed coordinate offset is calibrated, not trusted.**
  `calibrate_offset` scores candidates by how many catalogued junctions they
  match: `(+1,−1)` wins by >100× (1,529 matches vs 7/3/0). A silent off-by-one
  here would have been indistinguishable from "the annotation doesn't fit the
  data" — exactly the wrong conclusion to reach by accident.
* **A window with no reads abstains rather than becoming NN.** "No reads" is
  not evidence of "no event", and free NN acquired that way trains a coverage
  detector.

**A1 — `sim/real_label_gate.py`** (new). Runs the labeller over curated windows
and compares to curated truth. Criteria declared in the plan before running:
non-NN agreement ≥ 0.80, NN recall ≥ 0.50 on clean windows.

**Result: failed both.** `benchmark_new` dev 0.607 / 0.314; all 485 rows
0.620 / 0.229; `benchmark/` 0.614 — so not a `benchmark_new` artifact.

**Why, decomposed on dev non-NN rows:**

| stage | fraction |
|---|---:|
| correct label present in the catalogue at all | **0.975** |
| …and among events passing the presence filter | **0.804** |
| …and chosen by the assignment rule | 0.607 |
| …chosen when abstentions are broken by support | 0.712 |

The annotation is **not** the bottleneck. Windows carry 2.9–5.5 candidate
events and ~2 pass presence; **curated-NN windows average 1.97 present events**
with only 19% having zero. A curated NN is not a window without annotated
splicing — it is one where a human judged nothing convincing was happening, and
no support threshold sees that distinction. Fifteen rule/threshold combinations
were tried (uniqueness, dominance ratios, `min_reads` 2→30); best non-NN
agreement 0.762, and the two criteria trade against each other roughly
one-for-one. **A perfect disambiguator is capped at 0.804.**

**Stopped at A1 per the plan's own rule.** A2–A5 never ran: no real-supervision
pool exists, no pasteur time was spent on this track, and `benchmark_new`'s test
half stayed unspent (A5's spend was conditional on a Track A winner).

One bug found and fixed while gating: `split.tsv` is keyed by the *curated*
stem, not `csv_source` (the pre-curation candidate class). Keying on the latter
silently matched only the 304 rows where the two agree and dropped **every** NN
row.

---

## Track C — close-outs

**C0.** Retracted the contamination caveat in
`signal-native-v2-bnew-dev-results.md` (3 places), `resume-7` (3), `resume-8`
(1), with the curator's attestation beside each — never deleting the original.

**C1 — Stage 11's open mechanistic question, answered locally** (no pasteur
time), from `npz_union2` (n≈2,000 sampled) vs `benchmark_new`'s 485:

* **`MIN_INTRON_FOR_COV` is eliminated** — short introns are 0–4% of drops in
  every class.
* **The window-edge rule is the dominant filter but explains A3/A5, not
  CSE/NN.** It drops 42–92% of supported junctions; **only 9–24% of real A3/A5
  windows yield a measurable `intron_cov_frac` at all**, so Stage 11's "A3 10×
  short" was never a calibration reading. Relaxing the rule moves A3 21.7→9.6×
  and A5 7.8→3.3×, but CSE 4.05→3.90 and NN 4.68→4.60.
* **Within a domain the metric is largely set by window geometry** —
  `icf × (intronic/exonic territory) ≈ 1` in all five classes (RI 1.12, NN 0.64,
  CSE 1.06, A3 0.93, A5 1.04). This is the mechanism behind Stage 11's
  `--by-label` fix.
* **But the CSE/NN gap survives geometry matching**: binning sim and real by
  intronic fraction leaves CSE 2.9–9.4× and NN 4.1–8.0×. Stage 11's central
  finding stands.
* **Why the knob is sublinear (hypothesis, not established):** the pre-mRNA row
  is a fraction of the gene's *reads* (`sim/abundance.py:_add_low_abundance_row`)
  spread over the pre-mRNA's length, so its effect on the metric scales by a
  per-gene `L_mRNA/L_preMRNA` ratio. RI's retained intron is a genuine annotated
  isoform set by PSI, which is why it responds and CSE/NN do not. Consistent
  with Stage 11's own numbers: 3.5× more `premrna_frac` gave 1.29× in CSE's
  `intron_cov_frac`.
* **Verdict: the TRASH layer is revivable only by solving per row for the target
  metric**, never by raising the knob uniformly. Shelved until then.

**C2.** Recorded the consensus-purity measurements (see the numbers table
below) so they are not re-derived.

**C3.** `benchmark_new/README.md` now records that the test half is reserved
and why, plus the clean-vs-dirty comparison against `benchmark/`.

---

## Track D — the coverage-profile branch

Chosen by the user after Track A's death removed D's original gate.

**Built.** `features.build_coverage_profile` → `[2, bins]` depth-normalized log
coverage per condition. **Bin edges are fractional** and each bin is the exact
integral of the piecewise-constant coverage over it, so mirroring a record
reverses the profile **bit-for-bit** (measured error 0.0); integer edges break
that at odd window lengths and the flip augmentation would quietly stop being an
involution. `model.CoverageEncoder` is a dilated 1D conv stack (1/2/4/8) with
mean+max pooling, **added into the CLS embedding** rather than appended as a
token — CLS is never masked and is what the heads read.

`ModelConfig.coverage_bins` defaults to 0, so every pre-Stage-12 checkpoint
loads and behaves identically (verified: `geomwin_iv30_refit` still scores
exactly 53/70). The profile is built from the *augmented* `NormalizedRecord`, so
flip, condition_swap, jitter, subsample and condition-swap TTA all apply to it
without the augmentations knowing the branch exists. `model.batch_inputs` routes
the optional tensor from `collate` to the model. `verify_features` gained
**check 6** (mirror and condition-swap symmetries, exact equality).

**Measured — three paired seeds, base recipe on `npz_geomwin`:**

| | `benchmark/` dev (n=70) | `benchmark_new` dev (n=242) | in-sim macro-F1 |
|---|---|---|---|
| base | 53 / 56 / 58 → **55.7** | 117 / 127 / 122 → **121.7** | 0.8604 |
| +coverage | 62 / 63 / 60 → **61.7** | 126 / 121 / 121 → **122.7** | 0.8671 |
| paired McNemar | p = 0.023 / 0.119 / 0.754 | p = 0.12 / 0.41 / 1.00 | — |

**It is a reallocation.** The same trade on both sets (three-seed mean): A3
+0.167/+0.000, A5 +0.111/+0.167, CSE +0.101/+0.031, **RI −0.167/−0.143**, NN
−0.004. `benchmark/` dev is 89% A3/A5/CSE and 11% RI so the trade nets +6;
`benchmark_new` dev is 33% NN so it nets +1/242. **NN recall — the reason the
branch was built — does not move (0.114 → 0.110).** Hypothesis for the RI loss,
untested: a simulated retained intron is a separate annotated isoform with
near-uniform intronic coverage by construction, while a real one is ragged, so a
branch reading coverage shape learns sim-RI's signature and mis-transfers.

**Not adopted.** Code stays, defaulted off. **The 88.1% dev figure must not be
quoted** — it is what a class-mix-favourable benchmark says about a model that
got worse at RI.

One bug: the refit loop's forward call was formatted differently from the other
three and was missed, so the first run trained fold 0 then died at refit.

---

## Track B — the renewable model-blind arena

**B0.** Accessions frozen in the plan *before* any scoring, verified against
the ENCODE portal and NCBI SRA (not quoted from memory): PTBP1 KD
`ENCSR064DXG` (SRR3469390/SRR3469391) vs its own ENCODE `possible_controls`
`ENCSR603TCV` (SRR3469516/SRR3469517), HepG2. PTBP1 chosen for a strong,
well-characterized exon-skipping phenotype distinct from SRSF3.

**B1.** `sim/fetch_align_benchmark_new.sh` parameterized — runs and output
prefix from the environment; unset, it reproduces the SRSF3 invocation exactly.
Fetch + STAR on pasteur, ~2 h.

**B2.** `benchmark_build/run_arena.sh` (new) scripts what Stage 8 did
interactively: salmon → TPM matrices → SUPPA2 psiPerEvent → diffSplice → rMATS
→ consensus → candidate CSVs, every stage skipped when its output exists so a
failure resumes in place. **An arena that cannot be rebuilt is not renewable,
and renewable is the entire point.**

### Three build failures, all fixed in the committed scripts

1. **rMATS: `ModuleNotFoundError: rmatspipeline`.** The compiled module lives in
   the rMATS *install* directory, not the env's site-packages, and `rmats.py`'s
   shebang is a bare `#!/usr/bin/env python` which that env does not provide
   (only `python3`/`python3.12`). Fixed with an explicit interpreter +
   `PYTHONPATH`. Cost: the run sat dead for two days.
2. **SUPPA2 diffSplice exits 0 leaving `.dpsi.temp.0`** and never renames it on
   this install (Stage 8's run did get the rename). The temp files were complete
   — identical row counts to the reference, log says "Done!" — differing only by
   an extra leading `Event_id` header cell, which `consensus.py`'s
   `read_csv(...).reset_index()` cannot take. Recovery strips that one cell, and
   runs **before** the diffSplice loop as well as after, since the loop skips on
   `.dpsi` — otherwise a resumed run recomputes every empirical p-value.
3. **`make_candidate_csvs.py` did not create its output directory**, so the run
   completed rMATS and the consensus and then died writing `candidates/A3.csv`.

### The NN half — `benchmark_build/make_nn_candidates.py` (new)

The consensus pipeline emits only positives, so the arena had no NN class. This
inverts the criterion using the same two callers: **a locus both see and both
call clearly unchanged**, in a demonstrably expressed gene — deliberately *hard*
negatives (a real catalogued event with real splicing at it, no differential
usage), which is what `sim`'s `nn_hard` models and what most of
`benchmark_new`'s curated NN turned out to be.

    SUPPA2:  dPSI measurable (not NaN), p > 0.20, |dPSI| < 0.05
    rMATS:   PValue > 0.20, |IncLevelDifference| < 0.05
    matched at one locus by overlap (TOLERANCE=25, as for positives)
    expressed: summed transcript TPM >= 1.0 in both conditions, every replicate
    and not overlapping any positive candidate

800 sampled (seed 0) from pools of 1,652 / 1,000 / 6,159 / 903, allocated
**proportionally to the positive class mix** so widths follow the positives'
distribution by construction.

**Two traps caught before anything was scored:**

1. **The originally pre-registered `p > 0.5` selects the empty set**, for a
   structural reason worth remembering: **`p == 1.0` in a SUPPA2 `.dpsi` means
   "not measurable", not "measured and unchanged"** — `dPSI` is NaN for exactly
   those rows and no others (A3 6,684/19,347; SE 17,501/47,833; RI 1,963/7,322),
   and a measurable permutation p-value never exceeds ~0.485. Amended to
   `p > 0.20` plus an explicit non-NaN guard, recorded as a dated amendment in
   the script. Also fixed an expression floor of my own that required *every*
   transcript ≥ 1 TPM rather than their sum.
2. **The BAM subsets had been cut over the positives-only BED**, leaving 11 of
   the first 12 NN windows at **zero coverage**. Every NN would have scored
   correct for a model that had learned nothing but a coverage detector — the
   exact hazard the script's own docstring warns about. Recut over all 2,422
   regions (`candidates/all_events.sorted.bed`).

### Leak checks — measured, not assumed

| shortcut | arena | curated `benchmark_new` |
|---|---:|---:|
| width alone (AUC) | **0.493** (p=0.57, KS p=0.34) | — |
| mean coverage alone (AUC) | 0.417 (p=0.004) | 0.564 (p=0.050) |

No width leak. The coverage skew is small and — the interesting part — the
**same magnitude but opposite sign** as in curated data: curated NN skews to
*higher* coverage (a curator looking at a busy window and judging nothing
convincing), automated NN slightly lower (callers have more power at depth). A
model leaning on depth is rewarded on one set and punished on the other.

### Final state

`ptbp1_arena/`: **2,422 events** — 1,073 CSE / 800 NN / 284 RI / 152 A3 /
113 A5. `dsignal.verify` **4,844/4,844** on coverage and junction parity. BAM
subsets (881 MB) gitignored; `consensus.tsv`, the candidate CSVs and BEDs are
committed. **B3 has not run.**

---

## Every number this session produced

**Consensus label quality** (from `benchmark_new/curation.tsv`, 485 unambiguous
rows) — the reason curation cannot simply be automated away:

| quantity | value |
|---|---:|
| consensus class correct, conditional on the event being real | 304/326 = **93.3%** |
| consensus candidates that are real events | 326/485 = **67.2%** |
| candidate == truth, all rows | 62.7% |
| … clean windows (n=345) | 76.2% |
| … TRASH windows (n=126) | **28.6%** |

Caller statistics **do not** separate real from NN: purity is flat at 0.66–0.73
across every threshold on `rmats_pvalue`, `|suppa_dpsi|`, `|rmats_incleveldiff|`
and their conjunction (`|dpsi|≥0.3 & |dinc|≥0.2 & p≤0.01` → n=96, purity 0.708).
Median `rmats_pvalue` is *lower* for the NN group.

**Seed variance — the methodological finding that outlives every track.**
Baseline spread alone is **5 events at n=70** (53/56/58) and **10 at n=242**
(117/127/122), the same magnitude as most improvements this branch has ever
compared. Seed 0 alone would have supported a +9-event claim at p=0.023; three
seeds cut it to +6 on one benchmark and +1 on the other. **Seed-replicate before
believing an A/B.**

---

## Files

**New:** `sim/real_label.py`, `sim/real_label_gate.py`,
`benchmark_build/run_arena.sh`, `benchmark_build/make_nn_candidates.py`,
`ptbp1_arena/` (README, consensus.tsv, candidates/), `plans/` × 3 (plan,
results, resume-9), this file.

**Modified:** `dsignal/features.py` (`build_coverage_profile`),
`dsignal/model.py` (`CoverageEncoder`, `batch_inputs`, `_cls_embedding`,
`ModelConfig.coverage_bins`), `dsignal/dataset.py` (`coverage_bins`, collate),
`dsignal/train.py` (`--coverage-bins`, all four forward sites),
`dsignal/predict.py` (coverage through TTA views + checkpoint-derived
`coverage_bins`), `dsignal/verify_features.py` (check 6),
`sim/fetch_align_benchmark_new.sh` (env-parameterized),
`benchmark_build/make_candidate_csvs.py` (mkdir), `.gitignore`, `CLAUDE.md` and
`plans/README.md` (entry-point pointers), plus the three docs carrying the C0
retraction.

**Gates all pass:** `dsignal.verify` (both benchmarks and the arena),
`dsignal.verify_features` (now 6 checks).

## Operational state

* **pasteur is idle** of this branch's work. `ptbp1_work/fastq` holds 79 GB of
  FASTQ and `ptbp1_work/align` the merged BAMs if a rebuild is needed; `/data`
  was at 85% (1.5 TB free) when last checked. Load stayed ≤ 30 throughout.
* **Locally**: `ptbp1_arena/c{1,2}.bam` (881 MB, gitignored),
  `ptbp1_arena/build/` (31 MB of intermediates, gitignored),
  `runs/geomwin_cov128*`, `runs/geomwin_base_s{1,2}` checkpoints (gitignored per
  the existing `runs/*/` rule; their JSON and logs are committed).
* **Nothing pushed.** 15 commits from `e143ad7f` to `b02595a0`.

## Open questions for the next session to plan against

1. **B3: spend the arena's single use, or hold it?** Running it gives a
   v1-vs-v2 paired McNemar across all five classes including NN — the
   comparison neither existing benchmark offers cleanly. But the "best v2" that
   would be entered (`e4_union2` / `e3_bagged`) predates this whole stage, since
   neither A nor D produced a model worth promoting. It may be worth waiting for
   a new deliverable.
2. **NN remains the whole problem, and three separate attacks have now failed
   to move it**: Stage 11's real-supervision ceiling (11.4%/14.3% at n=242),
   Track A (cannot even label it automatically), Track D (0.114 → 0.110). That
   is worth treating as evidence about the problem, not just three unlucky
   attempts.
3. **The unmeasured quantity that would inform everything**: the NN half of the
   arena has no purity estimate, unlike the positives' 67.2%. A small curation
   of automatically-derived NN would give one — but the user has ruled new
   manual curation out, so any estimate has to come from somewhere else.
4. **`benchmark_new`'s test half (243 events) is unspent and now owed to
   nothing**, since A5's spend was conditional on a Track A winner.
   `benchmark/`'s test half stays spent (Stage 6).
5. **Pairs 2–3 of Track B are unfrozen** — the pipeline now runs end to end, so
   a second arena is cheap (~4 h wall clock, mostly unattended).
6. Carried unchanged: `nn_weight=0.1`, and `DPSI_BETA`/`PSI_LEVEL_BETA` still
   fitted on `benchmark/`'s 140 events.
