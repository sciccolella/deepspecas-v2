# signal-native-v2 — START HERE (2026-09-03, Stage 12)

Resume document. Branch `signal-native-v2`, **nothing pushed** (per repo
convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-8.md` as the entry point.
`plans/signal-native-v2-real-supervision.md` is the Stage 12 plan and
`plans/signal-native-v2-real-supervision-results.md` is its full
stage-by-stage record.

---

## The three things to know

**1. `benchmark_new`'s "descriptive, not pre-registered" caveat is retracted.**
The curator (the user) states the `v1_pred`/`v2_pred` columns were visible in
the review sheets but **did not inform the manual assessment**. So the
dev-half v1-vs-v2 result (`e4_union2` 58.7% vs v1 41.3% 5-class, p<0.0001) is
a real result, and **`benchmark_new`'s test half is a legitimate
pre-registered arena**. The caveats stay in the earlier docs with the
retraction beside them, per the never-delete convention.

Related, and stated by the user: **`benchmark/` is probably the *cleaner*
set**, not merely the older one — every one of its 140 events is a
curator-affirmed positive with IGV evidence retained, whereas `benchmark_new`
keeps the whole 573-candidate pool (26% TRASH, 159 NN-because-the-candidate-
failed). They trade off in opposite directions; `benchmark_new`'s one
structural advantage is having an NN class at all. See the three-benchmark
table in the plan.

**2. Track A is dead, and for a reason worth keeping.** Annotation-derived
labels cannot replace curation: the pre-registered gate failed at 0.607 non-NN
agreement (needed 0.80) and 0.314 NN recall (needed 0.50), with the same 0.614
on `benchmark/`. Not a tuning miss — the catalogue holds the correct label for
**97.5%** of non-NN dev windows, but ~2 *other* annotated events per window are
equally "present" by any BAM support criterion, capping a perfect
disambiguator at **0.804**. Curated-NN windows average 1.97 present events: a
curated NN is not a window without annotated splicing, it is one where a human
judged nothing convincing was happening, and no support threshold sees that.

**3. The coverage branch (Track D) is a reallocation, not an improvement.**
Three paired seeds: on `benchmark/` dev it looks like a large win (55.7 →
61.7 of 70, wins all three pairings), on `benchmark_new` dev it is a null
(121.7 → 122.7 of 242). It makes the same trade on both — buying A3/A5/CSE
and paying in **RI** (−0.167, −0.143) — and the verdict flips purely on class
mix. **NN recall, the reason it was built, does not move** (0.114 → 0.110).
Not adopted; the code stays, defaulted off.

## The methodological finding that outlives all three

**Baseline seed spread alone is 5 events at n=70 (53/56/58) and 10 at n=242
(117/127/122)** — the same magnitude as most "improvements" this branch has
ever compared. Seed 0 alone would have supported a +9-event claim at
p=0.023; three seeds cut it to +6 on one benchmark and +1 on the other.
**Seed-replicate any future comparison before believing it.**

## What is running right now (2026-09-03)

* **pasteur**: `benchmark_build/run_arena.sh ptbp1_work ptbp1_arena ptbp1`
  (salmon → SUPPA2 → rMATS → consensus → candidate CSVs). Log
  `ptbp1_arena.log`. Started ~09:37 after the fetch/align finished
  (`c1_ptbp1.bam`/`c2_ptbp1.bam` are written, 79G of FASTQ retained under
  `ptbp1_work/fastq`). **Check this first when resuming.**
* Nothing local.

## Track B state (the main remaining line)

* **B0 done, frozen in the plan**: PTBP1 KD `ENCSR064DXG`
  (SRR3469390/SRR3469391) vs its own ENCODE `possible_controls`
  `ENCSR603TCV` (SRR3469516/SRR3469517), HepG2, all verified against the
  ENCODE portal and NCBI SRA.
* **B1 done**: fetched and aligned on pasteur.
  `sim/fetch_align_benchmark_new.sh` now takes its runs and output prefix
  from the environment; unset, it reproduces the SRSF3 invocation exactly.
* **B2 in flight**: `benchmark_build/run_arena.sh` (new) scripts what Stage 8
  did interactively — an arena that cannot be rebuilt is not renewable.
* **B2 remainder, not yet written**: automated **NN candidates** (catalogued
  loci where both callers report clearly non-significant, near-zero dPSI,
  above an expression floor). The consensus pipeline emits none, so without
  this the arena has no NN class. Its noise level is unknown, unlike the
  measured 33% for positives — say so in the arena's README.
* **B3 not started**: freeze, commit, then one scoring pass, v1 six-net
  ensemble vs best v2, paired McNemar, per-dataset and pooled. Declare up
  front that absolute accuracy is attenuated by ~33% positive-label noise and
  **only the paired difference is interpretable**.
* Pairs 2-3 deliberately not frozen — pair 1's throughput and disk first.

## Numbers measured this stage, so they are not re-derived

From `benchmark_new/curation.tsv` (485 unambiguous rows): the rMATS × SUPPA2
consensus **class** is right for **93.3%** of events *that are real*, but only
**67.2%** of candidates are real, and **no** caller-statistic threshold
separates the two (purity flat at 0.66-0.73 across `rmats_pvalue`,
`|suppa_dpsi|`, `|rmats_incleveldiff|` and their conjunction). Candidate ==
truth is 76.2% on clean windows and **28.6%** on TRASH.

**C1 (Stage 11's open mechanistic question) is answered.**
`MIN_INTRON_FOR_COV` is eliminated (0-4% of drops). The window-edge rule is
the dominant filter but explains **A3/A5**, not CSE/NN — only 9-24% of real
A3/A5 windows yield a measurable `intron_cov_frac` at all, so Stage 11's
"A3 10x short" was never a calibration reading. Within a domain the metric is
largely set by window geometry (`icf × intronic/exonic ≈ 1` in all five
classes), which is the mechanism behind the `--by-label` fix. **But the
CSE/NN gap survives geometry matching** (CSE 2.9-9.4×, NN 4.1-8.0×), so Stage
11's central finding stands. The knob is sublinear because the pre-mRNA row is
a fraction of the gene's *reads* spread over pre-mRNA length, scaling its
effect by a per-gene length ratio — **the TRASH layer is revivable only by
solving per row for the target metric, never by raising the knob uniformly.**

## Open honesty caveats

1. **`nn_weight=0.1`** and **`DPSI_BETA`/`PSI_LEVEL_BETA`** — unchanged from
   resume-7/8.
2. **`benchmark_new`'s test half (243) is unspent and reserved** for one
   specific measurement: Track A's winning arm vs v1 (plan step A5). Track A
   died, so **that spend is not owed to anything right now** — do not spend it
   on an intermediate result. `benchmark/`'s test half stays spent (Stage 6).
3. **A1's thresholds were pre-registered; the sweep around them was inspected
   afterwards on dev.** Nothing was carried forward (the track stopped), so
   nothing is fitted — but if the route is revisited, pre-register a fresh
   criterion rather than reading one off that table.
4. **Track D's 88.1% dev figure must not be quoted as a headline.** It is what
   a class-mix-favourable benchmark says about a model that got worse at RI.

## Conventions (unchanged)

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar — **and now, seed replication**.
  n=70 needs 8-10 discordant events; n=140 5-6; n=242 roughly 10-14.
- Never delete a measured number from a docstring or plan doc — retractions
  and corrections go next to what they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/arena building on
  pasteur; wrap pasteur jobs in `limitram`, cap all four thread-pool env
  vars, `nice -n 19`, check `/proc/loadavg` before and after.
- Gates: `dsignal.verify`, `dsignal.verify_features` (**now six checks** —
  check 6 covers the coverage profile's symmetries), `sim.width_leak_check`,
  `sim.psi_recovery_check`, `dsignal.audit_noise compare --by-label`.
