# signal-native-v2 — START HERE (2026-09-02)

Resume document. Branch `signal-native-v2`, **nothing pushed** (per repo
convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-5.md` as the entry point.
`plans/signal-native-v2-trash-signal.md` (the plan) and
`plans/signal-native-v2-trash-results.md` (the full stage-by-stage record)
are both **implemented through Stage 2**; Stages 3–4 did not run (gated
off, see below). resume-5 and `plans/signal-native-v2-scale-union2-results.md`
still hold everything about the union2 scale-up and ensemble work; none of
that changed this session.

---

## The one thing to know

**A second curated real benchmark exists now, with real NN truth for the
first time — and the user's TRASH annotation, while it explains most of the
error, isn't cleanly recoverable from window-local signal at the bar this
session pre-registered.**

`benchmark_new/` (485 events: A3=33, A5=23, CSE=213, RI=57, **NN=159**) is
frozen and split (`benchmark_new/split.tsv`, stratified by `(truth, trash)`).
Its dev/test halves are **both distinct from `benchmark/`'s** — this is a
second, independent benchmark, not a replacement.

Stage 1's decision gate — pooled cluster-grouped logistic-regression AUC of
`trash` from `{intron_cov_frac, background_frac, antisense_frac, dup_rate,
unsupported_junction_frac, struct_span_frac, cov_cv}`, pre-registered at
~0.70 — came in at **0.612**. Per the plan's own discipline, Stages 3–4
(spuriousness event features, spuriousness-conditioned NN prior) **did not
run**. No model, checkpoint, or feature vector changed this session.

Stage 2 (sim-vs-real, independent of the gate) did run, and found something
worth carrying forward: sim's noise knobs already reproduce `real-clean`
closely but fall short of `real-TRASH` by **3.3x on intron_cov_frac, 2.3x on
novel_junction_rate and cov_median** — 26% of dev's real distribution (its
TRASH fraction) was never simulated. That's a specific, numbered target if
this branch ever re-simulates; it did not happen this session (user
decision, 2026-09-02).

## No v1-vs-v2 claim — still true, on both benchmarks

`benchmark/`'s test half (Stage 6, resume-4/resume-5) is spent: v1 60/70
dev + 63/70 test, v2 (E3) 54/70 dev + 60/70 test, McNemar p=0.549, no
parity, no win. `benchmark_new/`'s test half is **completely unspent** —
this session never read it. Stage 8 Track C's one-shot v1-vs-v2 spend
against the old benchmark's remaining potential is still owed and untouched;
nothing about `benchmark_new` changes that owed work, and `benchmark_new`'s
own test half is a *separate* future one-shot spend, not yet scheduled.

## What is now settled, and what it means for the next session

* **`benchmark_new/` is the right benchmark to reach for when NN recall
  matters.** `benchmark/` cannot measure it at all (no NN truth); this one
  can, and 33% of its truth is NN.
* **TRASH is a real, measured phenomenon — v1 six-net ensemble 58.0% clean
  / 18.3% TRASH, v2 70.4% clean / 24.6% TRASH on the 485 unambiguous rows —
  but it is not a phenomenon this session's 7-feature window-local metric
  block can turn into a per-event detector.** That doesn't mean the model
  can't be doing something more implicit with the same underlying
  information (attention over the coverage/junction tokens sees more than
  these 7 scalar summaries do) — it means a *simple, unfitted, hand-built*
  spuriousness score isn't the way in. If this is revisited, the more
  promising angle from what was measured here is Stage 2's sim-vs-real gap:
  widen the noise knobs toward the measured `real-TRASH` targets and see
  whether the *model* picks up a spuriousness-sensitive representation on
  its own, rather than hand-engineering one.
* **`nn_weight=0.1` is still not re-derived on any NN-containing benchmark.**
  `benchmark_new` now has both NN truth and a split, so
  `dsignal/prior_sweep.py` could re-derive the global constant on it directly
  — a small, separately-schedulable task, not blocked by the gate failure
  above (only the *conditioned* version was).
* **No re-simulation happened.** Stage 2's multipliers
  (`intron_cov_frac` needs 3.3x more, `--premrna-frac-{low,high}` is the
  knob) are a target, not a plan — turning them into one means checking
  `sim/build_experiment.py`'s current knob ranges against these numbers and
  deciding how much of the gap is worth chasing, which nothing in this
  session did.

## Open honesty caveats (carried forward, one new)

1. **The NN operating point — `nn_weight=0.1`, still not re-derived on any
   NN-containing set.** Unchanged this session; `benchmark_new` now makes
   the re-derivation possible.
2. **`DPSI_BETA`/`PSI_LEVEL_BETA` still fitted on `benchmark/`'s 140
   events.** Unrelated to and unaffected by this session.
3. **`benchmark_new`'s test half (243 events) is completely unspent.**
   Nothing in this session read it. `benchmark/`'s own test-half spend
   (Stage 8 Track C) is separately still owed.
4. **New: Stage 2's sim-vs-real TRASH gap is measured but not acted on** —
   `intron_cov_frac` 3.3x, `novel_junction_rate` 2.3x short of `real-TRASH`
   on dev. See `plans/signal-native-v2-trash-results.md`'s Stage 2 section
   for the full table.

## What is new in the code

| file | what |
|---|---|
| `benchmark_build/ingest_curation.py` | (new) `reviews/*.tsv` -> `curation.tsv`, 573 rows, all normalization rules |
| `benchmark_build/make_curated_csvs.py` | (new) `curation.tsv` -> `benchmark_new/{A3,A5,ES,IR,NN}.csv` + `events.bed` |
| `benchmark_build/trash_probe.py` | (new) Stage 1's AUC / cluster-grouped-logreg / vs-v2-correctness probe |
| `dsignal/score_benchmark.py` | `CSV_LABEL` gains `NN`; `report()` reads CSV stems and the NN-truth wording from the loaded rows instead of hardcoding the old 4-class benchmark |
| `dsignal/benchmark_split.py` | `make` gains `--trash-tsv`/`--group-col`-style stratification by `(truth, trash)`; unused, unchanged behavior |
| `dsignal/audit_noise.py` | `measure` gains `cov_cv`; `compare` gains `--group-tsv`/`--group-col`/`--half`/`--split` for the N-way sim/clean/TRASH table (`compare_grouped`) |

No `dsignal/features.py`, `dsignal/dataset.py`, `dsignal/model.py`,
`dsignal/predict.py`, `dsignal/train.py`, or `dsignal/prior_sweep.py`
changed this session — the gate failure means none of Stage 3/4's touch
points were exercised.

## Artifacts

| path | what |
|---|---|
| `plans/signal-native-v2-trash-results.md` | the full stage-by-stage record |
| `benchmark_new/{A3,A5,ES,IR,NN}.csv`, `events.bed`, `c1.bam`/`c2.bam`, `README.md`, `curation.tsv`, `split.tsv` | the frozen, curated, split, 5-class benchmark |
| `runs/noise_audit_bnew.tsv` | `audit_noise measure` over all 485 `benchmark_new` events |
| `runs/trash_probe_dev.log` | Stage 1's full AUC/logreg/vs-correctness output |
| `runs/noise_audit_bnew_grouped_dev.log` | Stage 2's full 3-way sim/clean/TRASH table |

## Running jobs on pasteur — read `CLAUDE.md` first

One ssh job this session: regenerating `benchmark_new/c1.bam`/`c2.bam` as a
`samtools view -L events.bed` subset of the existing `c1_new.bam`/`c2_new.bam`
merges (already on pasteur from Stage 8). Single-threaded `samtools view`,
no `limitram` needed for a job this small, but the merges themselves
(8.4/8.8 GB) stay on pasteur per the branch's usual data-doesn't-travel
convention — only the ~70/80 MB curated subset came back.

## Conventions

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar; 4–6 events is noise at n=140,
  8–10 at n=70. `benchmark_new` is n=485 total but its dev/test halves are
  ~242/243, so the same order-of-magnitude caveat applies once it's scored.
- Never delete a measured number from a docstring or plan doc — retractions
  go next to the number they retract. This session added one: the plan
  doc's own `1/0/NA = 160/380/17` all-573 tally does not sum to 573;
  `ingest_curation.py`'s `126+345+14=485` / `396+160+17=573` is internally
  consistent and is what `plans/signal-native-v2-trash-results.md` uses.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/rendering on
  pasteur.
