# signal-native-v2 — START HERE (2026-09-02, Stage 11)

Resume document. Branch `signal-native-v2`, **nothing pushed** (per repo
convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-7.md` as the entry point.
`plans/signal-native-v2-trash-sim-results.md` is the full stage-by-stage
record of this session. `resume-7`/`signal-native-v2-bnew-dev-results.md`
(Stage 10) still hold everything about `benchmark_new` dev's first proper
scoring and the `nn_weight` sensitivity curve; nothing there changed this
session.

---

## The one thing to know

**Stage 11 (`plans/signal-native-v2-trash-sim.md`) stopped after Stage 2,
before any full re-simulation.** Two results, in order:

1. **A dev-only real-supervision ceiling (Stage 1) did not clear zero-shot
   NN recall** (11.4%/14.3% under the two truth conventions vs zero-shot
   `e4_union2`'s 13.9%/9.2%) — the plan's own trigger to reconsider whether
   more/better simulation is the right lever at all. n=242 is thin, so this
   is suggestive, not conclusive.
2. **The TRASH-simulation pilot (Stage 2) does not clear its pre-registered
   noise gate**, and the real finding is more specific than a knob value:
   RI is already well-calibrated (both clean and TRASH), but CSE/NN are
   genuinely short on `intron_cov_frac` and **do not respond proportionally
   to the same `premrna_frac` elevation that works for RI** — a mechanistic
   gap, not a magnitude one. See the results doc's stratified table.

**The TRASH layer itself is real, tested infrastructure** — a per-row
class-conditional (or uniform-control) trashy flag on its own RNG stream in
`build_experiment.py`, `--trash-label-policy nn` (a trashy positive becomes
NN in training while its underlying event is still simulated), all
reproducing bit-identically when off. A future session can reuse it once
the CSE/NN mechanism is understood — see "Next session" below.

## Everything else is unchanged from Stage 10

`benchmark_new` dev's proper scoring, the `nn_weight=0.1` deployment
decision, the v1-vs-v2 descriptive comparison caveats, and both benchmarks'
test halves (unspent) all carry forward exactly as `resume-7` describes them.
Nothing about the model, features, or deployed checkpoints changed this
session.

## New this session, beyond the TRASH layer

* **Two truth conventions are now dual-tracked everywhere**:
  `dsignal/score_benchmark.py --trash-as-nn` and
  `dsignal/benchmark_split.py compare --trash-as-nn` — any `curation.tsv`
  TRASH event's truth is forced to NN (dropping `trash == "NA"`), reported
  *alongside*, never instead of, the existing convention.
* **A dev-only real-supervision ceiling adapter**:
  `dsignal/real_supervision_ceiling.py` + `dsignal/train.py --dump-oof`.
  Diagnostic only — never scored on test, never ensembled, never a
  deliverable.
* **`audit_noise.py compare --by-label`**: stratifies the 4-way
  clean/TRASH × sim/real gate by class, because the pooled median is
  biased by class composition (RI's structural signal dominates whichever
  side has more of it). Use this by default from now on when reading any
  `intron_cov_frac`-based gate — the pooled number alone is misleading.

## Open honesty caveats (carried forward, one new)

1. **`nn_weight=0.1`**, **`DPSI_BETA`/`PSI_LEVEL_BETA`**, both test halves
   unspent — unchanged from `resume-7`, see there for detail.
2. **New**: the real-supervision ceiling (Stage 1) suggests NN's gap may be
   partly intrinsic to the feature representation/data scale rather than a
   pure sim-vs-real domain gap — this reframes how much upside any future
   TRASH-simulation effort can realistically buy, independent of whether
   Stage 2's CSE/NN mechanism gets fixed.

## Next session, if this is picked back up

Read `plans/signal-native-v2-trash-sim-results.md` first — its last section
lays out the specific mechanistic question (why `intron_cov_frac` doesn't
respond to `premrna_frac` for CSE/NN the way it does for RI) and candidate
causes (`MIN_INTRON_FOR_COV`/flank filtering, intron-length structure). That
question should be answered — cheaply, with data already in hand or a
targeted local check, not another blind pasteur pilot — before spending any
more simulation compute on this line. Independently, Stage 1's ceiling
result is worth weighing before committing to the TRASH-simulation approach
at all: if the ceiling doesn't move with more real supervision, no amount of
better-calibrated simulated TRASH will close the gap either.

## Conventions (unchanged)

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar.
- Never delete a measured number from a docstring or plan doc — retractions
  go next to the number they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/rendering on
  pasteur; wrap pasteur jobs in `limitram`, cap thread pools, `nice -n 19`,
  check `/proc/loadavg` before and after launching.
