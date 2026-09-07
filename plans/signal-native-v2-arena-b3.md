# Stage 13 — B3 pre-registration and results

Pre-registered per `plans/signal-native-v2-arena-b3-plan.md` Step 0, written
and committed **before Step 2 (v1 scoring) or Step 3 (v2 scoring) produce
anything**. Nothing below this line, up to the "## Results" heading, may be
edited once scoring starts.

## The two entrants, named and pinned

- **v1**: the six-net ensemble, `dsEnsemble.py` with all six
  `data/trained_nets/<mode>/trained.pth`, `--zoom -1 --a --cse`. Same
  invocation as CLAUDE.md and `runs/v1_uncurated_new/`.
- **v2 headline**: `runs/ensembles/e4_union2.json` (6 members, `npz_union2`),
  `--target-prior nn-weight --nn-weight 0.1`, `--tta-condition-swap`. Best
  model by dev evidence on `benchmark/`, a different dataset from the arena —
  a legitimate held-out entry here.
- **v2 secondary** (descriptive only, never promoted to headline post hoc):
  `runs/ensembles/e3_bagged.json` (16 members, the test-validated one), same
  flags; and `e4_union2` with the prior correction off (no `--target-prior`,
  i.e. `--tau 0` / raw posterior).

## The set

`ptbp1_arena/`, all 2,422 events: A3=152, A5=113, CSE(`ES.csv`)=1073,
RI(`IR.csv`)=284, NN=800. Model-blind, unfrozen at build time, automated
(rMATS × SUPPA2 consensus + hard-negative NN), not curated. Full caveats in
`ptbp1_arena/README.md`.

## Headline metric (fixed now)

Pooled McNemar on `correct_collapsed` (v1 never distinguishes A3 from A5, so
comparison is over the four effective classes A / CSE / NN / RI), all 2,422
events, v1 six-net ensemble vs `e4_union2` at `--nn-weight 0.1`.

## Secondary metrics (fixed now)

- Per-class recall table (5 classes) for both entrants.
- Macro-recall (unweighted mean of the 5 per-class recalls) — the number that
  is not a function of the arbitrary 800-NN sampling count.
- Per-class McNemar, 5 tests — reported together, multiplicity noted, no
  cherry-picking one class's p-value as if it were the headline.
- `e4_union2` at `--tau 0` (no prior correction) — prior-free secondary,
  reported alongside the headline, never substituted for it after seeing which
  wins.
- `e3_bagged` at `--nn-weight 0.1` — descriptive, test-validated model shown
  for context.
- `benchmark_new`-class-mix re-weighted accuracy (re-weight the arena's
  per-class recalls to `benchmark_new`'s class proportions) — a sensitivity
  check on Trap 3, not a replacement for macro-recall.

## The three traps, fixed as caveats that ship with every number

1. **Operating point mismatch.** v2's `nn_weight=0.1` was tuned on
   `benchmark/`, which is 0% NN; the arena is 33% NN. The headline is scored
   at the *as-deployed* point anyway (v1 has no comparable knob to match), and
   the `--tau 0` prior-free run is reported alongside as the honest
   alternative — never substituted in after seeing which wins.
2. **Absolute accuracy is not quotable.** By analogy to `benchmark_new`
   (67.2% candidate purity), the arena's positive-side purity is presumed
   comparable and its NN-side purity is **unmeasured** — no curated set of
   automated NN exists to check it against. Label noise pushes both models
   toward chance and *attenuates* the paired difference, so a null here is
   weaker evidence than a null on curated data. Only the **paired**
   v1-vs-v2 difference is meaningful; neither entrant's raw accuracy is a
   claim about real-world performance.
3. **The 800-NN count is a construction choice** (proportional allocation to
   the positive mix, not a natural base rate). Per-class recall + macro
   average are the substantive numbers; pooled accuracy/McNemar is reported
   too (it is still the right test for "which model is better on this exact
   set"), but is not to be read as an external NN prevalence estimate.

## Decision rule for Stage 14 (fixed now, evaluated in Step 5)

- **v2 wins pooled, and NN recall is the largest per-class deficit for v2**
  → Stage 14 is the NN attack: build pair 2 of the arena from a different KD
  experiment, use its NN as hard negatives for training/fine-tuning, hold
  pair 1 as eval. Check any resulting NN-recall move against curated
  `benchmark_new` dev before trusting it (arena NN purity is unmeasured).
- **v2 wins pooled, but NN is *not* the largest per-class deficit** → the
  three-failure NN narrative is wrong about where the gap is; Stage 14
  attacks whichever class the arena shows as the largest deficit instead.
- **v1 wins pooled, or the pooled comparison is null (McNemar p > 0.05)** →
  stop attacking and consolidate. Track A's 0.804 cap, Track D's
  reallocation-not-improvement, and the seed-variance finding are the
  write-up.

Every subsequent stage still obeys the standing rules: seed-replicate before
believing any A/B (baseline spread is 5 events at n=70, 10 at n=242), and
`benchmark_new`'s test half stays reserved.

## Results

*(filled in after Step 4; nothing above this heading is to be edited once
scoring starts)*
