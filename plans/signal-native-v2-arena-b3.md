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

All 2,422 events scored by both entrants; `(csv, region)` key sets identical
across all four TSVs (`runs/arena_v1_ensemble.tsv`,
`runs/arena_v2_{e4union2,e3bagged,e4union2_noprior}.tsv`). Gates: `dsignal.verify`
4,844/4,844, `verify_features` 6/6, regression guard on `benchmark/` dev
53/70 (unchanged).

### Headline — v1 six-net ensemble vs `e4_union2` at `--nn-weight 0.1`

Pooled McNemar on `correct_collapsed`, all 2,422 events:

**v1 881/2422 = 36.4%   v2 1175/2422 = 48.5%   base-only 213  new-only 507
McNemar p < 0.0001**

v2 wins pooled, decisively (n=2,422, 507 vs 213 discordant events dwarfs the
8-10-discordant-event power floor this branch uses at n=70).

Per-class recall (collapsed correctness, 5 truth strata) and per-class McNemar:

| class | n | v1 recall | v2 recall | base-only | new-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| A3 | 152 | 27.6% | 59.2% | 13 | 61 | <0.0001 |
| A5 | 113 | 36.3% | 60.2% | 9 | 36 | 0.0001 |
| CSE | 1073 | 47.6% | 62.3% | 106 | 264 | <0.0001 |
| **NN** | 800 | **4.1%** | **16.9%** | 28 | 130 | <0.0001 |
| RI | 284 | **89.4%** | 75.0% | 57 | 16 | <0.0001 |

macro-recall (unweighted over 5 classes): **v1 41.0%   v2 54.7%**

`benchmark_new`-mix reweighted accuracy (A3=6.8%, A5=4.7%, CSE=43.9%,
NN=32.8%, RI=11.8%): v1 36.4%, v2 48.6% — close to the unweighted pooled
number, since the arena's own mix is not wildly different.

**v2 wins every class except RI**, where v1 is the clear winner (89.4% vs
75.0%, McNemar p<0.0001 — v1's only per-class win here). **NN is v2's
largest per-class deficit by a wide margin**: 16.9% recall vs the
next-lowest (A3, 59.2%), a >40-point gap, even though v2 more than quadruples
v1's near-zero NN recall (4.1%).

### Secondary — the prior-free run (`--tau 0`, no `--target-prior`)

Reported alongside, never substituted for the headline:

v1 881/2422 = 36.4%   v2 (no prior correction) 1361/2422 = 56.2%,
McNemar p<0.0001. Per-class: A3 50.7%, A5 50.4%, CSE 54.7%, **NN 56.0%**,
RI 67.6%. Macro-recall 55.9%.

This is Trap 1 made concrete: at the as-deployed `nn_weight=0.1` (tuned on a
0%-NN benchmark), NN recall is suppressed to 16.9%; with no prior correction
at all on this 33%-NN set, NN recall jumps to 56.0% and stops being the
worst class (A5 is, narrowly, at 50.4%). The as-deployed headline is still
the number that matters for "how the model is actually run", but the
prior-free run shows the NN deficit above is partly an artefact of an
operating point tuned for a different class mix, not purely a modelling
failure.

### Secondary — `e3_bagged` (16 members, test-validated) at `--nn-weight 0.1`

v1 881/2422 = 36.4%   v2 (e3_bagged) 1201/2422 = 49.6%, McNemar p<0.0001.
Per-class: A3 52.0%, A5 56.6%, CSE 61.7%, NN 22.9%, RI 75.0%. Macro-recall
53.6%. Slightly ahead of `e4_union2` on NN recall (22.9% vs 16.9%) and on
pooled accuracy by ~1 point — consistent with `e3_bagged` being the
test-validated model, but not a large enough gap at this scoring depth to
change which model is descriptive vs headline; `e4_union2` remains the
headline entrant per the pre-registration above.

### Reading against the three traps

1. **Operating-point mismatch (Trap 1), confirmed and quantified**: NN
   recall at the deployed prior (16.9%) vs prior-free (56.0%) is a >3×
   difference from the operating point alone, on this class mix. Both are
   reported; the headline stays at the deployed point since that is the
   model as it would actually be used.
2. **Absolute accuracy is not quotable (Trap 2)** — both entrants' raw
   accuracy (36.4%, 48.5%) sit well below curated-benchmark levels for both
   models (v1 87.9%/140, v2 75-88% dev depending on ensemble), consistent
   with unmeasured label noise attenuating both. Only the **paired**
   difference (v2 wins by 12.1 points, p<0.0001) is being claimed.
3. **The 800-NN count (Trap 3)** — macro-recall (54.7% v2 vs 41.0% v1, an
   unweighted average that gives NN no more weight than any other class)
   and the `benchmark_new`-mix reweighted accuracy (48.6% vs 36.4%) both
   agree with the pooled headline in direction and rough magnitude, so the
   800-NN sampling choice is not manufacturing the result.

### Decision (Step 5, per the rule fixed above)

**v2 wins pooled, and NN recall is the largest per-class deficit for v2** →
per the pre-registered rule, **Stage 14 is the NN attack**: build arena pair
2 from a different KD experiment, use its caller-derived NN as hard
negatives for training/fine-tuning, hold pair 1 (this arena) as eval. Any
resulting NN-recall move must be checked against curated `benchmark_new` dev
(independent NN class) before being trusted, since this arena's NN purity is
unmeasured. `benchmark_new`'s test half stays reserved regardless.
