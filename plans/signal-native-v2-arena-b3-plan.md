# Stage 13 — B3: score the PTBP1 arena, then decide

## Context

Stage 12 ended with three things: Track A dead at its gate, Track D measured as
a reallocation and not adopted, and a **built, verified, unspent 2,422-event
model-blind arena** (`ptbp1_arena/`). Its open question was whether to spend it.

Two facts settle that in favour of spending it now:

1. **It is renewable.** `benchmark_build/run_arena.sh` + `make_nn_candidates.py`
   now run end to end, and pairs 2–3 are unfrozen at ~4 h mostly-unattended
   wall clock. "Spending the arena's single use" costs a rebuild, not a
   permanent asset — so the reluctance recorded in `resume-9` should be
   discounted.
2. **It is the only set that can compare v1 and v2 on NN.** `benchmark/` has no
   NN at all; `benchmark_new` has one but at n=242 dev with a spent-or-reserved
   test half. The arena gives n=2,422 with 800 NN — roughly 10× the discordant
   power of anything this branch has measured, on the exact class that three
   separate attacks (Stage 11, Track A, Track D) have failed to move.

The user's stated endgame is **"one more genuine attempt at NN, then
consolidate"**. B3 is the measurement that tells us what that attempt should be
— or whether NN is already less of the gap than assumed.

Scope of this plan: **B3 only**, plus a pre-registered decision rule for what
follows. It is a measurement stage; no model changes.

## What must be true before any number is looked at

The branch's convention is pre-registration, and this stage has three specific
traps that only pre-registration defuses.

**Trap 1 — the operating point is mis-specified for this set.** v2's deployed
`--nn-weight 0.1` was tuned on `benchmark/`, which contains **zero** NN events.
The arena is **33% NN**. Scoring v2 at 0.1 will suppress NN recall; scoring it
at a mix-matched prior would be tuning on the arena. Resolution: **the headline
is the as-deployed operating point** (`--target-prior nn-weight --nn-weight
0.1`), because that is the model as it would actually be used and v1 has no
comparable knob at all. A prior-free run (`--tau 0`, no correction) is a
pre-registered **secondary**, reported alongside, never substituted for the
headline after seeing which one wins.

**Trap 2 — absolute accuracy is not quotable.** By analogy to `benchmark_new`'s
curation, ~67% of consensus candidates are real events, and the NN half's purity
is **unmeasured**. Label noise pushes both models toward chance and *attenuates*
the observed difference — so a null here is weaker evidence than a null on
curated data, and this must be stated with the result, not discovered afterward.
Only the **paired** v1-vs-v2 difference is meaningful.

**Trap 3 — the 800 NN count is a construction choice.** Pooled accuracy is a
function of an arbitrary sampling decision. So report **per-class recall** and
a **macro average** as the substantive numbers, with pooled McNemar as the
significance test, plus a sensitivity re-weighting to `benchmark_new`'s class
mix.

Also fixed before scoring: **v1 never distinguishes A3 from A5** (`--a --cse`
is baked into its predictions), so every comparison is on `correct_collapsed`
over the four effective classes A / CSE / NN / RI. This is what
`dsignal/v1_reference.py` already documents and enforces.

## The plan

### Step 0 — pre-register, commit, then stop looking at it (strong model)

Write `plans/signal-native-v2-arena-b3.md` containing, *before any scoring
command runs*:

- the two entrants, named and pinned (see Step 1);
- the headline metric: pooled McNemar on `correct_collapsed`, all 2,422 events,
  at `--nn-weight 0.1`;
- the secondary metrics: per-class recall table, macro-recall, per-class
  McNemar (5 tests — note the multiplicity, do not cherry-pick), the `--tau 0`
  prior-free run, and the `benchmark_new`-mix re-weighted accuracy;
- the three traps above as caveats that ship with the number;
- the **decision rule for Stage 14** (Step 5), written now so the result cannot
  choose it.

Commit this before Step 2 produces anything.

### Step 1 — pin the entrants (strong model)

- **v2 headline: `runs/ensembles/e4_union2.json`** (6 members, `npz_union2`).
  It is the best model by the only dev evidence available, and the arena is a
  fresh set, so entering it here is a legitimate held-out evaluation — its
  dev-tuning was on `benchmark/`, a different dataset.
- **v2 secondary: `runs/ensembles/e3_bagged.json`** (16 members) — the
  test-validated one. Reported descriptively; it is *not* an alternative
  headline to be promoted if it wins.
- **v1: the six-net ensemble**, `data/trained_nets/<mode>/trained.pth` for all
  six modes, `--zoom -1 --a --cse` — the same invocation as CLAUDE.md's and as
  `runs/v1_uncurated_new/`.

### Step 2 — run v1 over the arena (cheap model, mechanical; strong model for the pilot)

`dsEnsemble.py` renders six representations per event on GPU. The arena's width
distribution is nearly identical to `benchmark/`'s (median 3,226 vs 3,062 bp,
p90 19,093 vs 19,122, max 252 kb vs 182 kb), so per-event cost carries over —
but it is **17× the events** and 4.2× `benchmark_new`'s 573.

1. **Timed pilot first**: 20 events from `ES.csv`, wall-clocked, to extrapolate.
   If the full run projects past ~6 h, use `--shard i/N` (event-level, every CSV
   line independent, reassembled by the `ix` prefix) across the local GPU.
2. Full run, in the `DSv1` env, one invocation per class CSV **including
   `NN.csv`** (`runs/v1_uncurated_new/` has no `pred_NN.csv` — the arena needs
   one), writing `runs/v1_arena/pred_{A3,A5,ES,IR,NN}.csv` + `.log`.
3. `python -m dsignal.v1_reference --glob "runs/v1_arena/pred_*.csv" --out
   runs/arena_v1_ensemble.tsv`. Use `--glob`, **not** `--curation` — the arena
   has no curation ledger and its candidate stem *is* its truth stem, which is
   exactly the condition `--glob` requires.

Note `--gpu-render` is not bit-exact with the default PIL path; do not use it
for a number that will be quoted.

### Step 3 — run v2 over the arena (cheap model, mechanical)

In `DSnet`, `--csv` takes arbitrary paths and `--half both` bypasses
`benchmark/split.tsv` entirely (`benchmark_split.filter_rows` returns all rows
for `both`), so no split file is needed:

```bash
python -m dsignal.score_benchmark \
  --ensemble runs/ensembles/e4_union2.json \
  --csv ptbp1_arena/candidates/{A3,A5,ES,IR,NN}.csv \
  --half both --tta-condition-swap \
  --target-prior nn-weight --nn-weight 0.1 \
  --train-labels sim/experiment/labels_union2.tsv \
  --out runs/arena_v2_e4union2.tsv
```

Repeat for: `e3_bagged` (secondary), and `e4_union2` with the prior correction
off (the `--tau 0` / no-`--target-prior` secondary). Confirm the `--train-labels`
path matches what `e4_union2` was trained on before running — `union2`, not
`geomwin`. Extraction dominates the cost and `dsignal.verify` already passes
4,844/4,844 on this arena, so no surprises are expected here.

### Step 4 — compare (strong model)

`python -m dsignal.benchmark_split compare runs/arena_v1_ensemble.tsv
runs/arena_v2_e4union2.tsv --half both` for the pooled headline, then the
per-class breakdown. Per-class McNemar and the macro-recall / re-weighting
tables are small additions; check whether `benchmark_split.cmd_compare` already
emits a per-class split before writing anything new — if it does not, add a
`--by-class` flag there rather than a new script, so every comparison on this
branch keeps going through one code path.

Seed replication is **not** required for this stage: both entrants are fixed
checkpoints and nothing is trained. The 5-events-at-n=70 spread applies to the
*next* stage, not this one.

### Step 5 — the pre-registered decision rule for Stage 14

Written in Step 0, evaluated here:

- **If v2 wins pooled and NN recall is the largest per-class deficit** → the
  "one more genuine attempt" is the NN attack, and the arena's own renewability
  supplies it: build **pair 2** from a different KD experiment and use its
  caller-derived NN as real hard negatives for training/fine-tuning, holding
  pair 1 as eval. This sidesteps the curation constraint that killed Track A.
  Its known risk — the NN half's unmeasured purity — is partly addressable by
  checking whether training on pair-2 NN moves NN recall on *curated*
  `benchmark_new` dev, which has an independent NN class.
- **If NN is *not* the largest deficit** → the three-failure narrative is wrong
  about where the gap is, and Stage 14 attacks whatever the arena says instead.
- **If v1 wins or the comparison is null** → stop attacking and consolidate.
  The negative results (Track A's 0.804 cap, Track D's reallocation, the
  seed-variance finding) are substantial and are the write-up.

In all three branches, Stage 14 obeys the standing rules: **seed-replicate
before believing any A/B** (baseline spread is 5 events at n=70, 10 at n=242),
and `benchmark_new`'s test half stays reserved — it is owed to nothing and must
not be spent on an intermediate result.

## Files

**New:** `plans/signal-native-v2-arena-b3.md` (pre-registration + results),
`runs/v1_arena/pred_*.csv`, `runs/arena_v1_ensemble.tsv`,
`runs/arena_v2_{e4union2,e3bagged,e4union2_noprior}.tsv`.

**Modified (possibly):** `dsignal/benchmark_split.py` — a `--by-class` flag on
`compare`, only if it does not already break results down per class.

**Not modified:** no model, feature, simulation or arena file. If this stage
edits `dsignal/model.py` or anything under `sim/`, something has gone wrong.

## Verification

- Gates before scoring: `python -m dsignal.verify` on the arena (expect
  4,844/4,844, already passing) and `python -m dsignal.verify_features`
  (6 checks).
- **Regression guard**: re-score `runs/geomwin_iv30_refit` on `benchmark/ --half
  dev` and confirm it still gives exactly **53/70** collapsed. This caught the
  Track D checkpoint-compatibility question in Stage 12 and is cheap.
- Row-count sanity: `runs/arena_v1_ensemble.tsv` and each v2 TSV must each hold
  2,422 rows, and their `(csv, region)` keys must match as sets — a mismatch
  means a shard was lost or a class CSV was skipped, and would silently bias the
  paired test. Assert this before running `compare`.
- Sanity on the v1 side: v1's `pred` is `"A"` for every A3/A5 event by
  construction, so its raw 5-class `correct` column will read 0 there. That is
  expected, documented in `dsignal/v1_reference.py`, and the reason only
  `correct_collapsed` is compared.
- Commit after each of Steps 0, 2, 3, 4 (build → verify → commit). Never push.
