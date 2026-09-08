# Stage 16 — real data spanning multiple labels: pre-registration

Written and committed **before** any extraction/training runs, per
`plans/signal-native-v2-stage15-17-shortcut-attack-plan.md`. Nothing below
this line, up to the "## Results" heading, may be edited once building/
scoring starts.

## Motivation

Stage 15 confirmed (not merely hypothesized) that Stage 14's regression is
driven by real-vs-simulated realness, not splicing content: the model
learned "this is a real BAM record" → NN, because the 474 real rows added to
the pool were the *only* real data in it and were labelled NN uniformly.
This stage's attack: **add real data across all four non-NN classes**, so
realness is no longer perfectly confounded with the NN label. If the
shortcut is exactly what Stage 15 found, decorrelating realness from NN
should recover most or all of the lost accuracy while keeping some of the
NN-recall gain.

## Mixing design

- **Keep `ptbp1_arena` (pair 1) entirely eval-only**, unchanged from
  Stage 13/14 — no pair-1 row of any kind enters training. This preserves
  the confirmatory-arena structure (an arena never scored is a wasted
  investment; pair 1 is Stage 16/17's remaining confirmatory check).
- **Pool composition**: `npz_union2` (12,480 sim) + the existing 474 real NN
  rows from `aqr_arena` (unchanged from Stage 14, `real_nn_aqr`, `kind`
  column preserved) + new real positive rows from `aqr_arena`'s
  `consensus.tsv` (1,216 candidates: A3 128, A5 88, CSE 716, RI 284),
  gene-disjoint from `ptbp1_arena`/`benchmark`/`benchmark_new` by the same
  50kb-clustering rule Stage 14 used. No upsampling, no
  `--class-weight-boost`, no re-weighting of the real fraction — same "as
  is, unweighted" decision as Stage 14, so any effect is attributable to
  label diversity, not to a second confound (weighting).
- Real-data share of the pool rises from 474/12,954 = 3.7% to
  (474+1,216)/14,170 ≈ 11.9% (all-arithmetic estimate; the true number is
  fixed once gene-disjointness filtering is applied and reported in
  Results — matches the plan doc's ~13.6% estimate to within the rounding
  the plan used before the exact filtered count was known).
- **Gene-disjointness**: reuse Stage 14's exclusion machinery
  (`build_exclusion_set` in `sim/build_real_nn_extract_csv.py`), generalized
  to run over `aqr_arena/consensus.tsv`'s positive rows (same shape:
  `gene_id`, `chr`, `region_start`, `region_end`) rather than
  `nn_candidates.tsv`. Because positives and the NN half occupy different
  loci within `aqr_arena`, the two subsets can cluster differently against
  curated `benchmark`/`benchmark_new` regions — the exclusion set for
  positives is **rebuilt from scratch** against the positives' own regions
  (not reused verbatim from Stage 14's `exclude_genes_nn_attack.tsv`), then
  **unioned** with Stage 14's set before filtering either batch, so both
  real batches in the merged pool share one exclusion set and stay mutually
  consistent. Written to `sim/experiment/exclude_genes_stage16.tsv`.
  Asserted empty-overlap after filtering, same visible-assertion style as
  Stage 14.

## Gate (identical to Stage 14's, plus one comparison)

Same 3-condition pass/fail Stage 14 used, scored in the same fixed order:

1. **`benchmark_new` dev — primary gate.** `dsignal.score_benchmark --half
   dev --split benchmark_new/split.tsv --tta-condition-swap --target-prior
   nn-weight --nn-weight 0.1`, `dsignal.benchmark_split compare --by-class`
   against baseline `runs/bnew_dev_e4_union2.tsv` (142/242 = 58.7%
   collapsed, NN recall 11/79 = 13.9%).
2. **`benchmark/` dev — regression guard.** Same protocol against
   `runs/benchmark_events_stack_dev.tsv` (53/70 baseline).
3. **`ptbp1_arena` pair 1 — confirmatory only.** `--half both` against
   `runs/arena_v2_e4union2.tsv`, `--by-class`. Scored once per seed, never
   used to pick a seed or checkpoint.

**PASS** requires all three:
1. Median-over-3-seeds `benchmark_new` dev NN recall beats the 13.9%
   baseline.
2. ≥2/3 seeds individually beat the 13.9% baseline on NN recall.
3. No more-than-seed-noise regression on `benchmark_new` dev pooled
   collapsed accuracy (baseline 58.7%) or `benchmark/` dev (baseline
   53/70) — seed noise read against the branch's measured spread (5/70 at
   n=70, 10/242 at n=242).

Anything else is **FAIL**, including a pass on `ptbp1_arena` pair 1 alone
with a fail/null on `benchmark_new` dev — identical rule to Stage 14.

**Fourth comparison, informative not gating**: this mix's per-seed results
vs. Stage 14's own three seeds (`runs/nn_attack_realnn_seed{0,1,2}`), same
benchmarks, same McNemar convention — does multi-label real data regress
*less* than Stage 14's single-label mix, even if it doesn't pass condition 3
outright? This is the direct test of Stage 15's diagnosed mechanism and
belongs in the write-up regardless of the PASS/FAIL gate outcome.

## Seeds and hyperparameters

Identical to Stage 14 for comparability: 3 seeds (0, 1, 2),
`--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30`.

## Standing constraints (unchanged, restated)

- `benchmark_new`'s test half stays reserved, untouched.
- Never delete a measured number from this doc or any other; retractions go
  next to what they retract.
- Build → verify → commit each step; never `git push`; never create/switch
  branches unprompted.
- pasteur jobs: `limitram`, all four thread-pool env vars capped, `nice -n
  19`, `/proc/loadavg` before and after. Only `.npz`/labels/TSVs return from
  pasteur, never BAMs.

## Results

(filled in after scoring)
