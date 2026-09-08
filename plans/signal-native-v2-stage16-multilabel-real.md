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

Steps 2-3, 5 executed exactly as pre-registered (510/1,216 `aqr_arena`
positives kept after gene-disjointness, exclusion set 2,102 genes;
`dsignal.verify` 1,020/1,020 both checks; merged onto `npz_union2` +
Stage 14's 474 real NN rows -> `npz_union2_realmix`, 13,464 events, real
share 7.3%). 3 seeds trained (Step 4-6), in-sim macro-F1 0.906-0.911 (5-class
uncollapsed), pooled collapsed macro-F1 0.909, per-class F1 A 0.927 / CSE
0.886 / NN 0.882 / RI 0.937 -- close to Stage 14's own in-sim numbers
(0.905-0.911), so the mix isn't harder to fit in simulation, only different
zero-shot.

### benchmark_new dev (primary gate)

| seed | pooled collapsed | vs baseline (142/242=58.7%) | McNemar p | NN recall | vs baseline (11/79=13.9%) |
|---|---:|---:|---:|---:|---:|
| 0 | 144/242 = 59.5% | +2 events | 0.878 | 8/79 = 10.1% | −3 events |
| 1 | 141/242 = 58.3% | −1 event | 1.000 | 6/79 = 7.6% | −5 events |
| 2 | 145/242 = 59.9% | +3 events | 0.761 | 7/79 = 8.9% | −4 events |
| **median** | **59.5%** | **+2** | | **8.9%** | **−4** |

**Gate: FAIL on conditions 1 and 2.** Median NN recall (8.9%) does not beat
the 13.9% baseline -- it drops below it -- and 0/3 seeds individually beat
it. Condition 3 (no regression) trivially passes, since pooled collapsed
accuracy doesn't regress at all here (+2/−1/+3 events, all within noise) --
but the gate requires all three conditions, so this is an unambiguous
**FAIL**, on the opposite condition from Stage 14's.

Per-class detail (seed medians vs. baseline): CSE 83.2% -> ~94-97% (a real
gain, McNemar p=0.002-0.013 in each seed), A5 75.0% -> 50-58%, RI 67.9% ->
57-61%, A3 87.5% -> 75%. Macro-recall over the 5 classes drops 65.5% ->
57.5-59.6% -- CSE's gain is large enough to hold pooled accuracy flat while
NN and the other three classes all lose ground individually; no single
comparison but CSE's is McNemar-significant on its own.

### benchmark/ dev (regression guard)

| seed | pooled collapsed | vs baseline (53/70=75.7%) | McNemar p |
|---|---:|---:|---:|
| 0 | 57/70 = 81.4% | +4 events | 0.481 |
| 1 | 57/70 = 81.4% | +4 events | 0.481 |
| 2 | 59/70 = 84.3% | +6 events | 0.238 |

**No regression -- a consistent gain**, 3/3 seeds, none individually
significant but all in the same direction and one (+6) at the branch's own
measured noise ceiling. Same shape as `benchmark_new` dev: CSE gains hard
(25/33 -> 32-33/33, p=0.008-0.016 every seed), A5 drops (14/15 -> 10/15 in
2 of 3 seeds), RI ticks up.

### ptbp1_arena pair 1 (confirmatory only, not gating)

| seed | pooled collapsed | vs baseline (1175/2422=48.5%) | NN recall | vs baseline (135/800=16.9%) |
|---|---:|---:|---:|---:|
| 0 | 1668/2422 = 68.9% | +493 events | 296/800 = 37.0% | +161 events |
| 1 | 1685/2422 = 69.6% | +510 events | 313/800 = 39.1% | +178 events |
| 2 | 1672/2422 = 69.0% | +497 events | 313/800 = 39.1% | +178 events |

A large, consistent gain on both axes -- but per the pre-registration and
Trap 2 (B3 doc), **this is not sufficient on its own**: `ptbp1_arena`'s NN
half is built by the identical automated rMATS x SUPPA2 procedure as
`aqr_arena`'s (`benchmark_build/make_nn_candidates.py`, same thresholds),
so a large chunk of this gain is plausibly the model learning to recognize
*that construction's* NN signature specifically, generalizing between two
automated arenas built the same way, rather than "no splicing happening" in
general -- exactly the caveat that made this arena confirmatory-only rather
than gating in the first place. It does confirm the mixing changed the
model's real-data behavior substantially, consistent with the direct
Stage 14 comparison below.

### Direct comparison against Stage 14's own seeds (informative, not gating)

Same benchmark, same seed indices, `benchmark_new` dev:

| seed | Stage14 pooled collapsed | Stage16 pooled collapsed | McNemar p | Stage14 NN recall | Stage16 NN recall | McNemar p (NN pred) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 128/242 = 52.9% | 144/242 = 59.5% | 0.121 | 46/79 = 58.2% | 8/79 = 10.1% | b=38,c=0, p=4e-12 |
| 1 | 122/242 = 50.4% | 141/242 = 58.3% | 0.073 | 45/79 = 57.0% | 6/79 = 7.6% | b=39,c=0, p=2e-12 |
| 2 | 126/242 = 52.1% | 145/242 = 59.9% | 0.064 | 45/79 = 57.0% | 7/79 = 8.9% | b=38,c=0, p=4e-12 |

**This is the cleanest evidence in the whole Stage 15-17 line.** Adding
multi-label real data recovers essentially all of Stage 14's pooled-accuracy
regression (+16 to +19 events per seed vs. Stage 14, none individually
significant at n=242 but consistent in direction and magnitude across all
3 seeds) -- and it does so by **completely erasing** Stage 14's NN-recall
gain in the same motion (`b` = 38-39, `c` = 0 in every seed: every event
Stage 14 called correctly-NN and Stage 16 doesn't, none the other way).
That is exactly what Stage 15 predicts if the NN-recall gain **was never
disambiguating splicing from no-splicing** -- it was the model keying on
"real BAM data" and decorrelating that signal from the NN label removes
both the shortcut and the phantom recall gain in the same step, leaving
behind a small, non-significant net CSE-favoring shift (real positives are
majority CSE, 338/510) rather than a genuine NN classifier.

## Verdict: FAIL, but confirms Stage 15 rather than merely failing to fix it

The pre-registered gate fails cleanly: this mix does not move NN recall in
the intended direction, at all, in any seed. But unlike Stage 14, it also
does not cost anything -- `benchmark_new` dev and `benchmark/` dev pooled
collapsed accuracy both hold or improve. The mechanism is legible: NN
recall and the regression were the *same* phenomenon (the shortcut), and
removing the confound removes both together, landing very close to (in
pooled-accuracy terms, slightly above) the pre-Stage-14 baseline with a
CSE-favoring composition shift from the added real positives (already
majority-CSE by construction). Real NN data still does not teach genuine
"no splicing happened" recognition on curated data; it only ever taught
"this is a real BAM record from the one real cohort in the pool."

**For whoever picks this up next**: this closes the "just add more real
labels" branch of the real-data attack -- Stage 16's own data confirms
Stage 14's NN gain was entirely the shortcut, not partially. A future real-
NN attempt would need either (a) real NN data spanning genuinely multiple
*cohorts* (not just multiple labels within one cohort) so "real" itself
stops being a learnable feature, or (b) Stage 17's architectural route,
which attacks the shortcut directly rather than trying to dilute it with
more data. `runs/ensembles/e4_union2.json` stays untouched; this mix is not
folded into any ensemble.

