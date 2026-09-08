# Stage 15 — verifying the real-vs-simulated shortcut hypothesis

Diagnostic only, no training. Written per
`plans/signal-native-v2-stage15-17-shortcut-attack-plan.md`.

## Pre-registered metric and criterion (fixed before scoring)

For each of the 3 Stage-14 seed models (`runs/nn_attack_realnn_seed{0,1,2}`)
and the `e4_union2` baseline, compute the **NN-prediction rate** over
`ptbp1_arena/candidates/{A3,A5,ES,IR}.csv` — 1,622 real, non-NN-truth events.
`ptbp1_arena` is held out from Stage 14 training (only `aqr_arena` supplied
training rows), and every one of these events is a real BAM record, not
simulated.

Paired McNemar (exact binomial on the discordant pairs, same events, same
`(csv, region)` key) on "predicted NN vs not" between each seed model and the
baseline.

**Hypothesis is supported if** all 3 seeds show a significant, consistent
increase in NN-prediction rate over baseline on this set, consistent in
direction and magnitude with the recall/regression pattern already measured
(NN recall +34-35pp on `benchmark_new` dev, pooled collapsed accuracy down
14-20 events / 242, zero compensating wins on A3/A5/CSE/RI).

## Data source — no new scoring needed

Stage 14's own confirmatory step (gate condition 3, "ptbp1_arena pair 1")
already scored all 3 seeds and the baseline over the *full* arena (2,422
events, all five `candidates/*.csv` including `NN.csv`) with `--half both`:
`runs/arena_v2_e4union2.tsv` (baseline) and
`runs/nn_attack_arena_seed{0,1,2}.tsv`. These already contain per-event
`truth`/`pred` for the 1,622 non-NN-truth rows this diagnostic needs — same
scoring protocol (`--tta-condition-swap --target-prior nn-weight --nn-weight
0.1`) the pre-registration specifies. Filtering existing artifacts rather
than rescoring: n=1,622, keyed on `(csv, region)`, checked equal across all
four files.

## Results

**NN-prediction rate, pooled over all 1,622 non-NN-truth events:**

| model | NN-prediction rate | discordant vs baseline (b / c) | McNemar p |
|---|---:|---:|---:|
| baseline (`e4_union2`) | 38/1622 = 2.3% | — | — |
| seed 0 | 779/1622 = 48.0% | 0 / 741 | 1.7e-223 |
| seed 1 | 865/1622 = 53.3% | 0 / 827 | 2.2e-249 |
| seed 2 | 816/1622 = 50.3% | 0 / 778 | 1.3e-234 |

`b` = baseline predicted NN, seed didn't; `c` = baseline didn't, seed did.
`b = 0` in all three seeds — every single discordant event moves *toward*
NN, none moves away. This is about as clean a signal as a diagnostic on real
data produces.

**Per-class breakdown** (NN-prediction rate, baseline vs. each seed):

| truth | n | baseline | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|---:|
| A3 | 152 | 3.9% | 57.9% | 60.5% | 58.6% |
| A5 | 113 | 2.7% | 52.2% | 55.8% | 55.8% |
| CSE | 1073 | 2.3% | 44.8% | 49.1% | 45.5% |
| RI | 284 | 1.4% | 53.2% | 64.4% | 62.0% |

Uniform across all four classes — not concentrated in one label the way a
genuine splicing-signal confusion would be. Effect size is largest on RI,
smallest on CSE, but every class moves 40-60pp.

**Collapsed accuracy on this same non-NN subset** (context, not part of the
pre-registered criterion): baseline 64.1%, seeds 35.6% / 31.6% / 32.9% — a
~30-point drop, consistent with (larger than) the `benchmark_new`/`benchmark`
dev regressions already measured in Stage 14, and consistent with the model
gaining almost nothing back in correct-non-NN calls to offset the NN
overcalls.

## Verdict: **hypothesis confirmed**

All three pre-registered conditions hold, by a wide margin:

1. All 3 seeds show a significant increase (p < 1e-220, not merely
   "significant" — essentially deterministic at this n).
2. Consistent in direction: 3/3 seeds, every class, `b=0` in all three (no
   event moves away from NN).
3. Consistent in magnitude with the regression: the 20-25x jump in
   NN-prediction rate on *known-real, known-non-NN* data lines up with
   Stage 14's finding that NN recall gains concentrated on real eval sets
   (`benchmark_new` +43pp, `ptbp1_arena` confirmatory +64-68pp) while in-sim
   macro-F1 didn't move. This diagnostic isolates that mechanism directly:
   the model is not confusing splicing types, it is calling "real BAM
   record" → NN, almost independent of what the event actually is.

The 474 real NN rows being the *only* real BAM data in a 12,954-event pool,
labelled NN uniformly, is confirmed as the mechanism, not merely a plausible
story. Optional Step 3 (replicate on `aqr_arena` positives, which requires a
pasteur extraction) was not run — the effect here is already unambiguous
enough (p < 1e-220, `b=0` in every seed) that a second cohort would not
change the verdict; skipped to move to Stage 16, which attacks the
now-confirmed mechanism directly.

## Resume-doc update

`plans/signal-native-v2-resume-9.md`'s open caveat #5 ("Stage 14's
real-vs-simulated-shortcut mechanism is a hypothesis, not a verified
finding") is **resolved: confirmed**, not merely "consistent with." See
resume-9's update alongside this note.
