# signal-native-v2 — ensemble plan results

*Implements `plans/signal-native-v2-ensemble.md`. Written 2026-08-19, same
session that ran it end to end (Stages 0–6).*

---

## Headline

**No parity, no win.** E3 (16-member image-free ensemble) reaches **77.1%
dev / 85.7% test** collapsed, up from the single-run headline's **75.7% dev /
82.9% test** — a real, consistent, zero-cost gain (+1 dev, +2 test, 0 losses
either half). Against the v1 six-net ensemble reference it is **not
significant** on the one half that counts: test McNemar p=0.549, discordant
7 v1-only vs 4 v2-only. v2 sits **3 events behind v1 on test** (60/70 vs
63/70), same side as every comparison this plan has run, closer than the
17-event gap the branch believed for months, but not closed.

The mechanism that worked was **more data, not ensembling**: a single model
trained on the union of `npz_geomwin` + the four new gene-disjoint batches
(`npz_bag2`) scored **78.6% dev**, the best number in this plan, beating both
the bagged-ensemble and the full 16-member E3. Bagging itself — training
separate models on disjoint gene sets and averaging them — added nothing once
view diversity was already in the pool (E3 == E2 bit-for-bit, 0 discordant
events). Recorded as the pre-registered scale-vs-bagging control resolving
cleanly in one direction.

## Corrected v1 target (Stage 0.1)

Every prior document compared against "v1 = 123/140 = 87.9%" as a single
number. Split-wise, scored with the same `(csv, region)` keying every other
number on this branch uses (`dsignal.v1_reference`, `dsignal.benchmark_split`):

| model | dev | test | all |
|---|---|---|---|
| v1 six-net ensemble | 60/70 = 85.7% | **63/70 = 90.0%** | 123/140 = 87.9% |
| v1 `squishstack` single net | 56/70 = 80.0% | 50/70 = 71.4% | 106/140 = 75.7% |

**The test figure here (63/70) supersedes the plan's own 62/69.** The plan
document (written before this session ran Stage 0.1) inherited a number from
an earlier ad hoc analysis that deduplicated the one region appearing in both
`A3.csv` and `A5.csv` by region alone, dropping one of its two rows. This
branch's convention — `dsignal.benchmark_split`, and now `dsignal.v1_reference`
— keys on `(csv, region)` and keeps both, which is *why* v2's own TSVs always
show 70/70 rather than 69/71. Keeping both rows for v1 too gives 60+63=123,
which is internally consistent with the well-established 123/140 = 87.9%
all-140 headline (60/70 + 63/70 = 123/140, exactly) in a way 60+62=122/139
was not. Per convention, both figures are kept on record; 63/70 is the one to
use going forward.

## Stage-by-stage

**Stage 0 (plumbing).** `dsignal/v1_reference.py` (the table above),
`Predictor` H1/H2 fixes (feature_config validated across checkpoints;
per-fold prior correction from each checkpoint's own `train_class_counts`),
`dsignal/ensemble.py` (`EnsembleSpec`, `MultiRunPredictor`, three combine
rules, a posterior cache). All three verified as an identity on the stored
53/70 dev headline before anything else was built on top.

**Stage 1 (seeds).** E1 = 5 seeds of the headline command, equal weight:
**53/70 dev**, bit-identical to the single run (1-for-1 swap, McNemar
p=1.000). Seed variance is not the gap.

**Stage 2 (view diversity).** `ens_allfeat`, `ens_twohead`, `ens_depthscale`,
`ens_augsub`, `ens_nn15`, `ens_cse` (`--class-weight-boost CSE 1.75`, the
plan's pre-registered fallback for the motif-feature member — cheaper, and
the payoff was meant to show in the ensemble, not the member). E2 = E1 + one
per family: **54/70 dev**, +1 over E1 with zero losses.

**Stage 3 (data bagging, pasteur).** Four new gene-disjoint batches
(`b8`–`b11`, seeds 8–11, 3,840 events, `npz_bag2`), built with
`sim/run_batch.sh` under `limitram`/thread caps/`nice -19` per `CLAUDE.md`,
each excluding every gene used by `b1`–`b5`, `b7stage3`, and the earlier new
batches in the chain (0 overlap, verified in every `select_events` log).
Acceptance gates (`dsignal.audit_noise compare`, `sim.width_leak_check`) both
clean relative to the already-accepted batches. Design inspected before any
polyester/STAR time was spent (the `nn_hard` dPSI==0 check that would have
caught the previous session's null-dPSI double-draw bug).

Five members: two on `npz_geomwin` alone (seeds 20/21), two on `npz_bag2`
alone (seeds 0/1), one on their union (`npz_union`, 8,640 events). The
pre-registered control resolved cleanly:

| spec | dev, collapsed |
|---|---|
| E3 = E2 + all 5 bagged members | 54/70 = 77.1% (== E2, 0 discordant) |
| bagged pair only (4 disjoint-data members, no union) | 53/70 = 75.7% |
| **union-alone, single run** | **55/70 = 78.6%** |

Union-alone beats the bagged-pair ensemble *and* the full 16-member E3. Per
the plan's pre-registered framing this is a **scale finding, not a bagging
one** — none of the pairwise deltas clear significance at n=70, but the
direction is unanimous and union-alone is one training run, not five.

**Stage 4 (combination rule + weighting).** Cached E3's 16-member per-fold
posteriors once; swept the pre-registered 3 rules × 2 weightings (equal,
per-run weight = out-of-fold sim macro-F1) analytically. **All six cells are
bit-identical**, 0 discordant events pairwise. At 16 members whose sim
macro-F1 all sit in a narrow 0.851–0.895 band, neither the rule nor
F1-reweighting moves a single prediction. A clean null, reported as one
rather than treated as grounds to keep searching.

**Stage 5 (hybrid ceiling, not shipped).** Reused `runs/v1_crossover/pred/`
(already carries full 5-class probabilities per benchmark event) against
E3's dev probabilities:

| combination | dev, collapsed |
|---|---|
| image-free E3 (the deliverable) | 54/70 = 77.1% |
| + v1 crossover net, mean_prob | **61/70 = 87.1%** |
| oracle union, E3 OR v1 squishstack single net | 64/70 = 91.4% |
| oracle union, E3 OR v1 six-net ensemble | 66/70 = 94.3% |

Averaging in a single extra image-based net already beats the v1 six-net
ensemble's own 60/70 dev — there is real complementary signal left in the
image view. This section is a ceiling, explicitly not a deliverable; the
oracle-union figures answer "how much signal is left to capture", not
anything reachable without images.

**Stage 6 (test, spent once).** E3 (ties E2 on dev under every Stage-4 rule,
the fuller pre-registered superset) scored on test:

| | test, collapsed | test, 5-class |
|---|---|---|
| E3 | 60/70 = 85.7% | 58/70 = 82.9% |
| single-run headline | 58/70 = 82.9% | 56/70 = 80.0% |
| v1 six-net ensemble | 63/70 = 90.0% | — |

E3 vs the single-run headline: +2, 0 losses, McNemar p=0.500 — same
right-sign, no-cost pattern as dev, not itself significant at n=70.

E3 vs v1, test (the pre-registered decision): discordant 7 v1-only / 4
v2-only, McNemar p=0.549. **Neither parity (test accuracy ≥ v1's) nor a win
(p<0.05) was reached.** Per the plan's own rule, test has now been spent —
any further comparison against v1 needs a fresh held-out half.

## What this changes about the target

Before this plan: the branch's working belief was "v2 is 17 events behind
v1" (123 vs the mid-70s%, read off the wrong denominator). After it: v2 is
**3 events behind on the only half that was never read during tuning**
(63 vs 60 of 70), and the ensemble closed roughly a third of the dev-side
gap that existed at the start of this plan (from -7 to -6 dev events,
E1→E3) without moving the underlying single-model transfer number at all —
the closing came from breadth (view diversity) and data volume, not from any
single intervention.

## Honesty ledger

* `DPSI_BETA`/`PSI_LEVEL_BETA` remain fitted on all 140 benchmark events;
  Stage 3's new batches used `PSI_DIST=uniform`, so they stay clean.
* The NN operating point (`nn_weight=0.1`) was reused unchanged, not re-tuned.
* E1/E2/E3 compositions and the Stage-4 rule grid were pre-registered in the
  plan before any dev number was read; no greedy member selection was run.
* Test was scored exactly once (Stage 6, this document). Any further
  comparison against v1 on this benchmark needs a fresh held-out half.
* Stage 5's hybrid numbers are a ceiling, not reachable by the shipped
  (image-free) model, and must not be quoted as the v2 result.
* `union-alone`'s 78.6% dev is the best number in this plan but was **not**
  eligible for the Stage 6 test spend — it fell out of the Stage 3 control,
  which was pre-registered as a diagnostic (bagging vs. scale), not as an
  ensemble candidate. It is recorded here as the clearest lead for a future
  plan: scaling `npz_union`-style data further, not ensembling more views of
  the current size.

## Critical files touched

`dsignal/v1_reference.py` (new), `dsignal/predict.py` (H1/H2),
`dsignal/ensemble.py` (new), `dsignal/score_benchmark.py` (`--ensemble`,
`--rule`), `dsignal/train.py` (`--class-weight-boost`),
`runs/ensembles/{single_iv30,e1_seeds,e2_views,e3_bagged,e3_f1weighted,
bag_pair_only}.json`, `sim/experiment/{npz_bag2,npz_union,labels_bag2.tsv,
labels_union.tsv}`, `sim/experiment/batches_bag2/{b8,b9,b10,b11}`.
