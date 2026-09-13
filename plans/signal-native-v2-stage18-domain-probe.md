# signal-native-v2 — Stage 18: domain probe (pre-registration + results)

Parent plan: `plans/signal-native-v2-stage18-22-accuracy-plan.md` (Stage 18,
with 20.1/20.2 started in parallel). This file is written **before any
probe number is produced**; the decision rules below are copied from the
parent plan verbatim and are not to be edited after scoring — results go in
the Results section at the bottom, retractions next to what they retract.

## What Stage 18 asks

Stages 14–17 showed the model can tell a real-BAM record from a simulated
one and that curated NN is a population neither sim nor arena NN covers.
Nobody has measured *which features* carry the real-vs-sim signal, whether
the window convention alone is a tell, or whether the current features can
express curated NN at all when given real labels. Stage 18 answers those
three questions on existing npz (no re-simulation, no main-model training)
so that Stages 19/20/22 spend pasteur and GPU time on the right arm.

## Pre-registered decision rules (fixed 2026-09-11, before scoring)

From the parent plan §18.4:

1. **18.1 real-vs-sim discriminator.** Pooled cluster-grouped OOF AUC
   ≥ 0.9 **and** the greedy removal curve falls below 0.7 within k ≤ 15
   removed features → those features are Stage 19's target list; both
   Stage 19 arms (sim-side coverage shape, feature-side replicate/read-QC)
   viable. AUC still > 0.85 past k = 30 → the gap is diffuse; the
   feature-side arm is dropped and Stage 22b (self-supervised pretraining on
   real windows) moves up behind Stage 20.
2. **Jittered-pad variant.** If the jittered-pad real pool's AUC is
   materially lower than the fixed-pad pool's (operationally: a drop
   ≥ 0.05 in pooled AUC under the same feature subset), the window
   convention is a cheap fix — the sim window sampler and/or a
   deployment-time jitter TTA go into Stage 19/21.
3. **18.2 real-only NN expressibility.** Binary NN-vs-positive AUC on the
   curated `bnew_dev` rows (trained on ptbp1+aqr minus any cluster shared
   with `bnew_dev`) < 0.70 with ~3.4k real labels → current features cannot
   express curated NN; Stage 19's new feature groups become the NN lever and
   Stage 20's NN expectation is "arena-NN and positives only". ≥ 0.80 → real
   supervision (Stage 20) is the NN lever. Between 0.70 and 0.80: report,
   no rule fires; Stage 20 proceeds with the Stage 14 gate unchanged.
4. **18.5 covshape/delcov.** Scored first with the 18.1 probe (are they
   domain tells: do they enter the top-15 of the removal curve?), then 3
   seeds vs base on both dev sets with McNemar. Adopted only if median
   collapsed accuracy is up on `benchmark_new` dev with no beyond-noise
   regression on `benchmark/` dev (noise ceilings 10/242, 5/70) and no
   per-class recall loss of the Stage 16 shape (CSE-only gain, A5/RI loss).

Nothing here touches `benchmark_new` test or `benchmark/` test.

## Stage 20.1 selection (pre-registered 2026-09-11, verified on the ENCODE portal before fetching)

Criteria from the parent plan: ENCODE shRNA KD with its own
`possible_controls`, paired-end ≥ 100 bp, one K562, RBPs distinct from
SRSF3/PTBP1/AQR.

| pair | condition | ENCODE | runs (all single-run replicates, PE100) |
|---|---|---|---|
| 3 | PRPF4 shRNA KD, HepG2 (the recorded pair-2 backup) | `ENCSR466NRW` | rep1 `SRR14845398`, rep2 `SRR14845397` (14.8M / 15.3M pairs, HiSeq 2500) |
| 3 | non-target control, HepG2 | `ENCSR093ERJ` (= `ENCSR466NRW`'s sole `possible_controls`) | rep1 `SRR14831566`, rep2 `SRR14831565` |
| 4 | U2AF2 shRNA KD, K562 | `ENCSR904CJQ` | rep1 `SRR3469570`, rep2 `SRR3469571` (25.0M / 23.8M pairs, HiSeq 2000) |
| 4 | non-target control, K562 | `ENCSR344XID` (= `ENCSR904CJQ`'s sole `possible_controls`) | rep1 `SRR3469464`, rep2 `SRR3469465` |

U2AF2 was chosen for pair 4 over HNRNPC/SF3B4/HNRNPK (all K562, all with
matched controls) because it is the core 3'-splice-site recognition factor
— a mechanism none of the existing pairs (SR protein, hnRNP, intron-binding
spliceosomal, tri-snRNP) touches, and one expected to produce A3/RI
phenotypes, which are v2's weakest positive classes. Launched on pasteur
2026-09-11 09:39 (`pairs34.sh` → `prpf4_work.log`, `u2af2_work.log`),
sequential, `nice -n 19`, `limitram 48G`, 12 threads, thread-pool vars
capped; load average 5.2 at launch. STAR now writes `@RG`/`RG:Z:<srr>`
(`sim/fetch_align_benchmark_new.sh`).

## Results

*(filled in as each step completes; nothing above this line changes)*

### 18.0 — real pool extracted (2026-09-11)

| cohort | kind | rows | source | gene id |
|---|---|---:|---|---|
| ptbp1 | `real_pos_ptbp1` / `real_nn_ptbp1` | 1,622 + 800 | `ptbp1_arena` (local BAMs), `--no-exclude` | ENSG (arena) |
| aqr | `real_pos_aqr` / `real_nn_aqr` | 510 + 474 | Stage 14/16 npz, unchanged | ENSG (arena) |
| bnew_dev | `real_bnew_dev` | 242 | `runs/real_dev_npz` (new build, dev only) | ENSG via `consensus.tsv` keyed through `curation.csv_source`, 242/242 |
| tra2 | `real_tra2` | 140 | `runs/real_tra2_npz` (both halves; test is spent) | SUPPA catalogue best-boundary match 125/140, rest the split's cluster |

3,788 rows; `dsignal.verify` 3244/3244 on the ptbp1 positives. A jittered-pad
twin of every cohort (`npz_jitter/`, `sim/jitter_extract_csv.py`, U[0,100]
per edge, seeded) was extracted for rule 2.

### 18.1 — real-vs-sim discriminator (2026-09-13)

Cluster-grouped (50 kb) 5-fold OOF LightGBM on the deployed
`no-geometry` row, 3,788 real vs 3,788 class-matched `npz_union2` rows
(`runs/domain_probe/rvs*.log`, `*/summary.json`, `*/features.tsv`,
`*/removal_curve.tsv`):

| variant | pooled AUC | aqr / bnew_dev / ptbp1 / tra2 | LOCO (held-out cohort) | AUC after removing 15 / 30 families |
|---|---:|---|---|---|
| deployed (`no-geometry`, union2) | **0.999** | 1.000 / 0.997 / 0.998 / 0.996 | 0.999 / 0.996 / 0.996 / 0.992 | 0.998 / **0.996** |
| `all` | 0.999 | 1.000 / 0.997 / 0.999 / 0.996 | — | 0.998 / — |
| `depth_norm=scale` rows | 0.998 | 0.999 / 0.994 / 0.998 / 0.995 | — | 0.994 / **0.987** |
| jittered-pad real pool | 0.998 | 1.000 / 0.996 / 0.998 / 0.994 | — | 0.998 / — |
| sim = `npz_beta` | 0.998 | 1.000 / 0.998 / 0.998 / 0.995 | — | — |
| sim = `npz_stage3` | 0.997 | 0.999 / 0.995 / 0.997 / 0.992 | — | — |
| sim = `npz_geomwin` | 0.998 | 1.000 / 0.997 / 0.998 / 0.996 | — | — |
| + `covshape` + `delcov` groups (18.5) | 1.000 | 1.000 / 0.999 / 1.000 / 0.999 | — | — |

Class-conditional AUC 0.998–1.000 for every class. Every past simulation
is equally far from real data by this measure (0.997–0.998) — none of the
realism knobs the branch has tried moved the axis this probe sees.

**What carries it** (gain rank, univariate Mann-Whitney AUC real-vs-sim,
real vs sim mean of the raw column):

| feature | gain | MW-AUC | real mean | sim mean | reading |
|---|---:|---:|---:|---:|---|
| `mean__log1p_count_2` | 68,843 | 0.026 | 1.54 | 4.24 | mean per-junction reads e^1.5 ≈ 3.7 vs ≈ 68 |
| `mean__log1p_count_1` | 9,133 | 0.031 | 1.67 | 4.23 | same |
| `mean__log1p_min_anchor` | 8,096 | 0.974 | 2.62 | 1.98 | shallow junctions never reach the 5 bp anchor floor |
| `log1p_n_junctions` | 4,033 | 0.853 | 2.86 | 2.03 | ≈ 16 junctions per real window vs ≈ 6.6 |
| `wmean__disjoint_from_top` | 2,777 | 0.695 | 0.50 | 0.34 | more of the read mass sits on junctions unrelated to the event |
| `wmean__log1p_interior_cov_2` | 1,608 | 0.259 | 0.99 | 2.03 | depth |
| `wmean__present_2` (scale variant, top) | 5,630 | 0.062 | 0.978 | 0.989 | real junctions are often present in one condition only |

The ~15× gap in the *mean* per-junction count is far larger than the
3.5–5.7× depth ratio `audit_noise` measured: real windows carry ~2.5× as
many distinct junctions, most of them at 1–3 reads, present in one
condition, disjoint from the event. Depth-rescaling does not remove it
(0.998; `present_*`, `n_junctions`, `interior_nonzero_frac` take over).
**The dominant tell is the sparse junction tail — noise/novel junctions at
realistic (very low) counts — not coverage shape**: `cov_autocorr` is
MW-AUC 0.56, i.e. real and simulated coverage are about equally smooth at
lag 50. `delcov` *is* a tell (MW-AUC 0.84): polyester writes no indels.

**Rules.** Rule 1: AUC > 0.85 past k = 30 in every variant → **the gap is
diffuse; the feature-side arm of Stage 19 is dropped as a *gap-closing*
device and Stage 22b moves up behind Stage 20.** (Stage 19's feature
groups — replicate concordance, read QC — remain justified as *curator
information*, not as realism.) The Stage 19 sim-side target list that this
probe does support is the junction tail (count distribution of
noise/novel junctions, one-condition-only junctions) ahead of positional/GC
coverage bias, which the coverage-autocorrelation column says is not where
sim and real differ. Rule 2: jittered pad 0.998 vs 0.999 — **does not
fire**; the window convention is not a measurable tell.

### 18.2 — real-only NN expressibility (2026-09-13)

`runs/domain_probe/realnn.log`, `realnn/summary.json`, `realnn/dev_pred.tsv`.
Real rows only, same features.

| setting | 5-class acc | collapsed | NN-vs-pos AUC | NN recall @5% / @10% FPR |
|---|---:|---:|---:|---|
| within ptbp1+aqr, 5-fold cluster-grouped (3,406) | 73.8% | 74.0% | **0.878** | 0.525 / 0.651 |
| aqr → ptbp1 (LOCO) | 67.9% | — | 0.850 | — |
| ptbp1 → aqr (LOCO) | 67.5% | — | 0.834 | — |
| ptbp1+aqr (minus 229 cluster-overlapping rows) → **bnew_dev** (242) | 54.1% | 54.1% | **0.581** | 0.101 / 0.152 (argmax 16/79) |
| same model → tra2 (140, no NN) | 70.7% | 72.1% | — | — |

Arena NN is expressible and transfers between arenas (0.83–0.85); curated
NN is not (0.58, barely above the 0.64 the zero-shot Transformer's `p_NN`
reaches). **Rule 3 fires (< 0.70): the current features cannot express
curated NN. Stage 19's new groups are the NN lever; Stage 20's NN
expectation is "arena-NN and positives only".**

Bound comparison vs the Transformer real-supervision ceiling
(`runs/real_supervision_ceiling/oof.tsv`, trained *within* bnew_dev): the
real-only GBM that never saw a bnew_dev row gets 131/242 5-class correct
vs the ceiling's 99 (McNemar 59 vs 27, p = 0.001; collapsed 58 vs 32,
p = 0.008). Different protocols (3.2k caller-labelled rows vs 5-fold on
242 curated rows), so not an A/B — but it says the ceiling was
data-limited, not that the Transformer's inductive bias helps on real
labels.

### 18.3 — free re-scores on `benchmark_new` dev (2026-09-13)

`runs/bnew_dev_{beta_allaug,beta_depthscale,stage3,ens_union2_twohead}.{log,tsv}`,
per-class in `runs/bnew_dev_stage18_3_compare.log`, vs `e4_union2` 142/242
(NN 11/79, CSE 89/107, RI 19/28):

| run | collapsed | p | NN | CSE | RI |
|---|---:|---:|---:|---:|---:|
| `beta_allaug` (dPSI realism) | 102 | <0.0001 | 7 | 52 | 19 |
| `beta_depthscale` | 83 | <0.0001 | 5 | 38 | 17 |
| `stage3` (effect-size realism) | 96 | <0.0001 | 10 | 43 | **25** (p = 0.11) |
| `ens_union2_twohead` alone | 136 | 0.21 | 14 | 86 | 16 |

None moves NN. The three older realism runs lose almost all of it on CSE.
`stage3`'s RI 25/28 is the one per-class signal worth remembering when
Stage 19 chooses its dPSI regime, at n = 28.

### 18.5 — covshape / delcov groups (feature code done 2026-09-13; 3-seed runs pending)

`FeatureConfig.event_groups` (opt-in; every existing checkpoint's 32-column
vector is unchanged, verified bit-identical), `verify_features` check 7
passes. Probe verdict above: `delcov` is a pure domain tell on the current
sim (no indels in polyester reads), `covshape` is not.
`runs/stage18_5_queue.sh` trains base seeds 1/2 on union2 n12480 (only
seed 0 existed), `covshape` s0–2 and `covshape+delcov` s0–2, scoring each on
both dev sets — results to be appended here under rule 4.
