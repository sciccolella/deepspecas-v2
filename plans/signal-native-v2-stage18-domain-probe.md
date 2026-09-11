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
