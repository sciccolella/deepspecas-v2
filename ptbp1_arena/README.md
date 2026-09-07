# PTBP1 arena — an automated, model-blind candidate pool (Stage 12 / Track B)

**This is not a curated benchmark.** No human inspected these events and no
model prediction entered any file here. It exists to compare two models
against each other, and for nothing else.

## What it is

1,622 candidate events from the rMATS × SUPPA2 consensus on a PTBP1 knockdown
vs its own matched non-target control, built by
`benchmark_build/run_arena.sh` (see `plans/signal-native-v2-real-supervision.md`,
Track B).

| condition | ENCODE | GEO | runs |
|---|---|---|---|
| PTBP1 shRNA KD, HepG2 | `ENCSR064DXG` | GSE80847 | SRR3469390, SRR3469391 |
| non-target shRNA control, HepG2 | `ENCSR603TCV` | GSE80908 | SRR3469516, SRR3469517 |

`ENCSR603TCV` is `ENCSR064DXG`'s own `possible_controls` entry, so the pairing
is ENCODE's. Same cell line, protocol and assay as `benchmark_new`'s SRSF3
pair; different RBP. Accessions were frozen in the plan *before* any scoring.

| class | candidates |
|---|---:|
| CSE (`ES.csv`) | 1,073 |
| RI (`IR.csv`) | 284 |
| A3 | 152 |
| A5 | 113 |
| **NN** (`NN.csv`) | **800** |
| **total** | **2,422** |

The positives are 2.8× `benchmark_new`'s 573 from the identical pipeline —
PTBP1 is a stronger splicing perturbation than SRSF3.

## The NN half

`benchmark_build/make_nn_candidates.py` builds it from the same two callers
with the criterion inverted: a locus **both callers see and both call clearly
unchanged**, in a demonstrably expressed gene. These are deliberately *hard*
negatives — a real catalogued event with real splicing at it, just no
differential usage, which is what `sim/select_events.py` calls `nn_hard` and
what most of `benchmark_new`'s curated NN turned out to be.

    SUPPA2:  dPSI measurable (not NaN), p > 0.20, |dPSI| < 0.05
    rMATS:   PValue > 0.20, |IncLevelDifference| < 0.05
    matched at one locus by overlap (TOLERANCE=25, as for positives)
    expressed: summed transcript TPM >= 1.0 in both conditions, every replicate
    and not overlapping any positive candidate

800 sampled (seed 0) from pools of 1,652 / 1,000 / 6,159 / 903 available loci,
allocated **proportionally to the positive class mix** so window widths follow
the positives' distribution by construction.

Two properties were checked rather than assumed:

* **No width leak.** Width alone separates NN from positives at AUC **0.493**
  (Mann-Whitney p=0.57, KS p=0.34) — i.e. not at all. Width has leaked class
  information on this branch before (`sim/width_leak_check.py`).
* **Extractor parity**: `dsignal.verify` matches 1,600/1,600 NN regions on both
  coverage and junctions.

* **A weak coverage skew, of the size curation also produces.** Mean window
  coverage alone separates NN from positives at AUC **0.417** (p=0.004) — NN
  slightly *lower*, median 15.1× vs 26.8×. For calibration, the same
  measurement on curated `benchmark_new` gives AUC **0.564** (p=0.050) — the
  same small magnitude in the **opposite** direction (NN *higher*, 41.9× vs
  25.2×). So this arena is no more coverage-separable than a curated set, but
  the sign differs: a model leaning on depth would be rewarded on one set and
  punished on the other. Neither is a shortcut worth much (|AUC−0.5| ≈ 0.06–0.08).

One thing to know if you rebuild the BAM subsets: they must be cut over
`candidates/all_events.sorted.bed` (positives **and** NN). The first subset
here was built from the positives-only BED, which left 11 of the first 12 NN
windows with **zero coverage** — every NN would then have been "correctly"
predicted by a model that had learned nothing but a coverage detector.

**The NN side's noise level is not measured.** The positives' 67.2% purity
comes from `benchmark_new`'s curation; no curated set of automatically-derived
NN exists to check this half against, so a locus both callers call unchanged
may still be one a curator would call a real event. Treat NN here exactly like
the positives: paired comparisons only.

## How to read numbers from it — the part that matters

**Roughly a third of these events are not real.** Measured on `benchmark_new`,
where the same consensus pipeline's candidates were curated: only **67.2%** of
candidates survive review as real events, and the class label is right for
**93.3%** of those that do. **No stringency filter improves this** — purity is
flat at 0.66–0.73 across every threshold on `rmats_pvalue`, `|suppa_dpsi|`,
`|rmats_incleveldiff|` and their conjunction (Stage 12, C2).

Consequences, both binding:

* **Absolute accuracy on this arena is meaningless** and must never be quoted
  as an accuracy. A model scoring 60% here is not "60% accurate".
* **Only the paired difference between two models is interpretable.** Symmetric
  label noise attenuates a difference without biasing its sign, so a paired
  McNemar between two predictors is valid; the magnitude is a lower bound on
  the true difference.

**NN is present but its noise is unquantified** — see "The NN half" below. The
consensus pipeline emits only positives; the NN candidates come from a separate
inverted criterion whose error rate has never been checked against curation,
unlike the positives' measured 67.2%.

## Single-use

Each frozen arena is **single-use per model generation**. Once a model has been
scored here, tuning against these events makes the next score meaningless — the
next generation gets fresh accessions, which is the entire point of scripting
the build. Pairs 2–3 are deliberately not frozen yet.

## Files

* `candidates/{A3,A5,ES,IR}.csv` — headerless `region,bam1,bam2`, the same
  convention as `benchmark/` and `benchmark_new/` (`ES`→CSE, `IR`→RI). Window
  padding follows `make_candidate_csvs.py`: CSE +100bp per edge, RI/A3/A5 +0.
* `candidates/events.bed` — the same 1,622 regions, BED (0-based half-open).
* `consensus.tsv` — the full consensus table with both callers' statistics.
* `c1.bam`/`c2.bam` — **not committed** (242MB/274MB, see `.gitignore`).
  `samtools view -L candidates/events.bed` subsets of the merged alignments,
  which stay on pasteur under `ptbp1_work/align/`. Rebuild or re-copy them
  before scoring. Same caveat as the other two benchmarks: any quantity that
  depends on off-window signal (background fraction, antisense fraction) is a
  lower bound on these subsets.

Verified with `python -m dsignal.verify` against the local BAM copies:
3,244/3,244 regions match on both coverage and junction parity.
