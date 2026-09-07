# AQR arena — pair 2 (Stage 14, the NN attack)

**This is not a curated benchmark**, and unlike `ptbp1_arena` (pair 1) it is
**not itself scored as an arena** — no model is ever run against it. Its only
role in Stage 14 (`plans/signal-native-v2-stage14-nn-attack-plan.md`,
`plans/signal-native-v2-arena-nn-attack.md`) is to supply pair 2's **NN half**
as real, extracted hard negatives added to the sim training pool. No human
inspected these events and no model prediction entered any file here.

## What it is

2,016 candidate events (1,216 positives + 800 NN) from the rMATS × SUPPA2
consensus on an AQR knockdown vs its own matched non-target control, built
with the exact pipeline pair 1 used (`benchmark_build/run_arena.sh` +
`benchmark_build/make_nn_candidates.py`), same thresholds, same class-mix
proportional NN allocation, seed 0.

| condition | ENCODE | GEO | runs |
|---|---|---|---|
| AQR shRNA KD, HepG2 | `ENCSR431SXF` | GSE177214 | rep1 SRR14843100, rep2 SRR14843099 (one run each) |
| non-target shRNA control, HepG2 | `ENCSR225PRV` | GSE176746 | rep1 SRR14840064 (one run); rep2 SRR14840062 + SRR14840063 (two runs, concatenated before alignment) |

`ENCSR225PRV` is `ENCSR431SXF`'s sole `possible_controls` entry (verified
against the ENCODE portal + NCBI SRA — see
`plans/signal-native-v2-arena-nn-attack.md`'s Step 1 section, which also
records a correction to the plan draft's originally claimed rep1 run
structure). Same cell line, protocol and assay as `ptbp1_arena`'s PTBP1 pair
and `benchmark_new`'s SRSF3 pair; different RBP — AQR is a core
intron-binding spliceosomal factor, chosen for maximum mechanistic distance
from PTBP1 (hnRNP) and SRSF3 (SR protein).

| class | candidates |
|---|---:|
| CSE (`ES.csv`) | 716 |
| RI (`IR.csv`) | 284 |
| A3 | 128 |
| A5 | 88 |
| **NN** (`NN.csv`) | **800** |
| **total** | **2,016** |

Positives are fewer than `ptbp1_arena`'s 1,622 (AQR is a smaller
perturbation than PTBP1 by this consensus-candidate measure), but the NN half
is drawn to the same fixed count (800) regardless, per `make_nn_candidates.py`'s
`--n-total` default.

## The NN half

Same criterion as `ptbp1_arena`'s (already amended before any scoring — see
that README and `benchmark_build/make_nn_candidates.py`'s module docstring):

    SUPPA2:  dPSI measurable (not NaN), p > 0.20, |dPSI| < 0.05
    rMATS:   PValue > 0.20, |IncLevelDifference| < 0.05
    matched at one locus by overlap (TOLERANCE=25, as for positives)
    expressed: summed transcript TPM >= 1.0 in both conditions, every replicate
    and not overlapping any positive candidate

800 sampled (seed 0) from pools of 1,507 / 968 / 6,437 / 785 available loci
(A3/A5/CSE/RI respectively, after dropping positives), allocated
proportionally to the positive class mix: CSE 471, RI 187, A3 84, A5 58.

**No width leak, checked rather than assumed.** Width alone separates NN
from positives (1,216 pooled across A3/A5/CSE/RI) at **AUC 0.483**
(Mann-Whitney p=0.197, KS p=0.075, positives median 2,604 bp vs NN median
3,020 bp) — not separable, the same conclusion `ptbp1_arena`'s 0.493 reached.
Width has leaked class information on this branch before
(`sim/width_leak_check.py`), so this is checked per arena, not assumed to
transfer from pair 1.

**Extractor parity**: `dsignal.verify` matches 4,032/4,032 (2,016 events ×
2 checks: coverage and junctions) on both coverage and junctions — a clean
pass, same as `ptbp1_arena`'s 4,844/4,844 (2,422 × 2).

**The NN side's noise level is not measured**, exactly as for `ptbp1_arena`'s
NN half and for the same reason: no curated set of automatically-derived NN
exists to check it against. A locus both callers call unchanged may still be
one a curator would call a real event. Stage 14's pre-registration
(`plans/signal-native-v2-arena-nn-attack.md`) treats this as an open,
unquantified caveat on the real-hard-negative injection, exactly as B3's
pre-registration treated it for `ptbp1_arena`.

## Why this arena is not itself scored

Unlike `ptbp1_arena`, which was built specifically to compare v1 and v2
head-to-head (Stage 13/B3), this arena's positives are **not used** by
Stage 14 at all — only its NN half, extracted as real training examples (no
class-balance upsampling, unweighted addition to the existing sim pool, per
`plans/signal-native-v2-arena-nn-attack.md`'s fixed pool-composition
decision). No local BAM subset was built (unlike `ptbp1_arena/c1.bam` /
`c2.bam`, gitignored ~500MB copies used for repeated local scoring); instead
`dsignal.extract` reads directly from the full merged
`aqr_work/align/{c1,c2}_aqr.bam` on pasteur, since the arena is used exactly
once, for extraction, not for repeated scoring. `aqr_work/` (79G-scale FASTQ
+ BAMs) stays on pasteur only, per CLAUDE.md's "only `.npz` records, labels
and TSVs come back" rule; `consensus.tsv` and `candidates/*.csv` — the files
that fully define the arena, including their (pasteur-local) BAM paths for
provenance — are committed here, matching `ptbp1_arena`'s convention.

## Gene-disjointness

Genes contributing to this arena's NN half must not overlap
`ptbp1_arena`'s genes (held out as Stage 14's confirmatory eval),
`benchmark_new`'s or `benchmark`'s genes (both live curated eval sets) — see
`plans/signal-native-v2-arena-nn-attack.md`'s gene-disjointness rule and
`sim/experiment/exclude_genes_nn_attack.tsv` (built in Step 3, applied before
any row from `candidates/nn_candidates.tsv` enters the training pool).
