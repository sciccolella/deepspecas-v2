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
| **total** | **1,622** |

2.8× `benchmark_new`'s 573 from the identical pipeline — PTBP1 is a stronger
splicing perturbation than SRSF3.

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

**There is no NN class.** The consensus pipeline emits only positive
candidates, so NN recall — the weakness every v1 and v2 checkpoint measured on
`benchmark_new` shares — cannot be measured here at all. Any comparison run on
this arena covers the classes both models are already good at.

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
