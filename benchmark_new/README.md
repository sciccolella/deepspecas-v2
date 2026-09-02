# Curated benchmark, batch 2 (`benchmark_new/`)

A second manually-curated real-data benchmark, built and curated as
`plans/signal-native-v2-fresh-benchmark.md` (Stage 8) and
`plans/signal-native-v2-trash-signal.md` (Stage 9), independent of the
original `benchmark/` (which stays as-is and unspent for future use).

## Source data

`ENCSR376FGR` (GSE88229), SRSF3 shRNA knockdown, vs `ENCSR264TUE` (GSE88148)
non-target shRNA control, HepG2 cells. Two conditions, two replicates each.
Reads were aligned with STAR against the same reference used throughout this
branch, merged per condition (`c1_new.bam`/`c2_new.bam`, kept on `pasteur`
only -- 8.4GB/8.8GB, local disk unnecessary for the merge), and the events
below were extracted with rMATS + SUPPA2 and their consensus.

## How the truth set was built

1. **Consensus** (`benchmark_build/consensus.py`): rMATS and SUPPA2 calls at
   p<0.05 per class, matched by genomic-locus overlap (25bp tolerance) rather
   than exact-coordinate identity. This produced 573 candidate events across
   four SUPPA2 classes (`A3`, `A5`, `SE`->`CSE`, `RI`), padded per
   `benchmark_build/make_candidate_csvs.py`'s window convention (CSE
   +100bp/edge, RI/A3/A5 +0bp) and written to `benchmark_new_uncurated/`.
2. **Manual curation** (Stage 8 Track A4): every one of the 573 candidate
   windows was inspected in IGV and given a `Review` (curated truth class,
   possibly multi-label) and free-text `Notes`, recorded in
   `benchmark_new_uncurated/reviews/review_{A3,A5,ES,IR}.tsv`.
3. **Ingest** (`benchmark_build/ingest_curation.py`, Stage 9 Stage 0.1):
   normalizes `Review`/`Notes` into `curation.tsv` (573 rows, all retained).
   88 rows are ambiguous (multi-label `Review`, or blank) and are **not**
   scored anywhere -- they stay in `curation.tsv` only.
4. **Re-partition** (`benchmark_build/make_curated_csvs.py`, Stage 9 Stage
   0.2): the 485 unambiguous rows, grouped by curated truth, become this
   directory's CSVs. Regions are **not re-padded** -- each keeps the window
   geometry from its original candidate class; curation moves rows between
   truth labels, not window boundaries.

## Consensus -> curation counts

| truth | count | of which TRASH-noted |
|---|---:|---:|
| A3 | 33 | 1 |
| A5 | 23 | 0 |
| CSE (`ES.csv`) | 213 | 32 |
| RI (`IR.csv`) | 57 | 9 |
| NN | 159 | 84 |
| **total scored** | **485** | **126** |
| ambiguous (multi-label or blank, unscored) | 88 | 34 |
| **grand total** | **573** | **160** |

`trash` in `curation.tsv` is three-valued: `1`/`0`/`NA`. 14 of the 485 scored
rows carry `NA` (the unannotated prefix of `review_A5.tsv` -- see
`ingest_curation.py`'s docstring); those 14 are ordinary scored benchmark
events, just unlabelled along the TRASH axis. `NA` is its own stratum in
`benchmark_split.py`'s dev/test partition, not folded into `0`.

**This is the first curated v2 benchmark with real NN truth** (159 events --
windows where the consensus candidate did not hold up under review). The
original `benchmark/` has none; see its README for that caveat.

## Files

- `A3.csv`, `A5.csv`, `ES.csv`, `IR.csv`, `NN.csv` -- headerless
  `region,BAM_path_condition1,BAM_path_condition2`, paths relative to the repo
  root (`benchmark_new/c1.bam`, `benchmark_new/c2.bam`). CSV basename is the
  label, same convention as `benchmark/`: `ES`->`CSE`, `IR`->`RI`.
- `events.bed` -- the same 485 regions in BED form, interval-identical to the
  CSVs' region column (verified, not assumed -- see `dsignal.verify`).
- `curation.tsv` -- all 573 reviewed rows (the 485 scored ones plus the 88
  ambiguous/blank ones), with the full curation ledger: `review_raw`,
  `notes_raw`, `truth`, `ambiguous`, and the `trash`/`borderline`/
  `low_signal`/`complex`/`similar_signal` flags parsed from `Notes`.
- `split.tsv` -- committed dev/test split (Stage 9 Stage 0.3), stratified by
  `(truth, trash)` so the two halves carry comparable TRASH fractions per
  class, not just comparable class counts.
- `c1.bam`/`c2.bam` -- `samtools view -L events.bed` subsets of
  `c1_new.bam`/`c2_new.bam`, generated on `pasteur` and copied back (the full
  merges stay there). **Same caveat as `benchmark/`'s README:** any quantity
  measured on these BAMs that depends on off-window signal (background
  fraction, antisense fraction) is a lower bound on the real value.

## TRASH

The `TRASH` note (partial coverage -- see `ingest_curation.py`'s docstring for
exactly which rows are `NA` rather than a real negative) marks windows with
heavy spurious, usually intronic, alignment. It is not a reason to exclude
those events -- accuracy splits almost cleanly along it (v1 six-net ensemble
58.0% clean / 18.3% TRASH; v2 70.4% clean / 24.6% TRASH, measured on the 485
unambiguous rows before any dev/test split), and Stage 9 exists to measure
whether that axis is signal-recoverable and worth feeding into the model. See
`plans/signal-native-v2-trash-signal.md`.
