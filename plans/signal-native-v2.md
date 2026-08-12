# deepSpecas v2 — signal-native AS classification (paper follow-up)

## Context

`deepSpecas` today renders pileup data into one of six image representations and
classifies with six ImageNet-pretrained ResNet50s (25M params each), scoring 123/140 on
the manually-curated benchmark. Reading the extraction path surfaced that the image is not
just a lossy view of the signal — it discards the single most discriminative object in the
problem:

- **Junction identity is destroyed at extraction time.** `get_coords_from_bam`
  (`plot2s.py:991`) accumulates N-CIGAR ops into a per-base *depth-of-N-spans* array.
  Donor/acceptor coordinates, junction identity, and read-level junction pairing are gone
  before rendering. Two junctions sharing a donor — the definition of an A3 event — are
  indistinguishable from one long junction plus noise. The current nets solve A3/A5/CE/SE
  from **coverage shape alone**.
- **Absolute depth is destroyed.** `_getmat` (`plot2s.py:1217`) sizes the canvas as
  `h = max(c1.max(), c2.max(), s1.max(), s2.max())`, so bars are `depth/h`. A 10x and a
  1000x event render identically. Values are binary 0/255; no log scaling anywhere.
- **Resolution collapses on wide regions.** Every mode is 1000px wide (`RESIZE_SIZE`,
  `plot2s.py:26`), trained at `resize0.5` -> 500 columns. Benchmark region widths span
  936bp–181,765bp. A3 (widest tail among the small-event classes, max 41kb -> 83bp/column)
  is the worst class at 21/28; IR (max 3.1kb -> 6bp/column) is the best at 16/17.
- **4-channel modes are JPEG-degraded at training time only.** `ensure_JPGE_size` halves
  `colmplp2s`/`squish4d` to 500x100 and `comb4d` to 250x100 before writing; `predict`
  passes `ensure_JPGE_size=False`. Half the ensemble trained on blurrier input than it sees.
- **No augmentation.** The transform chain is `Resize -> ToTensor -> Normalize`, nothing else.
- **No read filtering.** No MAPQ/NH/anchor-length filter exists. A benchmark read with a
  285kb intron on a 10bp anchor at `NH:i:4` is counted 4x as junction evidence.

Intended outcome: a signal-native model operating on the junction set plus coverage, at
base resolution, replacing six 25M-param CNNs with one ~1-2M-param model; plus a rebuilt
simulation whose noise resembles real data. Framed as a **methods follow-up to the current
manuscript**.

### Known issue noted but deliberately not fixed

`plot2s.main` emits 8 images per genomic event (4 `CROSSFEEDS` x 2 `ZOOMS`,
`plot2s.py:1226` + `runDS.smk:19`); `runDS.smk:137` splits over **image rows**, so the same
event lands in train and val. Every historical validation number is inflated. The 140-event
benchmark is the only trustworthy metric. **v2 splits by gene.** We do not re-measure the
old models. (Related, also unfixed: `_crossfeed`'s `perc_1` branch sizes its subset with
`perc_2` (`plot2s.py:541`), so the `(0.2, 0)` variant injects nothing and duplicates
`(0,0)`; `mc_mplp.py` is dead code that would `NameError` on import, contra `CLAUDE.md:26`.)

## Decisions taken

| Question | Decision |
|---|---|
| Training data | Pretrain on simulation, fine-tune on rMATS+SUPPA consensus over held-out real data |
| Label set | 4-class (`A`, `CSE`, `RI`, `NN`) as the comparable headline; 6-class (`A3`,`A5`,`CE`,`SE`,`RI`,`NN`) as the extension |
| Confounders | All transcripts of the gene active at their own random PSI |
| Noise layers | Added incrementally, benchmark scored after each, producing a per-layer contribution table |
| Old val numbers | Noted, designed around, not re-measured |
| GTF | Used for **simulation only**. Never an input to classification. |

## Environment facts (verified from `benchmark/c1.bam`)

- STAR 2.7.10b, `Homo_sapiens.GRCh38.dna.primary_assembly` index, `--outSAMattributes All`
- 101bp paired-end, **unstranded** (no `XS:A` on any of 175,562 spliced reads)
- `jI:B:i` carries exact intron coordinates per read; `jM:B:c` carries intron motif
  (1=GT/AG, 2=CT/AC, 3=GC/AG, 4=CT/GC, 5=AT/AC, 6=GT/AT) — **strand without a GTF**
- Multi-mapping on spliced reads: NH=1 172,731 / NH>=2 2,821 (~1.6%)
- `~/SRR15133*.bam` (13KB, 164 reads) and `~/annotation.gtf` (3KB) are stubs, not usable

## Step 0 — Prerequisites (critical path, not code)

Re-acquire: GRCh38 primary assembly FASTA, matching Ensembl GTF, a STAR 2.7.10b index
built from them, and PRJNA255099 FASTQs. Everything else blocks on this.
*Tier: user / mechanical.*

## Phase 1 — Signal extractor

New module `dsignal/extract.py`. Reuse the existing `fetch`+CIGAR loop shape from
`get_coords_from_bam` (`plot2s.py:991`) — do not reuse the function itself, since its
output is the lossy form.

Per event, per condition, emit:

- **Junction dict** `(intron_start, intron_end) -> {count, motif, strand, min_anchor,
  nh_weighted_count}`. Fast path: parse `jI`/`jM` tags. Portable fallback: walk CIGAR N-ops
  and read the two boundary dinucleotides from the reference FASTA. **Mask `jM`'s +20
  annotated bit** (`jM mod 20`) — it encodes the STAR index's GTF and would leak annotation
  into the classifier.
- **Coverage arrays** `cov`, at base resolution, plus separately-tracked deletion coverage
  (the current code counts CIGAR D as coverage).
- **Read-level filters as configurable knobs**, defaulting on: min anchor length, max NH,
  min MAPQ. Absent from v1 entirely; expected to be a cheap accuracy lever.

Emit one portable record per event (npz or parquet): junction table, coverage, region,
strand vote, depth stats. This decouples extraction from modelling — every later phase
consumes these records.
*Tier: strong model (correctness-critical, subtle CIGAR/coordinate semantics; note
`get_reads_from_bam` already has a suspect `-1` offset at `plot2s.py:1067` and mishandles
`=`/`X` ops — do not replicate).*

## Phase 2 — Simulation

New `sim/` directory. Composition happens at the read level.

1. **Event catalogue**: `SUPPA generateEvents` on the GTF -> `.ioe` per type, giving event
   coordinates plus the inclusion/exclusion transcript sets. Labels are grounded in real
   annotation; no invented events. Maps directly onto the 6-class set (MX/AF/AL -> `NN` or
   excluded, decided at build time).
2. **Abundance matrix**: for the labelled event, split abundance between inclusion and
   exclusion transcript sets to hit target PSI per condition (ΔPSI is a knob). **All other
   transcripts of the gene get nonzero abundance at their own random PSI**, so every real AS
   event in that gene fires simultaneously and only one is labelled.
3. **Noise layers, added one at a time with a benchmark score after each** (this ordering
   *is* the per-layer contribution table):
   1. Real STAR alignment — simulate FASTQ, align with the same STAR 2.7.10b / same index /
      `--outSAMattributes All`, merge 3 replicates per condition into `c1.bam`/`c2.bam`
      exactly as the benchmark was built. **Never go simulator -> BAM directly**; this is
      what reproduces multi-mapping, short-anchor spurious junctions, soft clips and
      mismatch profiles.
   2. Pre-mRNA background — an unspliced transcript spanning the gene's full genomic extent
      at 1-10% of gene abundance. Expected to *lower* the RI score, which is the point.
   3. Neighbouring and antisense genes overlapping the window plus flank.
   4. Novel junctions — low-abundance transcripts with donor/acceptor shifted ±3-20bp or
      random exon joins, left **unlabelled**.
   5. PCR duplicates and adapter contamination (ASimulatoR provides both).
4. **Engine**: ASimulatoR (wraps polyester, adds PCR-dup/adapter noise) driven from the
   SUPPA catalogue rather than its own event generator; layers 2-4 enter as extra FASTA
   entries with explicit counts in polyester's per-transcript per-replicate count matrix.
   Negative-binomial overdispersion across replicates. 101bp PE. Fragment length fit from
   **unspliced pairs only** in the benchmark (raw mean TLEN 577 is inflated by
   junction-spanning pairs).
5. **Window randomization** — jitter region boundaries, vary flank 0.1x-10x the event span,
   never centre the labelled event reliably, cover the full 936bp-181kb width range.
   Without this the model learns window geometry instead of splicing.
6. **`NN` negatives** must carry the same noise stack: constitutive/single-isoform loci,
   plus **hard negatives** — real AS events with ΔPSI ~ 0, and novel-junction-only loci. If
   `NN` is "empty region," the model learns depth.
7. **Split by gene.** Not by event, not by row.

*Tier: strong model for the abundance/PSI design, window randomization and negative design
(these determine whether the dataset is learnable at all); cheaper model for the FASTA
assembly, count-matrix plumbing and STAR/samtools driver scripts.*

## Phase 3 — Models

1. **LightGBM / rules baseline** on hand-built junction-topology features (shares-donor,
   shares-acceptor, log intron length, per-condition counts and log-ratio, junction count
   over local exonic coverage, intronic-over-flanking-exonic coverage for RI). **Run this
   first.** If it approaches 123/140, that reframes the entire contribution and must be
   reported honestly. *Tier: cheap model.*
2. **Junction-set Transformer** (primary). Tokens = distinct junctions across both
   conditions; CLS token carries region length, per-condition depth, strand vote. 4 layers,
   d~192, 4 heads, ~1-2M params. **Relative-position bias from log10 genomic distance** —
   distances span 4 orders of magnitude, raw bp will not work. Attention over junctions is
   the explainability story (`--explain` was never implemented; `deepSpecas.py` still
   raises `NotImplementedError`). *Tier: strong model.*
3. **Augmentations**, none of which the image pipeline had: read subsampling for depth
   robustness, condition swap (label-invariant), coordinate jitter, junction dropout, and
   **left-right flip, which maps A3<->A5 exactly**. *Tier: cheap model.*
4. **Dense 1D dilated CNN** on `[cov1, cov2, skip1, skip2]` resampled to ~4096 bins plus a
   channel marking exact junction positions — only if the transformer misses coverage-shape
   events with no junction evidence. Complement, not replacement. *Tier: strong model,
   conditional.*

## Phase 4 — Real-data fine-tune

rMATS + SUPPA consensus over public RNA-seq **other than PRJNA255099**, held out by
chromosome and dataset — the same procedure `benchmark/README.md` used to build the truth
set. Fine-tune the sim-pretrained model. Anticipate the "isn't this distilling rMATS?"
review: the manual curation showed rMATS/SUPPA make debatable calls, so the claim is
beating them on the curated 140, plus annotation-free millisecond inference.
*Tier: strong model for the label-QC and split design; cheap model for the pipeline driver.*

## Phase 5 — Paper artifacts

- **Simulation realism figure**: sim vs. real benchmark BAMs on junctions per region, reads
  per junction, novel-junction fraction, intronic:exonic coverage ratio, NH distribution,
  spliced-read anchor-length distribution, 3' coverage bias.
- **Headline ablation**: train on old-style sim vs. new-style sim, both evaluated on the
  untouched 140-event benchmark — a direct measurement that noise realism transfers.
- **Per-layer contribution table** from Phase 2.3.
- **6-class extension**: strand from `jM mod 20` / reference dinucleotides, reporting the
  A3-vs-A5 split the image representation structurally cannot produce.
- **Cost comparison**: one ~1-2M-param model at ms/event vs. six 25M-param nets at ~17s/event
  of rendering (`report.md`).

*Tier: cheap model for figures, strong model for the ablation design and writing.*

## Verification

- **Extractor**: on benchmark regions, junction counts derived from `jI` tags must match
  those derived from the CIGAR-walk fallback exactly; strand from `jM mod 20` must match
  strand from reference dinucleotides; total coverage must match `get_coords_from_bam`'s
  `covs` where the semantics coincide (accounting for its unconditional `covs[0] = 0` and
  its counting of CIGAR D as coverage).
- **Leakage audit**: confirm no feature derives from the GTF — grep the feature builder for
  annotation inputs, and confirm the `jM >= 20` bit is masked. Additionally verify a model
  trained with window geometry shuffled scores no better than chance on geometry alone.
- **Simulation**: the realism statistics above must overlay the real BAMs before the sim is
  used for any headline number.
- **Scoring**: reuse `tools/score_benchmark.py` (per-CSV/aggregate accuracy + confusion
  matrix) so v2 numbers are directly comparable to the 123/140 baseline; add per-event
  region width to the output so accuracy-vs-width can be plotted for both v1 and v2.
- **End to end**: a v2 CLI accepting the same `region,bam1,bam2` CSV as `dsEnsemble.py`,
  run over all four `benchmark/{A3,A5,ES,IR}.csv` files, scored against the truth set.
- **Split integrity**: assert no gene appears in more than one of train/val/test, at every
  stage — the specific failure v1 had.
