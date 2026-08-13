# signal-native-v2 — checkpoint (2026-08-13)

Status snapshot for resuming work on `plans/signal-native-v2.md` in a new
session. Branch: `signal-native-v2` (created from `master`, not `ensemble-perf`).

## Done and committed (local repo, branch `signal-native-v2`)

**Phase 1 — signal extractor** (prior session)
- `dsignal/extract.py`, `dsignal/verify.py`. Verified 280/280 (coverage
  parity, junction parity, jM-vs-reference-dinucleotide strand) against
  `benchmark/{A3,A5,ES,IR}.csv`.

**Phase 2 steps 1-4 — simulation, up through real STAR realignment** (this
session)
- `sim/gtf_transcript_lengths.py`, `sim/gtf_gene_spans.py` — GTF-derived
  tables, simulation-only inputs. Row counts cross-checked exactly against
  STAR's own `transcriptInfo.tab` (227,818) and `geneInfo.tab` (60,676).
- `sim/catalogue.py` — parses `suppa.py generateEvents` `.ioe` files into a
  labelled `Event` catalogue. SUPPA type -> label: `SE`->`CSE`, `A3`->`A3`,
  `A5`->`A5`, `RI`->`RI`; `MX`/`AF`/`AL` excluded from labels. **Bug fixed**:
  coordinate regex `-?\d+` misread the SUPPA coordinate-pair hyphen as a
  sign, producing negative `region_start` — fixed to `\d+`.
- `sim/abundance.py` — `build_count_matrix` (forced event, target PSI per
  condition, Dirichlet baseline for every other transcript in the gene,
  independent NB draws per transcript per replicate) and
  `build_count_matrix_baseline` (plain NN negative, no forced event).
  Verified: mean abs PSI recovery error ~0.03 (count-matrix level, before
  any sequencing/alignment noise), p90 ~0.08, across 200-300 real events.
- `sim/select_events.py` — pilot design generator, one gene per event
  (avoids reconciling two forced events in one gene). Current pilot design:
  60 events/label (A3/A5/CSE/RI) + 60 NN-hard (real event, psi1==psi2) + 60
  NN-plain (catalogue-free gene, >=2 transcripts) = 360 rows, 360 distinct
  genes.
- `sim/build_experiment.py` — combines all design rows' per-gene count
  matrices into one transcript x sample matrix + transcript-id list for
  FASTA extraction + a PSI-recovery check file.
- `sim/run_polyester.R` — drives `polyester::simulate_experiment_countmat`
  directly from our count matrix (bypasses ASimulatoR's own event
  generator, per plan). Layer-1 only: no PCR-dup/adapter noise yet (that's
  noise layer 5, added later).
- `sim/align_reads.sh` — aligns each replicate with STAR 2.7.10b / same
  index / `--outSAMattributes All`, merges 3 reps/condition into
  `c1.bam`/`c2.bam`, exactly like the real benchmark.

### Bugs found and fixed along the way
- `sim/catalogue.py` coordinate sign bug (above).
- `polyester`'s `illumina5` empirical error model only has per-position
  error rates for positions 0-100 — a 101bp read needs position 101, which
  doesn't exist (the package's own error message says "reads < 102bp",
  which is off by one). Pinned `readlen=100` for now instead of the real
  data's 101bp. Revisit if this matters (could also try `error_model =
  "uniform"` at readlen=101).
- STAR's sorted-BAM output opens `threads x outBAMsortingBinsN` file
  descriptors; default `ulimit -n 1024` isn't enough at
  `--runThreadN 24`. `sim/align_reads.sh` now sets `ulimit -n 65536`.

### Pilot run result (verified working end to end)
On pasteur: 6205 transcripts, ~4.4M total simulated fragments, 6 samples
(cond1/cond2 x 3 reps) -> polyester FASTA reads -> STAR alignment -> merged
`c1.bam`/`c2.bam`. STAR stats: 98.45% uniquely mapped, 1.53% multi-mapped,
GT/AG-dominant canonical splicing (594137 GT/AG vs 7943 GC/AG, 37 AT/AC,
1088 non-canonical out of 603205 total splices) — realistic biology, not an
artifact. Ran `dsignal.extract.extract_event` against these simulated BAMs
for 20 positive design events: works cleanly, multi-junction/multi-isoform
signal present as expected from the confounder design (n_junctions 3-33 per
event, consistent with "every other transcript of the gene active
simultaneously").

**Not yet checked**: rigorous input-PSI vs. post-alignment-observed-PSI
recovery (the printed check used a "top junction share" proxy, not a real
inclusion-vs-exclusion PSI computed from the extractor's junction identity
— worth doing properly before trusting the full pipeline for training data).

## Environment state (pasteur, remote machine, offloaded per user instruction — no local GPU there)

Working directory: `/data/conda/ciccolella/deepSpecas-v2/` (on `/data`,
1.8TB free; **do not** use `/data/deepSpecas-v2` at root, not writable).

Conda envs (via `/opt/miniforge3/bin/mamba`, prefix
`/data/conda/ciccolella/envs/`):
- `DSpoly2v2` — python: pysam (pip-installed, not via mamba — a mamba
  install pulled in an unrelated statsmodels/python-version conflict),
  numpy, pandas, STAR 2.7.10b (`STAR-avx2` binary), samtools, SUPPA2
  (`suppa.py`).
- `DSsim` — R 4.3 (unpinned resolve), `bioconductor-polyester` 1.38.0,
  `r-remotes`, `r-logspline`, `bioconductor-limma`, `r-data.table`,
  `gffread`. **Note**: this is plain Bioconductor `polyester`, not yet the
  ASimulatoR fork (`biomedbigdata/polyester`, adds `pcr_rate`/
  `pcr_lambda`/`adapter_contamination`/`adapter_sequence`) — that fork was
  supposedly installed in an earlier session but did not persist (no R env
  existed at all when this session checked); if noise layer 5 needs those
  extras, install the fork on top of `DSsim` via
  `remotes::install_github("biomedbigdata/polyester", dependencies=FALSE, upgrade="never")`.

Reference data: `ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa` +
`ref/Homo_sapiens.GRCh38.99.gtf` (Ensembl release 99, downloaded fresh from
`ftp.ensembl.org` this pipeline run — matches the STAR index/annotation
vintage). `star_index/` built from these, `sjdbOverhang 100`.

SUPPA catalogue: `sim/events/events_{SE,A3,A5,AF,AL,MX,RI}_strict.{ioe,gtf}`.

Pilot artifacts (all under `sim/`, on pasteur only — **not yet copied
locally**, and shouldn't be as raw BAMs/FASTA since training happens
locally per user instruction, not on pasteur which has no GPU):
- `sim/transcript_lengths.tsv`, `sim/gene_spans.tsv`
- `sim/design.tsv` (360-row pilot design, seed 0)
- `sim/experiment/{count_matrix.tsv,transcript_ids.txt,psi_check.tsv,transcripts.fa,transcripts.ordered.fa}`
- `sim/experiment/reads/sample_0{1..6}_{1,2}.fasta`
- `sim/experiment/aligned/{c1,c2}.bam(.bai)`, per-replicate STAR output dirs
  + logs

`dsignal/extract.py` + `dsignal/__init__.py` were copied to pasteur
(`/data/conda/ciccolella/deepSpecas-v2/dsignal/`) for the PSI-recovery
check above — that's a manual copy for testing, not part of the repo's
normal deployment; keep in mind the pasteur copy can drift from the git
repo if `dsignal/extract.py` changes locally.

## Key constraint to remember

**Training must happen locally (this machine has the GPU; pasteur does
not).** Simulation/alignment (polyester, STAR) belongs on pasteur for
RAM/disk/CPU reasons, but once we're extracting per-event training
records with `dsignal/extract.py`, those (small `.npz` files) need to come
back to this machine — never the raw BAMs/FASTA/FASTQ, which are large and
irrelevant once extracted.

## Immediate next steps (not started)

1. Do a proper input-PSI vs. post-STAR-observed-PSI check: for each
   positive design event, use the event's actual `inclusion_transcripts`/
   `exclusion_transcripts` (from `sim/catalogue.py`) mapped to real
   junction coordinates (not just "top junction share"), and compare
   against `target psi1/psi2` from `sim/design.tsv`. This is the real
   validation that layer 1 (simulate -> real STAR alignment) preserves the
   PSI signal, before trusting it for anything downstream.
2. Extract `.npz` training records for the 360 pilot events via
   `dsignal/extract.py`'s CLI (needs a `region,bam1,bam2` CSV built from
   `sim/design.tsv` — all rows share the same `c1.bam`/`c2.bam`, unlike the
   real benchmark's per-event BAM pairs) and pull them back to this
   machine.
3. Score a first-pass model (even something trivial) against the extracted
   pilot records to sanity-check the label signal is learnable at all,
   before scaling up.
4. Then continue noise layers 2-5 (pre-mRNA background, neighbouring/
   antisense genes, novel unlabelled junctions, PCR-dup/adapter via the
   ASimulatoR fork), window randomization, hard-negative refinement, and
   split-by-gene — each layer benchmark-scored incrementally per the plan.
5. Scale the pilot design up once the above is trusted (pilot was
   deliberately small — 360 events — per explicit user choice to validate
   cheaply before committing to a full-scale run).

## User's stated preference

Explicitly said: "I like what you did till now" — the incremental
build-verify-commit-smoke-test discipline (catalogue -> abundance ->
selection -> experiment assembly -> polyester -> STAR, with a real bug
caught at each stage before scaling up) is the approach to keep using
going forward, not just for Phase 2 but Phase 3+ too.
