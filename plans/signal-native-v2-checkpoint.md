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

## Update (2026-08-13, later session) — steps 1-3 done

1. **Rigorous PSI-recovery check** — `sim/psi_recovery_check.py`. Parses
   each SUPPA event_id's raw coordinate pairs directly (not the
   min/max-only `Event.region_start/end`) and derives real inclusion/
   exclusion junction coordinates per type (SE: two flanking junctions vs.
   a derived skip junction; A3/A5: first listed pair = inclusion per
   SUPPA's own source, second = exclusion; RI: retention estimated from
   non-split coverage over the intron interior vs. the spliced junction).
   **Bug found**: SUPPA's own source comment claims the RI event_id's
   middle coordinate pair is already intron-bounds with "no offset" —
   checked against real STAR jI-tagged junctions in the pilot BAMs and
   found it's actually exon-boundary style needing the same +1/-1
   conversion as SE/A3/A5. Fixed; RI results then matched the other three
   types. Result across 240 positive pilot events (480 psi1/psi2 values):
   no systematic incl/excl sign bias per label (mean signed error -0.02 to
   +0.10), mean abs error 0.13-0.18 per label, overall Pearson r=0.73
   target-vs-observed. Confirms layer 1 (polyester -> real STAR 2.7.10b
   realignment) preserves the PSI signal (the ~0.03 mean abs error at the
   count-matrix level plus ~0.13 of realistic alignment/depth noise).
   Output: `sim/psi_recovery.tsv` (committed).
2. **Extracted `.npz` training records** — `sim/build_extract_csv.py`
   builds a `region,bam1,bam2` CSV from `sim/design.tsv` (all 360 rows
   share the pooled `c1.bam`/`c2.bam`), ran through `dsignal/extract.py`'s
   CLI on pasteur, pulled the 360 resulting `.npz` files (2.5MB total) back
   to this machine and committed them under `sim/experiment/npz/`, with
   `sim/experiment/labels.tsv` (ix/event_id/label/kind) alongside.
3. **Trivial sanity-check model** — `sim/baseline_model.py`. Hand-engineered
   summary features (junction count, top/second junction depth share,
   junction-depth ratio, coverage mean/CV, zero-coverage fraction,
   skip-to-coverage ratio, per condition + cross-condition diffs; 24
   features total), 5-fold CV, logistic regression + random forest.
   Random forest: 55% accuracy / 0.48 macro-F1 on 5 classes (120 NN + 60
   each A3/A5/CSE/RI). RI (F1=0.84) and NN (F1=0.68) separate cleanly, CSE
   decent (F1=0.48), A3 vs A5 is the weak point (confused with each other
   about as often as correct — expected, since these hand-features don't
   encode splice-site directionality). Confirms the label signal survives
   simulation + real STAR realignment and is learnable; clears the way to
   build the real signal-native model rather than these placeholder
   features.

## Update (2026-08-13, later session) — noise layer 2 done (pre-mRNA background)

Plan: `plans/signal-native-v2-noise-layer2.md`. Implemented in-session per
explicit user instruction ("Save and this time implement as well"),
deviating from the repo's default plan-then-separate-session convention for
this one item.

1. `sim/build_premrna_fasta.py` — for each of the 360 pilot design genes,
   extracts the full genomic span (1-based inclusive `sim/gene_spans.tsv`
   coords -> 0-based half-open `pysam.FastaFile.fetch`, reverse-complemented
   on minus strand) as a `PREMRNA_<gene_id>` FASTA entry. Not gffread-based
   (unlike every other transcript here) since it's not an annotated
   transcript. Verified: 360/360 genes matched, every sequence length
   exactly equals its `gene_spans.tsv` span length.
2. `sim/abundance.add_premrna_row` — appends a `PREMRNA_<gene_id>` count row
   to a per-gene count matrix at a log-uniform 1-10%-of-gene-depth fraction
   (drawn once per gene), NB noise per replicate matching every other
   transcript's noise model. `sim/build_experiment.py` wires this in by
   default for every design row regardless of kind (`--no-premrna` to fall
   back to layer-1-only). Verified: drawn `premrna_frac` mean 3.8% (range
   0.7-12%, consistent with the log-uniform target plus NB noise) across
   360 genes.
3. Re-ran the full layer-1 pipeline on pasteur with the pre-mRNA rows
   included (same 360-event design, seed 0, same fraglen=250/fragsd=25/
   readlen=100 as layer 1 -- the fraglen/fragsd values weren't logged from
   the original layer-1 run, so they were re-derived from layer-1's own
   simulated-read mate-pair coordinates: measured mean 250.0, sd 24.9 from
   20k reads). STAR stats: 97.9-98.4% uniquely mapped, GT/AG-dominant
   splicing -- same profile as layer 1. Layer-1 outputs preserved on pasteur
   at `sim/experiment_layer1/` before being overwritten in place.
4. Re-ran `dsignal.extract` (360/360 events) and `sim/baseline_model.py`
   (same RF/5-fold-CV setup) on the new BAMs; pulled `.npz` + labels back
   locally, committed as `sim/experiment/npz_layer2/` +
   `sim/experiment/labels_layer2.tsv`, plus `sim/psi_recovery_layer2.tsv`.
   Apples-to-apples vs layer 1 (RF, per-class F1): RI 0.842->0.818, NN
   0.679->0.642, CSE 0.476->0.420, A3 0.240->0.231 (flat), A5 0.184->0.191
   (flat); overall acc 0.553->0.519, macro-F1 0.484->0.461. Junction-level
   PSI-recovery mean-abs-error moved the same direction, RI worst-hit
   (0.177->0.213). Matches the plan's prediction: RI degrades most (its
   defining signal is intronic coverage, which pre-mRNA background
   directly dilutes), CSE/NN degrade moderately (uniform intronic noise
   hurts coverage-shape-dependent classes generally), A3/A5 near-flat
   (junction-identity-based, less sensitive to background coverage).
   Confirms noise layer 2 is adding real, signal-relevant noise rather than
   inert padding -- clears it to keep going with layer 3.

Local GPU-side env note (not previously recorded): this machine's own
DSpoly2 conda env lives at `/datasets/deepSpecas-v2/envs/DSpoly2` (mambaforge
at `/home/ciccolella/mambaforge`), separate from pasteur's `DSpoly2v2`/
`DSsim` envs -- used here to run `sim/baseline_model.py` locally against the
pulled-back `.npz` files.

## Immediate next steps (not started)

1. Continue noise layers 3-5 (neighbouring/antisense genes, novel
   unlabelled junctions, PCR-dup/adapter via the ASimulatoR fork), window
   randomization, hard-negative refinement, and split-by-gene -- each layer
   benchmark-scored incrementally per the plan, same pattern as layer 2
   above.
2. Scale the pilot design up once noise layers are trusted (pilot was
   deliberately small — 360 events — per explicit user choice to validate
   cheaply before committing to a full-scale run).
3. Design and build the actual signal-native model architecture (Phase 3)
   — the baseline_model.py hand-features were only a sanity check, not a
   candidate architecture.

## User's stated preference

Explicitly said: "I like what you did till now" — the incremental
build-verify-commit-smoke-test discipline (catalogue -> abundance ->
selection -> experiment assembly -> polyester -> STAR, with a real bug
caught at each stage before scaling up) is the approach to keep using
going forward, not just for Phase 2 but Phase 3+ too.
