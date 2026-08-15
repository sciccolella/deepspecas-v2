# signal-native-v2 — checkpoint (2026-08-14)

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

## Update (2026-08-14, later session) — noise layer 3 done (neighbouring/antisense genes)

Plan: `plans/signal-native-v2-noise-layer3.md`. Implemented in-session per
explicit user instruction ("keep on working without my input or permission
... write me a report" — user was catching a flight), same deviation from
the plan-then-separate-session convention as layer 2.

1. `sim/gene_neighbors.py` — for each of the 360 pilot design rows, finds
   every other gene (from `sim/gene_spans.tsv`, all 60,676 genes) on the
   same chromosome whose span overlaps `[region_start - flank, region_end +
   flank]` (flank fixed at 10kb placeholder — window randomization/layer 5
   isn't built yet), tagged `same`-strand vs `antisense`, excluding genes
   already used as another row's own design gene. Pilot result: 265/360
   design genes have >=1 neighbour, 627 distinct neighbour genes (280
   same-strand pairs, 348 antisense pairs). Output: `sim/gene_neighbors.tsv`
   (committed).
2. `sim/build_experiment.build_neighbor_matrix` + `--neighbors` flag on
   `sim/build_experiment.py` — one background count-matrix row set per
   deduped neighbour gene: plain baseline expression (`build_count_matrix_
   baseline`) at its own log-uniform depth (100-3000, vs design genes'
   fixed 2000 — modelling real between-gene expression heterogeneity) plus
   its own pre-mRNA background row (`add_premrna_row`, same as layer 2).
   Reused existing abundance functions rather than adding new ones.
3. `sim/build_premrna_fasta.py --neighbors` — extends pre-mRNA FASTA
   extraction from design genes to design+neighbour genes (987 pre-mRNA
   sequences total: 360 design + 627 neighbour, 0 missing spans).
4. Re-ran the full pipeline on pasteur: spliced-transcript FASTA via
   gffread on a GTF filtered to the 9360 spliced transcript IDs (design +
   neighbour genes' transcripts) + the 987 pre-mRNA sequences = 10347
   transcripts total, matching `transcript_ids.txt` exactly. Same
   fraglen=250/fragsd=25/readlen=100 as layers 1-2, seed 0. STAR stats:
   97.8-98.2% uniquely mapped, GT/AG-dominant splicing (~97% of splices) —
   same realistic profile as prior layers. ~7.9M total simulated fragments
   (up from ~4.4M, expected given ~1.7x more transcripts).
5. Re-ran `dsignal.extract` (360/360 events) and `sim/baseline_model.py` on
   the new BAMs; pulled `.npz` + labels back locally, committed as
   `sim/experiment/npz_layer3/` + `sim/experiment/labels_layer3.tsv`, plus
   `sim/psi_recovery_layer3.tsv`.
   - Junction-level PSI recovery, mean abs error per label vs layer 2: A3
     0.148->0.157, A5 0.133->0.131 (flat), CSE 0.184->0.176 (flat/slightly
     better), RI 0.213->0.226 (worse, as expected — intronic-coverage-
     dependent).
   - Baseline RF, per-class F1 vs layer 2: A3 0.231->0.087, A5
     0.191->0.110, CSE 0.420->0.288, NN 0.642->0.565, RI 0.818->0.719;
     overall acc 0.519->0.436, macro-F1 0.461->0.354.
   - **Notable deviation from the layer-2 pattern**: layer 2 (pre-mRNA
     background) left A3/A5 nearly flat because it only adds smooth
     intronic coverage, no new junctions. Layer 3 degrades A3/A5
     substantially too, because neighbouring/antisense genes contribute
     real spurious splice junctions that can land inside an extraction
     window and directly confuse junction-identity-based discrimination —
     not just coverage-shape-dependent classes. This is a qualitatively
     different (and real) noise source, not redundant with layer 2 — worth
     keeping in mind when layer 4 (novel low-abundance junctions,
     unlabelled) is added next, since that's an even more direct source of
     junction-identity confusion for A3/A5.

## Update (2026-08-14, later session) — noise layer 4 done (novel unlabelled junctions)

Plan: `plans/signal-native-v2-noise-layer4.md`. Implemented in-session per
explicit user instruction (implement now, within-transcript exon-skip-joins
only for this pilot; cross-transcript joins deferred as a candidate
follow-up).

1. `sim/gtf_exon_structure.py` — per-transcript exon boundaries (not just
   exonic length), restricted to genes already in the experiment (987 =
   360 design + 627 neighbour genes). 9360 transcripts, matching the
   layer-3 spliced-transcript count exactly.
2. `sim/novel_junctions.py` — for each of the 987 genes, picks its
   most-exon-rich transcript and derives one never-annotated-junction
   variant by shifting a donor/acceptor +-3-20bp or dropping an interior
   exon (skip-join); the resulting intron is checked against every
   annotated intron of every transcript of the gene and re-drawn if it
   collides (the whole point is a junction absent from the annotation).
   Result: 838/987 genes got a variant (472 shift, 366 skip-join; 149
   genes failed, presumably too few exons or exhausted retries). Sequence
   assembled directly from per-exon genomic fetches, same revcomp
   convention as the layer-2 pre-mRNA script.
3. `sim/abundance.add_novel_junction_row` — low-abundance background row
   (0.5-3% of gene depth, log-uniform, lower than pre-mRNA's 1-10% since a
   never-annotated junction should be rarer), refactored alongside
   `add_premrna_row` into a shared `_add_low_abundance_row` helper.
   `sim/build_experiment.py` gets a `--novel-junctions` flag, applied to
   design genes and neighbour genes alike.
4. Re-ran the full pipeline on pasteur: same 360-row pilot design, seed 0,
   same fraglen/fragsd/readlen as layers 1-3. Reused layer 3's spliced.fa
   unchanged (verified byte-identical gene/transcript set — no GTF-side
   inputs changed), regenerated premrna.fa (987 seqs) and concatenated with
   the new novel.fa (838 seqs) -> 11185 transcripts total, matching
   `transcript_ids.txt` exactly. STAR stats: 97.3-98.1% uniquely mapped,
   GT/AG-dominant splicing — same realistic profile as prior layers.
   ~7.8M total simulated fragments.
5. Re-ran `dsignal.extract` (360/360), `sim/psi_recovery_check.py`, and
   `sim/baseline_model.py`; pulled `.npz`/labels back locally, committed as
   `sim/experiment/npz_layer4/` + `sim/experiment/labels_layer4.tsv` +
   `sim/psi_recovery_layer4.tsv` + `sim/novel_junctions.tsv`.
   - Junction-level PSI recovery, mean abs error vs layer 3: A3
     0.157->0.155 (flat), A5 0.131->0.141 (flat), CSE 0.176->0.197
     (slightly worse), RI 0.226->0.202 (better).
   - Baseline RF, per-class F1 vs layer 3: A3 0.087->0.156, A5
     0.110->0.149, CSE 0.288->0.389, NN 0.565->0.573, RI 0.719->0.726;
     overall acc 0.436->0.467, macro-F1 0.354->0.399.
   - **Result contradicts the plan's prediction.** The plan expected A3/A5
     to take the biggest hit yet (direct junction-identity confusion, novel
     junctions landing inside genes already in the design). Instead every
     metric went flat-to-better. **Important caveat, not previously
     flagged**: layers 2-4 all share one `np.random.Generator` seeded once
     at 0 in `build_experiment.py`, and each new layer's extra draws (here:
     one novel-junction fraction draw per gene, consumed *before* the
     neighbour-matrix loop) shift every downstream random draw relative to
     the prior layer's run. So layer-3-vs-layer-4 is not a clean ceteris
     paribus comparison — the design events' own PSI/depth realizations
     differ between the two runs for reasons unrelated to novel junctions,
     not just because of the added noise source. This likely explains most
     or all of the counter-intuitive direction. **Before trusting any
     layer's contribution number for the Phase 5 paper table, this needs a
     fix** — e.g. a separate `np.random.Generator` per noise-layer, or
     several reps per layer with different seeds averaged — otherwise the
     "per-layer contribution table" is confounded with RNG-stream drift.
     Flagged, not fixed, in this session.

## Update (2026-08-14, later session) — RNG stream isolation fix

Fixed the confound flagged in the layer-4 update above, per explicit user
instruction ("tackle RNG first then move to noise layer 5").

`sim/build_experiment.py` previously drew every layer (design abundance,
pre-mRNA, neighbour baseline, novel junctions) from one shared
`np.random.Generator` consumed in loop order. Now `main` spawns four
independent generators from one `np.random.SeedSequence(args.seed)` --
`rng_design`, `rng_premrna`, `rng_neighbor_base`, `rng_novel` -- one per
layer, so a layer's on/off state can never perturb another layer's draws.
`build_neighbor_matrix` takes the three relevant generators as separate
params instead of one shared `rng`.

**Verified on pasteur**: ran `build_experiment.py` twice with identical
seed/design/neighbours, once with `--novel-junctions` and once without --
all 10347 shared count-matrix rows and the entire `psi_check.tsv` came back
byte-identical between the two runs (previously they would have differed,
since the novel-junction draws used to shift the neighbour loop's RNG
state). Confirms the fix.

**Regenerated the layer-4 pilot** under the fixed code (same 360-row
design, seed 0, layers 1-4 all active) end to end: polyester -> real STAR
realignment (97.6-98.3% uniquely mapped, consistent with prior layers) ->
`dsignal.extract` -> `psi_recovery_check.py` -> `baseline_model.py`.
Results, committed as `sim/experiment/npz_layer4fix/` +
`sim/experiment/labels_layer4fix.tsv` + `sim/psi_recovery_layer4fix.tsv`:
junction-level PSI recovery mean abs error A3 0.144, A5 0.137, CSE 0.168,
RI 0.200; baseline RF 47.8% acc, macro-F1 0.416 (per-class F1: A3 0.186,
A5 0.143, CSE 0.452, NN 0.579, RI 0.718).

**Important scope note**: this corrected run is the trustworthy baseline
going forward -- noise layer 5 (and anything after) should be compared
against `npz_layer4fix`, not the earlier `npz_layer4` (kept only for
reference, built under the buggy shared-RNG code). The *old*
`npz_layer2`/`npz_layer3` numbers are **not** directly comparable to
`npz_layer4fix` either (entirely different RNG scheme -- SeedSequence
spawning vs. one continued stream), so the full layer-1-through-4
contribution table has a discontinuity at this fix. Re-deriving layer
2/3 under the new scheme was judged out of scope for this fix (would need
two more full pasteur pipeline runs, ~90min each, for numbers that are
superseded anyway once the pilot design is scaled up for the real paper
run) -- flagged here so it isn't forgotten, not resolved.

## Update (2026-08-14, later session) — noise layer 5 done (PCR duplicates + adapter contamination)

Plan: `plans/signal-native-v2-noise-layer5.md`. Implemented in-session per
explicit user instruction ("tackle RNG first then move to noise layer 5").
This is the last noise layer named in the original plan's Phase 2 step 3
list.

1. Installed the ASimulatoR polyester fork (`biomedbigdata/polyester`) into
   the `DSsim` conda env on pasteur via `remotes::install_github`,
   replacing plain Bioconductor polyester. Verified directly (by deparsing
   the installed functions, not trusting secondhand notes) that it adds
   `pcr_rate`/`pcr_lambda` to `generate_fragments` and
   `adapter_contamination`/`adapter_sequence` to `get_reads`, both threaded
   through `simulate_experiment_countmat`'s `...` -> `.check_extras` ->
   `sgseq` call chain.
2. **Fork bug found and worked around**: `simulate_experiment_countmat` has
   `stopifnot(class(readmat) == "matrix")`, but under R>=4 `class()` on any
   plain matrix always returns `c("matrix","array")` regardless of explicit
   `class<-` assignment (confirmed interactively), so that check is
   unconditionally broken on this R version -- unrelated to our own usage.
   `sim/run_polyester.R` now defines `simulate_experiment_countmat_patched`
   locally (byte-identical to the fork's function body except that one
   check loosened to `is.matrix()`) rather than patching the installed
   package file.
3. Extended `sim/run_polyester.R` with optional
   `pcr_rate`/`pcr_lambda`/`adapter_contamination`/`adapter_sequence` CLI
   args, off by default (backward compatible with layer 1-4 invocations).
   Pilot params: `pcr_rate=0.1`, `pcr_lambda=1` (~10% of fragments seeded
   for duplication, each getting `Poisson(1)+1` total copies -- modelling
   assumption, not fit to the benchmark's own duplication rate);
   `adapter_contamination=TRUE` with the package default sequence -- flagged
   as expected to be a near-no-op given our fraglen=250/fragsd=25 (adapter
   readthrough only fires for fragments shorter than readlen=100, a ~6-sd
   event at this fragment-length distribution).
4. Re-ran polyester (reusing the exact same `count_matrix.tsv`/
   `transcripts.fa` as `npz_layer4fix` -- layer 5 only changes the
   read-generation step, not the count matrix) -> real STAR realignment ->
   `dsignal.extract` -> `psi_recovery_check.py` -> `baseline_model.py`.
   Fragment count inflated 7.85M -> 9.42M (+20%, matches the pcr_rate
   prediction exactly). STAR stats slightly lower than layer4fix
   (96.2-96.8% vs 97.6-98.3% uniquely mapped) but still realistic.
   **Independent sanity check**: ran `samtools fixmate`/`sort`/`markdup` on
   `c1.bam` and confirmed a measurable duplicate signal in the aligned
   reads (4.2% duplicate pairs, 393133/9271406) -- the injected PCR
   duplication is visible post-alignment, not just at the pre-alignment
   fragment-count level.
   Results, committed as `sim/experiment/npz_layer5/` +
   `sim/experiment/labels_layer5.tsv` + `sim/psi_recovery_layer5.tsv`:
   junction-level PSI recovery roughly flat vs layer4fix (A3 0.144->0.143,
   A5 0.137->0.132, CSE 0.168->0.170, RI 0.200->0.199). Baseline RF:
   47.8%->48.1% acc (flat), macro-F1 0.416->0.402 (down); per-class F1 A3
   0.186->0.159, **A5 0.143->0.067 (biggest hit)**, CSE 0.452->0.458
   (flat), NN 0.579->0.600 (up), RI 0.718->0.726 (flat). A5 taking the
   biggest hit is a plausible, modest effect: PCR duplicates inflate
   junction/coverage counts non-uniformly (favouring whichever junction/
   isoform happened to seed a duplicate), which can distort the
   ratio-based features (skip-to-cov ratio, junction fractions) the
   baseline model leans on for A3/A5 discrimination more than the
   coverage-shape-dependent classes (RI/CSE/NN). Unlike the original
   (pre-RNG-fix) layer 4 result, this comparison to `npz_layer4fix` is a
   clean ceteris paribus ablation (same count matrix, same RNG-isolated
   code, only the read-generation step changed) -- no confound caveat
   needed here.

All five noise layers from the original plan's Phase 2 step 3 list
(real STAR realignment, pre-mRNA background, neighbouring/antisense genes,
novel junctions, PCR-dup/adapter contamination) are now implemented and
individually scored.

## Update (2026-08-14, later session) — window randomization done (Phase 2 step 5)

Plan: `plans/signal-native-v2-window-randomization.md`. Implemented
in-session per explicit user instruction. This is the last item in Phase
2 step 3-5's list before hard-negative refinement/split-by-gene and
scale-up.

1. `sim/window.py` (new) — `sample_window(event_start, event_end,
   chrom_len, rng)`: total flank drawn log-uniform in `[0.1x, 10x]` the
   event span, split into independent left/right shares (not halved), so
   the event ends up off-center in most draws; clipped to `[1,
   chrom_len]`.
2. `sim/select_events.py` — added `--chrom-lengths` (a `samtools faidx`
   `.fai`) and `win_start`/`win_end` columns to `design.tsv`. **Care
   taken on RNG**: the row-selection `rng` is still exactly
   `np.random.default_rng(args.seed)` (unchanged), and `rng_window` is a
   *separately seeded* generator (`args.seed + 1_000_000`), not spawned
   from a shared `SeedSequence` — spawning was tried first and silently
   produced a different row-selection stream even though window sampling
   runs after selection (spawned children of a `SeedSequence` are not
   equivalent to `default_rng(seed)` itself), which would have broken
   comparability with every prior layer's event/gene selection. Caught by
   diffing the regenerated `design.tsv` against a backup before proceeding
   — confirmed byte-identical selection columns after the fix.
3. `sim/gene_neighbors.py` — neighbour search now uses each row's actual
   `win_start`/`win_end` instead of `region_start`/`region_end` ± a fixed
   10kb placeholder; `--flank` repurposed as an optional pad on top
   (default 0).
4. `sim/build_extract_csv.py` — extraction region string now
   `win_start`-`win_end`. `sim/psi_recovery_check.py` deliberately
   untouched (re-extracts directly from BAMs using the true
   `region_start`/`region_end`, independent of the training-data window).
5. Regenerated on pasteur: `design.tsv` (same 360 events/genes as every
   prior layer, now with windows: width min 134bp / median 6631bp / max
   1.34Mb, vs the real curated benchmark's 1bp-181,765bp / median 3062bp —
   comparable order of magnitude, no runaway multi-Mb windows).
   Centering check: median `|event_mid - window_mid| / window_width` =
   0.076, 63% of windows more than 10% of their width off-center — not
   dramatically skewed, but not reliably centered either. Neighbour search
   under the new windows found fewer genes overall than the old fixed-10kb
   flank (184/360 design genes with >=1 neighbour, 450 distinct neighbour
   genes, vs layer 3's 265/360 and 627) since most windows are narrower
   than 10kb while a few outliers are much wider. Re-ran neighbours ->
   pre-mRNA + novel-junction FASTA for the new gene set (810 genes = 360
   design + 450 neighbour, vs layer 3-5's 987) -> polyester (reused layer
   5's `pcr_rate=0.1 pcr_lambda=1 adapter_contamination=TRUE`) -> real STAR
   realignment (97.6% uniquely mapped, same realistic profile as prior
   layers) -> `dsignal.extract` (360/360) -> `psi_recovery_check.py` ->
   `baseline_model.py`. Committed as `sim/experiment/npz_window/` +
   `sim/experiment/labels_window.tsv` + `sim/psi_recovery_window.tsv` +
   updated `sim/gene_neighbors.tsv` (design.tsv itself not committed, same
   as every prior layer — it's fully reproducible from seed 0 and the
   npz filenames already encode the window used per event).
   - Junction-level PSI recovery (uses the *true* event span, not the
     window, so expected to be roughly flat vs layer 5): A3 0.144->0.154,
     A5 0.132->0.134, CSE 0.170->0.163, RI 0.199->0.196 — flat, as
     predicted.
   - Baseline RF (uses the *randomized-window* `.npz` features, so this is
     where the effect should show up): accuracy 48.1%->35.8%, macro-F1
     0.402->0.296 — a real, substantial drop. Per-class F1 vs layer 5: A3
     0.159->0.146 (flat), A5 0.067->0.099 (flat/up), CSE 0.458->0.304
     (down), NN 0.600->0.473 (down), RI 0.726->0.460 (down, biggest hit).
     Matches the plan's prediction exactly: coverage-shape-dependent
     classes (RI/CSE/NN) degrade the most once window boundaries no
     longer align with event boundaries, while junction-identity-based
     classes (A3/A5) are comparatively robust to window geometry — the
     same qualitative pattern as noise layer 2 (pre-mRNA background).

## Update (2026-08-15) — nn_novel negatives + gene-level split (Phase 2 steps 6-7)

Implemented directly in-session per explicit user instruction (normally this
would go through the plan-file/new-session split, but the user asked to skip
that for this task). Committed as `c456ce4`.

1. `sim/select_events.py` — third NN-negative flavor, `kind="nn_novel"`:
   genes with no catalogued AS event (same eligibility pool as `nn_plain` —
   not in the SUPPA catalogue, >=2 isoforms) that get an elevated
   novel-junction row later in the pipeline instead of the standard
   background rate, so NN can't be shortcut as "no junction noise at all".
   `select_plain_negatives`/`select_novel_negatives` are now both thin
   wrappers around a new `select_constitutive_negatives(..., kind,
   event_suffix)`. New `--n-nn-novel` CLI arg (default 60, matching the
   other three pilot bucket sizes). Selection only needs *eligibility*, not
   the actual novel-junction variant — that's still generated later by
   `sim/novel_junctions.py` over the final design+neighbour gene set, same
   as before, avoiding a chicken-and-egg dependency on exon structure that
   itself depends on design.tsv existing.
2. `sim/build_experiment.py` — `nn_novel` rows get a baseline (no forced
   event) count matrix like `nn_plain`, but when that gene has a generated
   novel-junction variant its fraction is drawn from a new elevated range
   (`--novel-nn-frac-low/high`, default 0.05-0.2) instead of the standard
   layer-4 background range (0.005-0.03, `--novel-frac-low/high`). If
   `generate_gene_variant` found no valid shift/skip-join for that
   particular gene (single-exon transcripts etc.), it silently falls back
   to plain constitutive expression — now counted and reported
   (`nn_novel rows: N/M got an elevated novel-junction row`) rather than
   silently swallowed.
3. New `sim/gene_split.py` — deterministic gene-level train/val/test split:
   `split_genes()` partitions the *distinct gene set* (not rows/events) via
   a seeded shuffle, so a gene with multiple events always lands in one
   split once the pilot's one-event-per-gene selection constraint is lifted
   at scale-up. `assert_no_gene_leak()` is the reusable split-integrity
   check from the top-level plan's Verification section.
4. `sim/baseline_model.py` — `StratifiedKFold` replaced with
   `GroupKFold(groups=gene_ids)` (gene_id recovered from `event_id` via
   `sim.gene_split.gene_id_of`), plus an `assert_no_gene_leak` call on the
   resulting fold assignment. Previously no-gene-leak held only
   incidentally, via `select_events.py`'s one-event-per-gene design
   constraint — not enforced by the CV code itself, so it would have
   silently broken the moment scale-up puts >1 event on a gene.

Verified with synthetic in-memory smoke tests (no gene overlap between
`nn_plain`/`nn_novel` picks, `assert_no_gene_leak` catches an induced leak,
`GroupKFold` keeps a synthetic multi-event gene in one fold), **then run end
to end on pasteur** (`/data/conda/ciccolella/deepSpecas-v2`, see
`memory/reference_pasteur_signal_native.md` for the environment layout —
saved there per explicit user instruction, so future sessions don't have to
rediscover ssh host/conda envs/paths):

1. `select_events.py --n-per-label 60 --n-nn-plain 60 --n-nn-hard 60
   --n-nn-novel 60 --seed 0` -> 420 rows (360 unchanged + 60 new
   `nn_novel`); diffed the first 360 rows against the pre-existing
   window-layer `design.tsv` and confirmed byte-identical, confirming the
   RNG-isolation invariant held (new selection draws only append, they
   don't perturb earlier draws).
2. `gene_neighbors.py` -> 681 pairs / 663 distinct neighbours (up from 450
   at the window layer, tracks the larger design gene set).
3. `gtf_exon_structure.py` + `novel_junctions.py` -> 902/1083 genes got a
   variant; of the 60 `nn_novel` design genes specifically, **58/60** got
   one (2 fell back to plain constitutive expression, reported by
   `build_experiment.py`'s new `nn_novel rows: 58/60 got an elevated
   novel-junction row` line) -- a good hit rate, not mostly-fallback.
4. `build_experiment.py` (defaults, `--neighbors`, `--novel-junctions`) ->
   11209 transcripts, 9.0M simulated fragments.
5. Reassembled `transcripts.ordered.fa` by hand (`gffread` on the
   regenerated `spliced.gtf`/`spliced_ids.txt`, `build_premrna_fasta.py`,
   concatenated with `novel.fa`, then reordered to match
   `transcript_ids.txt` row-for-row -- `simulate_experiment_countmat`
   requires `nrow(readmat) == length(transcripts)`, i.e. positional, not
   name-matched); order-verified by diffing the FASTA header list against
   `transcript_ids.txt` before running polyester.
6. `run_polyester.R` (fraglen=250 fragsd=25 readlen=100 seed=0, layer-5
   `pcr_rate=0.1 pcr_lambda=1 adapter_contamination=TRUE`, same as the
   window layer) -> real STAR realignment (97.2-98.1% uniquely mapped,
   consistent with every prior layer) -> `dsignal.extract` (420/420) ->
   `psi_recovery_check.py` -> `baseline_model.py`. Pulled back to the local
   repo: `sim/experiment/npz_nn_novel/`, `sim/experiment/labels_nn_novel.tsv`,
   `sim/psi_recovery_nn_novel.tsv`, updated `sim/gene_neighbors.tsv`.
   - Junction-level PSI recovery on the unchanged 240 positive events, as
     expected, flat vs the window layer: A3 0.155 (was 0.154), A5 0.126
     (was 0.134), CSE 0.164 (was 0.163), RI 0.198 (was 0.196).
   - **GroupKFold-vs-StratifiedKFold isolation check**: re-ran the new
     `GroupKFold` `baseline_model.py` against the *old* `npz_window`/
     `labels_window.tsv` data first, before touching the new data --
     accuracy 35.8%, macro-F1 0.294, essentially identical to the
     previously-reported `StratifiedKFold` numbers (35.8%, 0.296). Confirms
     what the code comment predicted: the pilot's one-event-per-gene
     selection constraint already made `StratifiedKFold`'s folds
     incidentally gene-disjoint, so this particular swap is a no-op on
     today's data and only starts mattering once scale-up puts >1 event on
     a gene.
   - **`nn_novel` effect** (420-event data, same `GroupKFold` code):
     accuracy 46.2% (was 35.8%), macro-F1 0.271 (was 0.294) -- accuracy up,
     macro-F1 slightly down. Per-class RF F1: NN 0.638 (was 0.478, a big
     jump), RI 0.503 (flat), but **A3 0.077 (was 0.167), A5 0.027 (was
     0.067), CSE 0.108 (was 0.252) all dropped substantially**. Two
     candidate explanations, not disentangled: (a) `nn_novel` genes look
     easier for the model to call NN than the existing `nn_hard`/`nn_plain`
     mix (they have no labelled AS junction pattern to mimic, just a
     background novel junction), which mechanically raises NN's own recall
     and thus overall accuracy while leaving other classes' absolute
     performance unchanged but their relative/macro standing worse; (b)
     with only 60 examples/class split 5 ways (12/fold), `GroupKFold`'s
     specific fold composition is high-variance at this sample size and a
     10-gene reshuffle (663 vs 450 neighbours changes which genes overlap
     which windows, feeding back into extraction) could swing per-class F1
     this much on noise alone. **Not a trustworthy causal read at pilot
     scale** -- flagged here rather than resolved; re-examine once the
     design is scaled up (bigger per-class n stabilizes the CV estimate)
     rather than trying to explain it away now.

## Update (2026-08-15, later) — scale-up to 1260 events (3x pilot)

User picked "3x pilot" (~180/label) over a 10x or full-catalogue run when
asked directly (RI is the limiting catalogue label at ~7300 total events,
so headroom wasn't the constraint — time/compute for a first scale-up step
was). Ran the identical pipeline as the `nn_novel` update above, just with
`--n-per-label 180 --n-nn-plain 180 --n-nn-hard 180 --n-nn-novel 180
--seed 0` (1260 rows: 720 positive + 180 each nn_hard/nn_plain/nn_novel).

- **Not a superset of the 420-event design** — verified and expected: unlike
  the `nn_novel` addition (which only appended new selection calls),
  raising `n_per_label` changes how many genes each label consumes *before*
  later labels are processed in `select_positives`'s dict-iteration order,
  so a scaled-up run is a fresh draw, not a prefix-compatible extension.
  Documented so a future session doesn't mistake the lack of a byte-identical
  prefix for a bug.
- 171/180 `nn_novel` genes got an elevated novel-junction row (95%, better
  hit rate than the 420-event run's 58/60=97%... actually comparable).
- 27.4M simulated fragments (vs 9.0M at 420-event scale, roughly 3x as
  expected). 95.8-96.2% uniquely mapped by STAR (slightly below the pilot's
  97-98%, plausible given more neighbour/novel-junction competition at
  scale, still realistic).
- Junction-level PSI recovery on the 720 positive events: A3 0.144, A5
  0.144, CSE 0.176, RI 0.186 (mean abs err) — same order of magnitude as
  every prior layer, larger n per class (180 vs 60) gives a more stable
  estimate. 2/720 psi values undefined (no reads), noted not investigated
  further (negligible).
- **Baseline RF at 180/class**: accuracy 48.8% (was 46.2% at 420-event
  scale), macro-F1 0.325 (was 0.271). Per-class F1: A3 0.108 (was 0.077),
  A5 0.160 (was 0.027), CSE 0.132 (was 0.108), NN 0.657 (was 0.638), RI
  0.568 (was 0.503) — **every class improved**, A3/A5/CSE recovered most
  of the ground the 420-event run's flagged regression had lost (though
  CSE is still below the pre-`nn_novel` window-layer number of 0.252).
  This is consistent with (and partially resolves) explanation (b) from
  the prior update: the 420-event `GroupKFold` estimate at n=60/class was
  genuinely high-variance, not a real generalizable effect of adding
  `nn_novel` — more data stabilized it in the expected direction rather
  than confirming a real A3/A5/CSE-specific harm. Still a hand-feature
  sanity check, not the real model (Phase 3) — don't over-read absolute
  numbers here either way.
- Pulled back to the local repo: `sim/experiment/npz_scale1260/`,
  `sim/experiment/labels_scale1260.tsv`, `sim/psi_recovery_scale1260.tsv`,
  updated `sim/gene_neighbors.tsv`. `sim/design.tsv` itself intentionally
  not committed (reproducible from seed 0), same convention as every prior
  layer.

## Immediate next steps (not started)

1. Design and build the actual signal-native model architecture (Phase 3)
   — the `baseline_model.py` hand-features were only a sanity check across
   every layer/scale so far, never a candidate architecture. This is the
   next real piece of work; simulation (Phase 2) is now fully implemented
   and verified end to end at 1260-event scale.
2. If/when a bigger simulation scale is wanted for the actual paper run
   (current 1260-event scale-up was deliberately a conservative first
   step, not the final scale — RI's ~7300-event catalogue ceiling leaves
   plenty of room), that's a rerun of the exact same pipeline at a larger
   `--n-per-label`/`--n-nn-*`, not new code.
## User's stated preference

Explicitly said: "I like what you did till now" — the incremental
build-verify-commit-smoke-test discipline (catalogue -> abundance ->
selection -> experiment assembly -> polyester -> STAR, with a real bug
caught at each stage before scaling up) is the approach to keep using
going forward, not just for Phase 2 but Phase 3+ too.

## Update (2026-08-15, later) — Phase 3 done (the model)

Implemented `plans/signal-native-v2-phase3-model.md` steps 0-6 (step 6 is the
code change only; the pasteur batches themselves are not run). Every step
built, verified, smoke-tested and committed before the next started, per the
standing preference.

**Step 0 — local torch env.** `/datasets/deepSpecas-v2/envs/DSnet` via
mambaforge: torch 2.5.1 + working CUDA on the RTX 4080, lightgbm 4.7, sklearn
1.9, pysam 0.24. **Gotcha worth recording**: conda-forge's plain `pytorch`
package resolves to a **CPU-only** build (`Torch not compiled with CUDA
enabled`); `pytorch-gpu` is the package that pulls the CUDA build. Acceptance
test passed exactly: `sim/baseline_model.py` on `npz_scale1260` reproduces
48.8% acc / 0.325 macro-F1 and every per-class F1 in the new env.

**Step 1 — `dsignal/features.py`** (+ `dsignal/verify_features.py`). Shared
builder consumed by both the GBM and the Transformer, so they are compared on
identical information. Per-junction token matrix over the union of
`(start, end)` across conditions: counts/motif, **topology**
(`n_sharing_donor`/`n_sharing_acceptor` + per-condition donor-group and
acceptor-group count fractions) and **coverage context** (flank vs
intron-interior means, interior:flank ratio, interior nonzero fraction, so RI
needs no separate encoder); plus an event-level vector used as the CLS token
and, via `build_gbm_row`, as the LightGBM row. Minus-strand records are
canonicalized by mirroring within the window (motif codes mapped to their
reverse-complement partners), making A3/A5 strand-independent and the flip
augmentation an exact A3<->A5 map.

Two bugs caught by the verification checks:
- `mirror()` sorted rows by start alone, so shared-donor ties came back
  swapped and mirroring was **not an involution** — fixed to a lexicographic
  `(start, end)` sort in both `mirror` and `normalize_record`.
- The check-4 manual slice indexed `cov` with unclipped negative bounds, which
  wrap to the far end of the array. Junctions legitimately fall outside the
  extraction window (a read overlapping it can splice clean out), so every
  coverage lookup must clip *before* indexing — `features.py` already did, the
  check did not.

**Step 2 — `dsignal/baseline_gbm.py`** (the honest-reporting gate). Identical
protocol to `sim/baseline_model.py` (GroupKFold(5) by gene +
`assert_no_gene_leak`, same CLI/report shape); only the feature set changes.

**Step 3-4 — `dsignal/{dataset,model,train}.py`.** `JunctionSetTransformer`,
1.4M params: pre-norm encoder over junction tokens with a **relative-position
bias** (signed `log10(midpoint distance)` in 32 buckets — junction separations
span four orders of magnitude, raw bp would collapse the short range) and a
**relation bias** (per-head scalar for identical / shares_donor /
shares_acceptor / nested / overlapping / disjoint). Attention is written out
rather than using `nn.MultiheadAttention` (the biases are per-head per-pair and
must land on the logits pre-softmax); padded keys get a large *finite* penalty,
not `-inf`, so a fully-padded query row cannot softmax to NaN. The CLS token is
projected from the event vector, never masked, so zero-junction events (9/1260)
are a valid CLS-only forward pass. Augmentations act on the *normalized record*
and features are rebuilt afterwards, so flipping really does recompute
donor/acceptor groups from mirrored coordinates.

### Results on `npz_scale1260`, all vs the standing RF reference

| model | acc | macroF1 | A3 | A5 | CSE | NN | RI |
|---|---|---|---|---|---|---|---|
| RF hand-features (reference) | 0.488 | 0.325 | 0.108 | 0.160 | 0.132 | 0.657 | 0.568 |
| LightGBM, topology features | 0.688 | 0.621 | 0.527 | 0.525 | 0.585 | 0.826 | 0.644 |
| Transformer, no augmentation | 0.725 | 0.687 | 0.678 | 0.634 | 0.636 | 0.816 | 0.672 |
| Transformer, all augmentations | **0.745** | **0.720** | 0.699 | 0.705 | 0.693 | 0.820 | 0.682 |
| *geometry-only control* | *0.463* | *0.415* | *0.306* | *0.280* | *0.353* | *0.640* | *0.497* |

**The paper's central claim is confirmed.** Same protocol, same folds, same
data — only the feature set changed — and A3/A5 F1 go up ~4x at the *LightGBM*
level already. That isolates it: junction identity, not model capacity, was the
missing ingredient. The Transformer adds a further real increment on top
(0.621 -> 0.687 macro-F1 before augmentation), so the set encoder earns its
keep rather than just re-expressing the features.

**Augmentation ablation** (5-fold, same protocol; control = no augmentation):

| run | acc | macroF1 | A3 | A5 | CSE | NN | RI |
|---|---|---|---|---|---|---|---|
| control | 0.725 | 0.687 | 0.678 | 0.634 | 0.636 | 0.816 | 0.672 |
| flip | 0.737 | 0.710 | 0.697 | 0.695 | 0.673 | 0.814 | 0.672 |
| subsample | 0.737 | 0.701 | 0.678 | 0.649 | 0.646 | 0.824 | 0.707 |
| junction_dropout | 0.719 | 0.684 | 0.624 | 0.650 | 0.638 | 0.809 | 0.697 |
| jitter | 0.717 | 0.679 | 0.651 | 0.632 | 0.623 | 0.821 | 0.665 |
| condition_swap | 0.712 | 0.674 | 0.628 | 0.613 | 0.623 | 0.813 | 0.691 |
| **all** | **0.745** | **0.720** | 0.699 | 0.705 | 0.693 | 0.820 | 0.682 |

Flip is the most valuable single augmentation, exactly as the plan predicted
(+0.023 macro-F1, A5 0.634 -> 0.695) — it is the only one that is an *exact*
label-space symmetry rather than a noise model. condition_swap and jitter are
individually neutral-to-slightly-negative, but all-together still wins, so
they are kept.

### **Phase-2 bug found: the geometry-only control fails**

Plan Verification item 2 requires the geometry-only control to score at chance.
It does not: **window geometry alone reaches 46.3% acc / 0.415 macro-F1** with
the Transformer (0.465 / 0.369 with LightGBM) using *no* junction or coverage
information — above the RF reference's macro-F1. Cause: `sim/window.py` draws
each window's flank log-uniform in [0.1x, 10x] the **event span**, so window
width inherits the span, which tracks the label:

| label | median window width |
|---|---|
| RI | 1,452 bp |
| A3 / A5 | 2,563 / 2,728 bp |
| CSE | 17,764 bp |
| NN | 22,734 bp |

The window randomization is doing something real (it cost the RF baseline
48.1% -> 35.8% when it landed) but it does not decorrelate width from label,
and nothing about that shortcut transfers to the real benchmark, whose windows
come from a curator. So `--feature-subset` **defaults to `no-geometry`** in
both `baseline_gbm.py` and `train.py`, and every number in this section is
already geometry-excluded. With geometry included the GBM reaches 0.702 /
0.640 (vs 0.688 / 0.621); that 0.019 macro-F1 gap is the measure of how little
the artifact adds once topology is available. **Fixing the window sampler is
Phase-2 work — flagged, not done here.**

Leakage audit (Verification item 2) otherwise clean: no GTF/annotation input
anywhere in `features.py`/`dataset.py`/`model.py`/`train.py`, `gene_id` is used
only for CV grouping and never as a feature, and `extract.py`'s `% 20` mask
still strips STAR's "annotated in the index's GTF" bit.

### Step 5 — real benchmark, zero-shot sim->real

`dsignal/score_benchmark.py` (the plan's `tools/score_benchmark.py`, which does
not exist in this repo) + `dsignal/predict.py`, over all 140 curated events with
the 5 all-augmentation fold checkpoints softmax-averaged:

- **70/140 = 50.0%**, 5-class *and* A3+A5-collapsed (v1's comparable number is
  123/140 = 87.9%). Per CSV: IR 88.2% (15/17), A3 57.1% (16/28), CSE 41.5%
  (27/65), A5 40.0% (12/30).
- **The collapse buys literally nothing, and that is the interesting part**:
  the 5-class and collapsed numbers are identical because there are **zero
  A3<->A5 cross-errors on the real benchmark** (confusion matrix has 0 in both
  off-diagonal A3/A5 cells). v1's defining failure mode is not merely reduced,
  it is absent — the errors go to NN/RI/CSE instead. On the sim data the same
  thing shows up as A3<->A5 cross-confusion of 6/180 each way.
- The dominant real error is **30/140 events predicted NN** (a false positive
  by construction — the benchmark has no NN events). NN is 43% of the training
  design (540/1260), so this reads as a class-prior/domain mismatch, not a
  topology failure. Second is CSE->A3 (23/65).
- Accuracy is roughly flat across region width (56.7% under 1kb, 42.9% at
  5-20kb, 58.3% at 20-100kb), so this is not a width-scaling failure.
- **Wall clock: 0.031 s/event** (0.024 extract + 0.008 inference), 4.4 s for
  the whole benchmark, against v1's ~17 s/event of image rendering alone —
  roughly a 550x speedup on the end-to-end path.

This is the expected outcome, not a disappointment: it is zero-shot sim->real
with no fine-tuning, and Phase 4 is what closes the gap. Its value is as the
baseline for the Phase-5 "noise realism transfers" ablation.

### Step 6 — scale-up code change only

`sim/select_events.py --exclude-genes` (repeatable; accepts a prior batch's
`design.tsv` or a bare gene-id list) pre-seeds `used_genes` so scale-up batches
select disjoint genes — without it, different seeds re-pick the same genes and
create cross-batch leakage that per-batch `GroupKFold` cannot see. Verified on
both input forms. **The pasteur batches themselves were not run** — that is
pipeline driving, unchanged from the existing scale-up recipe.

### Immediate next steps

1. **Fix the window-geometry leak in `sim/window.py`** (Phase 2): draw window
   width from a distribution independent of event span — e.g. sample width
   from the real benchmark's own width distribution, or a fixed log-uniform
   range — then re-run the geometry control, which should then sit at chance.
2. Phase 4 (sim->real domain adaptation / fine-tuning), which is what the 50%
   zero-shot number exists to be measured against.
3. Optionally run the Step-6 pasteur batches (seeds 1-5, `--exclude-genes`) to
   go from 1,260 to ~7,560 events; the model code needs no change.

## Update (2026-08-15, later) — window-geometry leak fixed (Phase-2 follow-up)

Implemented `plans/signal-native-v2-window-geometry-fix.md`, which closes the
"Fix the window-geometry leak in `sim/window.py`" item above. Each step was
built, verified and committed before the next started.

### Diagnosis (three findings, all confirmed)

1. **The control's accuracy was just window width.** A depth-4 decision tree on
   window width *alone* (widths read off the `.npz` filenames of
   `npz_scale1260`) scores 0.457 acc / 0.235 macro-F1 — the same accuracy as
   the Transformer geometry control's 0.463, and the accuracy lives entirely in
   NN and RI; A5/CSE get nothing from width.
2. **The control's extra macro-F1 was not geometry at all.**
   `dsignal/dataset.apply_feature_subset("geometry")` zeroed feature *columns*
   but left the *number* of junction tokens and their spacing (which the
   attention position/relation biases read) intact. Junction count is
   structure, not geometry, so the control was never the control the plan's
   Verification item 2 asks for. Re-running the *honest* control on the *old*
   data isolates the two effects: Transformer 0.463/0.415 -> 0.444/0.399,
   LightGBM 0.465/0.369 -> 0.399/0.298.
3. **Width<->label correlation is partly real biology and does transfer.** The
   curated benchmark shows the same per-class ordering as the simulation
   (benchmark median region width: RI 943 < A5 1,544 < A3 1,731 < CSE 6,885).
   The genuine artifacts were narrower than "windows correlate with label":
   NN windows came from whole gene spans (median 22,734bp), and sim windows ran
   ~2x wider than benchmark ones because flank was a multiple of event span.

### What changed

- **`sim/window.py`** — window width is now drawn from a log-normal fitted to
  the pooled curated benchmark (all four CSVs pooled, so the draw carries no
  label information: n=140, log10 mean 3.4448, sd 0.7024, truncated by
  rejection to the observed [2, 181,766] range), independent of the event span.
  The event is then placed uniformly at random inside the window, preserving
  the off-center property. A draw narrower than the event span is clamped up to
  span + a log-uniform **absolute** 50-500bp flank per side — the residual,
  irreducible coupling, since a window must contain its event.
  `sample_window_detail` returns that clamp flag; `select_events.py` reports
  the rate. Redrawing from a span-truncated distribution instead of clamping
  was tried and rejected: same leak-test result, but it inflates pooled width
  (log10 mean 3.96 vs 3.78, benchmark 3.44).
- **`sim/select_events.py`** — `nn_plain`/`nn_novel` rows no longer take the
  whole gene span as their region; each gets a random sub-interval of its gene
  whose length is drawn from the pooled positive event-span distribution
  (`shrink_constitutive_regions`). `nn_hard` is untouched (real event span).
  Draws come from their own generator (`seed + 2_000_000`), like the window
  stream, so **the row-selection stream is bit-for-bit unchanged** — verified:
  the regenerated 1260-row `design.tsv` is byte-identical to the previous one
  on event_id/gene_id/label/kind/chrom/strand/psi1/psi2, and the only region
  changes are 139/180 `nn_plain` + 139/180 `nn_novel` rows (the other 41 each
  have genes shorter than their sampled span).
- **`dsignal/dataset.py` + `predict.py`** — the geometry control replaces the
  token set with a fixed 4 identical zero tokens and zeroes the meta
  coordinates the attention biases derive from (`apply_feature_subset_meta`),
  so it tests window geometry and nothing else. `dsignal/baseline_gbm.py`'s
  geometry subset likewise keeps only the two event-level geometry features,
  dropping the token-aggregate geometry columns — the same information the
  Transformer control now gets. **`all` and `no-geometry` are untouched**:
  LightGBM no-geometry (0.688/0.621) and the Transformer no-augmentation
  control (0.725/0.687, every per-class F1) reproduce the table above exactly
  on `npz_scale1260`.

### Re-extraction (no re-simulation, scope decided with the user)

Regenerated `design.tsv` on pasteur -> `build_extract_csv.py` ->
`dsignal/extract.py` against the **existing** scale1260 BAMs -> pulled back as
`sim/experiment/npz_windowfix/` + `labels_windowfix.tsv` (1260/1260 events,
13MB vs the old set's 22MB, consistent with narrower windows).
`select_events.py` reports: window width median 5,816bp, log10 mean 3.801
sd 0.541 (benchmark 3,063 / 3.445 / 0.702), **clamp rate 46.9%**.

**Accepted mismatch**: the neighbour background gene set baked into those BAMs
was chosen under the *old* windows, so some new windows contain genes never
simulated as background and vice versa. Fine for measuring whether the shortcut
is gone; a consistent re-simulation comes with the next scale-up, which is a
rerun of the existing recipe with no new code. It does mean the sim-CV numbers
below are not a perfectly clean ceteris-paribus comparison against the old
ones.

### Results — window width per label

| label | old median width | new median width |
|---|---|---|
| RI | 1,452 | 3,051 |
| A3 / A5 | 2,564 / 2,728 | 5,144 / 4,971 |
| CSE | 17,764 | 12,560 |
| NN | 22,735 | 6,365 |

NN is no longer the widest class by an order of magnitude; CSE is now the
widest, which is also the curated benchmark's ordering (ES 6,885bp). Pooled
log10 width sd drops 0.851 -> 0.541.

### Results — the leak tests (chance references: majority-class 0.429 acc /
0.120 macro-F1; stratified-random 0.264 / 0.197)

| test | old windows | new windows |
|---|---|---|
| width-only decision tree | 0.457 / 0.235 | **0.432 / 0.199** |
| LightGBM geometry control (honest) | 0.399 / 0.298 | **0.351 / 0.260** |
| Transformer geometry control (honest) | 0.444 / 0.399 | **0.364 / 0.331** |

Both controls now score *below* the majority-class accuracy with a diffuse
confusion matrix — they cannot even match "always predict NN". Macro-F1 sits
above stratified-chance (0.331 vs 0.197 for the Transformer), and the residual
is CSE: `apply_feature_subset("geometry")` keeps two features, and splitting
them (LightGBM, new windows) gives width alone 0.332/0.222 and
`frac_junctions_out_of_window` alone 0.400/0.232. **The honest reading is that
the control is at chance on accuracy but retains a weak CSE-vs-rest width
signal**, which is exactly the clamp residual — CSE event spans are genuinely
long, so their windows are genuinely wide, in the simulation and in the curated
benchmark alike. That part is biology, not an artifact, and it is now the same
relationship the real data has.

### Results — the real model on the new windows (5-fold, grouped by gene)

| run | acc | macroF1 | A3 | A5 | CSE | NN | RI |
|---|---|---|---|---|---|---|---|
| LightGBM, no-geometry | 0.658 | 0.578 | 0.536 | 0.424 | 0.580 | 0.831 | 0.519 |
| LightGBM, all features | 0.664 | 0.591 | 0.525 | 0.448 | 0.591 | 0.826 | 0.563 |
| Transformer, no aug | 0.694 | 0.647 | 0.636 | 0.607 | 0.611 | 0.818 | 0.561 |
| Transformer, all aug | 0.712 | 0.674 | 0.695 | 0.637 | 0.628 | 0.824 | 0.585 |
| Transformer, all aug, **all features** | 0.717 | 0.678 | 0.681 | 0.645 | 0.643 | 0.833 | 0.590 |

Against the old windows the no-geometry numbers move -0.03 to -0.05 macro-F1
(Transformer all-aug 0.720 -> 0.674), concentrated in RI (0.682 -> 0.585) and
CSE. That is the expected direction and size: narrower windows give the
coverage-shape classes less flanking context, and part of it is the accepted
neighbour-set mismatch. No large move, so the window change did nothing
unintended.

**`all` vs `no-geometry`, re-opened as the plan asked**: geometry now buys
+0.004 macro-F1 for the Transformer (0.678 vs 0.674) and +0.013 for LightGBM
(0.591 vs 0.578), against +0.019 for LightGBM before the fix. That is inside
run-to-run noise. **Decision: keep `no-geometry` as the default.** The gap no
longer measures a shortcut worth excluding, but it also buys nothing, and
keeping geometry out keeps the reported number provably free of the clamp
residual described above — a free guarantee at no measurable cost.

### Results — zero-shot sim->real benchmark: it got worse, not better

`dsignal/score_benchmark.py` over the 140 curated events, 5 all-augmentation
fold checkpoints (`runs/windowfix_allaug/`) softmax-averaged:

- **60/140 = 42.9%**, 5-class and A3+A5-collapsed alike, **down from 50.0%**.
  Per CSV: IR 70.6% (was 88.2%), ES 43.1% (was 41.5%), A5 36.7% (was 40.0%),
  A3 32.1% (was 57.1%).
- Paired over the 139 events both runs scored: 20 old-correct/new-wrong vs 10
  old-wrong/new-correct, **McNemar exact p = 0.099** — a real directional move
  but not significant at 0.05.
- The dominant error is still false NN, and it got worse: **41/140 NN
  predictions, up from 29** (the benchmark contains no NN events, so every one
  is a false positive).
- **The A3<->A5 result survives**: still exactly zero A3<->A5 cross-errors on
  the real benchmark, so v1's defining failure mode remains absent.
- Wall clock 0.037 s/event, unchanged in character (v1: ~17 s/event of image
  rendering alone).

**This is the honest outcome and it should not be spun.** The fix was expected
to help transfer, on the reasoning that NN's absurd window widths were the
biggest artifact; instead removing them made NN *more* attractive as a
prediction, because constitutive NN regions now look dimensionally like real
curated windows. Two candidate explanations, not disentangled: (a) NN is 43% of
the training design and the benchmark has no NN at all, so any change that
makes NN's window statistics more benchmark-like increases false NN — a
class-prior/domain problem that Phase 4 exists to fix; (b) part of the drop is
the accepted neighbour-set mismatch rather than the window change. Resolving
this needs a consistent re-simulation at the next scale-up, and it is a strong
argument for putting the NN class fraction (or a prior correction) on Phase 4's
list. The fix is still correct on its own terms — the control is at chance and
the reported CV numbers no longer rest on a shortcut — but it did not buy the
zero-shot gain the plan hoped for.

### Verification checklist (from the plan)

- [x] regenerated `design.tsv` selection columns byte-identical
- [x] window width distribution per label vs the benchmark table
- [x] NN window widths no longer the widest class by an order of magnitude
- [x] clamp rate reported (46.9%)
- [x] width-only tree at chance (0.432 acc vs 0.429 majority-class floor)
- [x] geometry-only control at chance on accuracy, with the residual CSE width
      signal reported rather than hidden
- [x] `no-geometry` reproduces the checkpoint table exactly on the old data
- [x] benchmark score reported against the 50.0% baseline (42.9%, p = 0.099)
