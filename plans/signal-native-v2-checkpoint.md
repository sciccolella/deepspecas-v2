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

## Immediate next steps (not started)

1. Hard-negative refinement and split-by-gene (plan Phase 2 steps 6-7).
2. Scale the pilot design up once noise layers + window randomization are
   trusted (pilot was deliberately small — 360 events — per explicit user
   choice to validate cheaply before committing to a full-scale run).
   When that happens, the full contribution table (layers 1-5 + window
   randomization) should be regenerated in one pass under the current
   code, making the layer-2/3 RNG discontinuity (noted in the RNG-fix
   update above) moot.
3. Design and build the actual signal-native model architecture (Phase 3)
   — the baseline_model.py hand-features were only a sanity check, not a
   candidate architecture.

## User's stated preference

Explicitly said: "I like what you did till now" — the incremental
build-verify-commit-smoke-test discipline (catalogue -> abundance ->
selection -> experiment assembly -> polyester -> STAR, with a real bug
caught at each stage before scaling up) is the approach to keep using
going forward, not just for Phase 2 but Phase 3+ too.
