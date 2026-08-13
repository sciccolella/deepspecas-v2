# signal-native-v2 — Phase 2 noise layer 3: neighbouring / antisense genes

Implements the third item of `plans/signal-native-v2.md` Phase 2 step 3
("Noise layers, added one at a time with a benchmark score after each").
Layers 1 (real STAR realignment) and 2 (pre-mRNA background) are done and
verified — see `plans/signal-native-v2-checkpoint.md`. This layer adds
transcription from genes near (or antisense to) each design gene's
extraction window, so reads from unrelated/overlapping loci can leak into
the region the way they do in real data.

Window randomization (plan Phase 2 step 5) hasn't happened yet — the
extraction window is still exactly `region_start`/`region_end` from
`sim/design.tsv` (event span for positives, full gene span for NN-plain).
So "window plus flank" here uses a fixed placeholder flank rather than the
eventual randomized one; revisit the flank width when window randomization
lands.

## Steps

1. **Find overlapping/neighbouring genes per design row.** New
   `sim/gene_neighbors.py`: for each of the 360 design rows, using
   `sim/gene_spans.tsv` (all 60,676 genes, already built), find every other
   gene on the same chromosome whose span overlaps
   `[region_start - flank, region_end + flank]` (flank fixed at 10kb for
   now). Record strand relationship (`same` vs `antisense`, i.e.
   opposite-strand overlap — the plan calls antisense out specifically) and
   overlap size. Exclude the design gene itself and any gene already used as
   another row's own design gene (`used_genes`, from `sim/design.tsv`) to
   avoid double-simulating or conflicting with a forced event elsewhere.
   Dedup neighbours globally (a neighbour near two design windows is
   simulated once). Output `sim/gene_neighbors.tsv`
   (design_gene_id, neighbor_gene_id, relation, overlap_bp) for QC, plus a
   stderr summary (how many design genes have >=1 neighbour, how many
   distinct neighbour genes total).
   *Tier: cheap model — mechanical interval overlap over an existing table.*

2. **Design the neighbour-gene abundance draw.** Real genes span several
   orders of magnitude of expression; modelling every neighbour at the same
   depth as a design gene would be unrealistic and would swamp the focal
   signal. Add to `sim/abundance.py` (or call existing
   `build_count_matrix_baseline` with a per-neighbour `AbundanceConfig`)
   with `gene_mean_depth` drawn log-uniform per neighbour gene (e.g.
   100-3000, independent of the focal gene's fixed 2000) — same spirit as
   the layer-2 log-uniform pre-mRNA fraction. Also give each neighbour its
   own pre-mRNA background row via the existing `add_premrna_row`, since
   real neighbouring genes transcribe pre-mRNA too. Modelling judgment call
   (depth range, whether antisense genes should be modelled any
   differently from same-strand ones — plan doesn't say, default to no
   difference), not just plumbing.
   *Tier: strong model.*

3. **Extend the count-matrix / experiment assembly.** Modify
   `sim/build_experiment.py`: after building each design row's own matrix
   (unchanged), also build one matrix per deduped neighbour gene (step 1
   output) and concat them into the full transcript x sample matrix once.
   Guard against transcript-id collisions with existing rows (shouldn't
   happen — neighbours are disjoint from `used_genes` by construction — but
   assert it, matching the existing `full.index.is_unique` assert).
   *Tier: strong model — touches the dedup/assembly logic that determines
   correctness.*

4. **Extend pre-mRNA FASTA extraction to neighbour genes.** `sim/
   build_premrna_fasta.py` currently only extracts `PREMRNA_<gene_id>` for
   the 360 design genes; extend its gene list to design genes + neighbour
   genes (same extraction logic, just a longer gene list).
   *Tier: cheap model — mechanical, same code path as layer 2.*

5. **Re-run simulation + real STAR realignment on pasteur.** Same drivers
   as layers 1-2 (`sim/run_polyester.R`, `sim/align_reads.sh`), same pilot
   design (360 rows), now with neighbour-gene transcripts (+ their own
   pre-mRNA rows) included in the count matrix and FASTA. Produces new
   `c1.bam`/`c2.bam`.
   *Tier: cheap model — same scripts, no new logic, just execution.*

6. **Re-run PSI recovery + extraction + baseline scoring, compare to
   layer-2 numbers.** `sim/psi_recovery_check.py`, `dsignal/extract.py`,
   `sim/baseline_model.py` against the new BAMs; pull `.npz`/labels back
   locally. Compare per-label F1 against the layer-2 pilot result (RF: 52%
   acc / 0.46 macro-F1; RI 0.818, NN 0.642, CSE 0.420, A3 0.231, A5 0.191).
   Expect further degradation concentrated in coverage-shape-dependent
   classes (RI, CSE, NN — extra junctions/coverage crossing the window
   boundary from overlapping loci), with A3/A5 relatively flat
   (junction-identity based, per the layer-2 pattern). Also worth checking
   whether any design windows picked up *many* neighbours (dense loci)
   skewing results for a few rows.
   *Tier: strong model.*

7. **Commit incrementally per stage** (neighbour-finding -> abundance ->
   assembly/FASTA -> rerun/results); update the checkpoint file with the
   noise-layer-3 result, same discipline as layers 1-2.

## Open questions to flag to the user before/while implementing

- Is a fixed 10kb flank reasonable as a layer-3 placeholder, or should it
  scale with event span like the eventual layer-5 window randomization
  (0.1x-10x span)? Defaulting to fixed 10kb for simplicity since layer 5
  isn't built yet.
- Neighbour depth range (100-3000 log-uniform) is a guess, not fit to real
  data — flag as an assumption, revisit if the benchmark contrast looks
  wrong.

## Status

Approved by user 2026-08-14 (implement now); implemented in the same
session per explicit user instruction (user was catching a flight and
asked to keep working autonomously and report back when done) — same
deviation from the repo's default plan-then-separate-session convention as
layer 2. See `plans/signal-native-v2-checkpoint.md` for the result.
