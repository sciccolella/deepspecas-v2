# signal-native-v2 — Phase 2 noise layer 2: pre-mRNA background

Implements the second item of `plans/signal-native-v2.md` Phase 2 step 3
("Noise layers, added one at a time with a benchmark score after each").
Layer 1 (real STAR realignment) is done and verified — see
`plans/signal-native-v2-checkpoint.md`. This plan adds layer 2: an unspliced
("pre-mRNA") background transcript per gene, expected to *lower* the RI
score (that's the check that the layer is doing something real).

## Steps

1. **Extract per-gene genomic span sequences.** For each of the 360 pilot
   design genes, pull the full genomic interval from
   `ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa` using `sim/gene_spans.tsv`
   (already built on pasteur), reverse-complementing on the minus strand.
   Write as new FASTA entries (`PREMRNA_<gene_id>`) to fold into
   `transcripts.fa`.
   *Tier: cheap model — mechanical FASTA extraction, coordinates already
   computed.*

2. **Design the pre-mRNA abundance draw.** Distribution for "1-10% of gene
   abundance" (log-uniform per gene, independent NB noise per replicate
   matching the style of the rest of `sim/abundance.py`), implemented as a
   new function there. Modelling judgment call, not just plumbing.
   *Tier: strong model.*

3. **Extend the count-matrix / experiment assembly.** Modify
   `sim/build_experiment.py` to add the `PREMRNA_<gene_id>` rows (transcript
   id + count matrix row) for every design gene, wired to the abundance
   function from step 2. Everything else (labelled transcripts, other-isoform
   confounders) unchanged.
   *Tier: strong model.*

4. **Re-run simulation + real STAR realignment on pasteur.** Same drivers as
   layer 1 (`sim/run_polyester.R`, `sim/align_reads.sh`), same pilot design
   (360 events), now with the pre-mRNA rows included. Produces new
   `c1.bam`/`c2.bam`.
   *Tier: cheap model — same scripts, no new logic, just execution.*

5. **Re-run PSI recovery + extraction + baseline scoring, compare to
   layer-1 numbers.** `sim/psi_recovery_check.py`, `dsignal/extract.py`,
   `sim/baseline_model.py` against the new BAMs; pull `.npz`/labels back
   locally. Compare per-label F1 (especially RI) against the layer-1-only
   pilot result (RF: 55% acc / 0.48 macro-F1, RI F1=0.84) to confirm the
   expected RI drop and that nothing else broke.
   *Tier: strong model.*

6. **Commit incrementally per stage** (sequence extraction -> abundance/
   matrix -> rerun/results); update the checkpoint file with the
   noise-layer-2 result, same discipline as layer 1.

## Status

Approved by user 2026-08-13; implementing in the same session (deviation
from the repo's default plan-then-separate-session convention, per explicit
user instruction).
