# signal-native-v2 — Phase 2 noise layer 4: novel unlabelled junctions

Implements the fourth item of `plans/signal-native-v2.md` Phase 2 step 3
("Noise layers, added one at a time with a benchmark score after each").
Layers 1-3 (real STAR realignment, pre-mRNA background, neighbouring/
antisense genes) are done and verified — see
`plans/signal-native-v2-checkpoint.md`. This layer adds low-abundance
transcripts carrying a splice junction that isn't in the annotation at all
(shifted donor/acceptor, or a same-transcript exon skip not present in any
annotated isoform) — real data always has some background of this kind
(mis-splicing, noise, incomplete annotation), and unlike layers 1-3 these
transcripts are never given a label.

Per the checkpoint's layer-3 finding: neighbouring-gene junctions already
degraded A3/A5 hard because they're a *direct* source of spurious junction
identity, not just coverage-shape confusion. Novel junctions injected right
at genes already in the experiment are expected to hit A3/A5 even harder,
since they can land inside the design gene's own extraction window right
next to the real event.

## Steps

1. **Exon-structure extraction.** `sim/gtf_transcript_lengths.py` currently
   sums exon length per transcript and discards exon boundaries. Add
   `sim/gtf_exon_structure.py`: parse the same GTF, yield per transcript the
   ordered list of exon `(start, end)` genomic intervals (1-based inclusive,
   GTF convention) plus `(gene_id, chrom, strand)`. Simulation-only input,
   same as the existing GTF-derived tables. Restrict output to genes already
   present in the experiment (design + neighbour genes, from
   `sim/design.tsv` + `sim/gene_neighbors.tsv`) to keep this cheap rather
   than parsing exon structure genome-wide.
   *Tier: cheap model — mechanical extension of the existing GTF exon loop.*

2. **Novel-junction transcript generator.** New `sim/novel_junctions.py`.
   For each gene in the experiment, pick one of its real transcripts (the
   one with the most exons, to maximize how often a splicing operation is
   even possible) and synthesize one novel-junction variant by one of two
   operations, chosen per gene:
   - **Shifted splice site**: pick a real intron of the chosen transcript,
     shift its donor *or* acceptor boundary by a random offset in
     [-20, -3] union [3, 20] bp (never 0, matching the plan's "±3-20bp"),
     producing a genuinely different intron. Reject the pair if it would
     make the exon length go <= 0.
   - **Within-transcript exon skip-join**: pick two non-adjacent exons of
     the chosen transcript (skipping >= 1 exon between them) and splice
     directly from the end of the first to the start of the second —
     equivalent to an exon-skipping event that was never annotated as a
     transcript. Cross-transcript exon joins (joining exons from two
     different transcripts of the same gene) were considered and are more
     realistic/aggressive, but deferred — noted as a candidate follow-up,
     not built now.
   Assemble the resulting spliced sequence directly from per-exon genomic
   fetches (`pysam.FastaFile`), concatenated in transcript order,
   reverse-complemented as a whole on minus strand — same approach as
   `build_premrna_fasta.py`'s single-segment fetch, extended to multiple
   exon segments. Emit as `NOVEL_<gene_id>_<op>` in a FASTA + a TSV record
   of what was done (gene, transcript used, operation, coordinates) for QC.
   One novel variant per gene for the pilot (not "1-2" — keeps the pilot
   change auditable per-gene; can revisit density later).
   *Tier: strong model — splice-boundary and exon-order arithmetic is easy
   to get subtly wrong, and correctness here is the entire point of the
   layer (the injected junction must land exactly where intended, not
   accidentally reproduce an annotated one).*

3. **Abundance.** `sim/abundance.add_novel_junction_row`: append the novel
   transcript's count row to a gene's count matrix at low abundance — a
   log-uniform fraction of that gene's total depth, deliberately lower than
   layer 2's pre-mRNA range (e.g. 0.5-3%, vs. pre-mRNA's 1-10%), same
   per-replicate NB draw as every other row. Applies uniformly to design
   genes and neighbour genes (matching how pre-mRNA background was applied
   to both in layers 2-3).
   *Tier: cheap model — direct copy of the `add_premrna_row` pattern.*

4. **Wire into `build_experiment.py`** via `--novel-junctions` flag (parallel
   to the existing `--neighbors` flag): after each gene's matrix (design row
   or neighbour row) is built, optionally append its novel-junction row.
   Extend FASTA assembly (either a new `sim/build_novel_fasta.py` mirroring
   `build_premrna_fasta.py`, or a `--novel-junctions` option added to it) to
   emit the `NOVEL_<gene_id>_<op>` sequences from step 2's output alongside
   the pre-mRNA sequences.
   *Tier: cheap model, mechanical wiring matching the layer-3 pattern.*

5. **Re-run the full pipeline on pasteur.** Same 360-row pilot design, same
   fraglen/fragsd/readlen as layers 1-3, seed 0. Produces new
   `c1.bam`/`c2.bam`.
   *Tier: cheap model — same driver scripts, no new logic.*

6. **Re-run PSI recovery + extraction + baseline scoring, compare to layer-3
   numbers.** `sim/psi_recovery_check.py`, `dsignal/extract.py`,
   `sim/baseline_model.py` against the new BAMs; pull `.npz`/labels back
   locally, commit as `sim/experiment/npz_layer4/` +
   `sim/experiment/labels_layer4.tsv` + `sim/psi_recovery_layer4.tsv`.
   Compare per-label F1 against layer 3 (RF: 43.6% acc / 0.354 macro-F1;
   RI 0.719, NN 0.565, CSE 0.288, A3 0.087, A5 0.110). Expect A3/A5 to
   degrade further and hardest of all layers so far (direct junction-identity
   confusion, landing inside genes already in the design). Also worth
   checking whether shifted-splice-site variants and skip-join variants
   have visibly different effects, given the .tsv from step 2 records which
   operation was used per gene.
   *Tier: strong model.*

7. **Commit incrementally per stage** (exon structure -> generator ->
   abundance -> assembly/FASTA -> rerun/results); update the checkpoint file
   with the noise-layer-4 result, same discipline as layers 1-3.

## Open questions / assumptions flagged to the user

- Cross-transcript exon joins (joining exons from two different transcripts
  of the same gene) are more realistic but deferred to a future layer/pilot
  per explicit user choice — within-transcript skip-joins only, for now.
- Novel-junction abundance range (0.5-3% of gene depth) is a guess, lower
  than pre-mRNA's 1-10% since novel/mis-spliced junctions should be rarer
  than an incomplete-splicing background — revisit if the benchmark
  contrast looks wrong.
- One novel variant per gene (not per plan's plural "transcripts") for
  pilot auditability; revisit density once this layer's contribution is
  scored.

## Status

Approved by user 2026-08-14 (implement now, within-transcript exon joins
only for this pilot); implemented in the same session per explicit user
instruction, same deviation from the repo's default plan-then-separate-
session convention as layers 2-3.
