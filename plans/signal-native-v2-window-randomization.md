# signal-native-v2 — Phase 2 step 5: window randomization

Implements `plans/signal-native-v2.md` Phase 2 step 5: "jitter region
boundaries, vary flank 0.1x-10x the event span, never centre the labelled
event reliably, cover the full real-benchmark width range." All five noise
layers (Phase 2 step 3) are done and scored (`sim/experiment/npz_layer5/`
is the current baseline this is scored against).

## Why this matters

Every extraction window used so far is exactly the SUPPA event span
(positives/hard negatives) or the full gene span (plain negatives) --
`sim/gene_neighbors.py`'s docstring already flags this as a placeholder.
Without jitter, a model can learn window geometry (e.g. "there's always a
junction near byte 0") instead of the actual splicing signal. Real
benchmark region widths (measured directly from `benchmark/{A3,A5,ES,IR}.csv`)
span 1bp-181,765bp, median ~3062bp -- the simulated windows should cover a
comparable range.

## Steps

1. **`sim/window.py`** (new). `sample_window(event_start, event_end,
   chrom_len, rng) -> (win_start, win_end)`: draws a flank ratio
   log-uniform in `[0.1, 10]` of the event span, splits it asymmetrically
   left/right (e.g. two independent log-uniform draws, not a single
   symmetric flank), clips to `[1, chrom_len]`. Applies uniformly to all
   three row kinds (positive, nn_hard, nn_plain) using each row's own
   `region_start`/`region_end` as the base span -- for `nn_plain` that's
   already the full gene span, so its window can also end up wider or
   narrower than the gene.
   *Tier: strong model -- the sampling distribution and centering-skew
   logic are the judgment call this whole step is about.*
2. **`sim/select_events.py`** -- spawn a fifth independent RNG stream
   (`rng_window`) from the same `SeedSequence` used for
   `rng_design`/etc. (matches the existing per-layer RNG-isolation fix), and
   add `win_start`/`win_end` columns to `design.tsv` output. Leave
   `region_start`/`region_end` (true event span) untouched --
   `sim/psi_recovery_check.py` needs those to parse real junction
   coordinates from the SUPPA `event_id`, independent of the extraction
   window. Needs chromosome lengths (from `sim/gene_spans.tsv`'s max gene
   end per chrom is not reliable enough; use `samtools faidx` `.fai` or
   `pysam.FastaFile.get_reference_length` against the reference FASTA on
   pasteur).
   *Tier: cheap model -- mechanical wiring once `sim/window.py` exists.*
3. **`sim/gene_neighbors.py`** -- search neighbours using each row's actual
   `win_start`/`win_end` instead of `region_start`/`region_end` ± a fixed
   10kb placeholder. Drop the `--flank` arg (or repurpose it as an
   additional pad beyond the window, defaulting to 0).
   *Tier: cheap model.*
4. **`sim/build_extract_csv.py`** -- extraction region string becomes
   `win_start-win_end`, not `region_start-region_end`.
   *Tier: cheap model.*
5. **Re-run on pasteur**: regenerate neighbours (window changed, so the
   neighbour gene set shifts) -> pre-mRNA + novel-junction FASTA for the
   new neighbour set -> polyester (reuse layer 5's `pcr_rate=0.1
   pcr_lambda=1 adapter_contamination=TRUE` settings) -> real STAR
   realignment -> `dsignal.extract` -> `sim/psi_recovery_check.py` ->
   `sim/baseline_model.py`. Pull `.npz`/labels back locally, commit as
   `sim/experiment/npz_window/` + `sim/experiment/labels_window.tsv` +
   `sim/psi_recovery_window.tsv`, plus `sim/design.tsv` regenerated with
   the new columns and `sim/gene_neighbors.tsv` regenerated.
   *Tier: cheap model -- driver/rerun scripting, same shape as prior
   layers.*
6. **Verification**:
   - Window-width histogram against the real 1bp-182kb / median-3kb range
     measured above.
   - Centering-offset check: fraction of windows where the event midpoint
     is off-center in the window beyond some small tolerance, confirming
     "never reliably centred" actually holds (not just close-to-centred by
     construction).
   - Compare `baseline_model.py` per-class F1 and
     `psi_recovery_check.py` mean-abs-error against `npz_layer5` /
     `psi_recovery_layer5.tsv`. Expect degradation concentrated in
     coverage-shape-dependent classes (RI/CSE) since window boundaries no
     longer align with event edges; A3/A5 (junction-identity-based) should
     be comparatively more robust, similar to the layer-2/pre-mRNA
     pattern.
   *Tier: cheap-model scripts, strong-model interpretation of the
   resulting deltas (same split as every prior layer's writeup).*

## Out of scope for this step

Hard-negative refinement and split-by-gene (Phase 2 steps 6-7), pilot
scale-up, and the real model architecture (Phase 3) -- tracked separately
in the checkpoint's "immediate next steps" list.
