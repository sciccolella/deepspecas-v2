# signal-native-v2 — Phase 2 noise layer 5: PCR duplicates + adapter contamination

Implements the fifth (and per `plans/signal-native-v2.md` Phase 2 step 3,
final planned) noise layer: "PCR duplicates and adapter contamination
(ASimulatoR provides both)." Layers 1-4 are done; layer 4 was regenerated
under the RNG-isolation fix (`sim/experiment/npz_layer4fix/`), which is the
baseline this layer is scored against.

## What's confirmed available (checked directly on pasteur, not assumed)

Installed `biomedbigdata/polyester` (the ASimulatoR fork) into the `DSsim`
conda env via `remotes::install_github`, replacing plain Bioconductor
polyester. Inspected the installed package's function bodies directly
(`deparse(get(fn))`) rather than trusting the checkpoint's secondhand note
about what the fork provides:

- `generate_fragments(..., pcr_rate = NULL, pcr_lambda = NULL)`: when
  `pcr_rate` is set, `floor(N * pcr_rate)` fragments are chosen (without
  replacement) as PCR-duplication seeds; each gets `rpois(., pcr_lambda) +
  1` total copies appended (named `PCR_DUP;<original name>`), on top of the
  untouched original fragment set.
- `get_reads(..., adapter_contamination = FALSE, adapter_sequence = NULL)`:
  for fragments shorter than `readlen`, appends `adapter_sequence` (default
  `"CTGTCTCTTATACACATCT"`, a real Illumina Nextera adapter) and truncates
  back to `readlen` -- i.e. it only ever fires for fragments *shorter than
  the read length*.
- Both are threaded through `simulate_experiment_countmat`'s `...` ->
  `.check_extras` -> `sgseq` -> `generate_fragments`/`get_reads` call chain;
  confirmed by reading `sgseq`'s body directly (it passes
  `extras$pcr_rate`, `extras$pcr_lambda`, `extras$adapter_contamination`,
  `extras$adapter_sequence` into the respective calls).

**Important implication for adapter contamination**: our pilot uses
fraglen=250/fragsd=25 (readlen=100) -- at that distribution, a fragment
shorter than 100bp is a ~6 standard-deviation event, i.e. essentially never
happens. So turning `adapter_contamination` on is expected to have **no
measurable effect** at current settings; that's polyester's own correct
mechanism (adapter readthrough is a short-insert phenomenon), not a bug in
our wiring. This layer is being added because the plan calls for it and
because it costs nothing to wire correctly, not because we expect it to
move the benchmark numbers -- the PCR-duplicate rate is the part of this
layer actually expected to matter.

## Steps

1. **Extend `sim/run_polyester.R`** with four new optional CLI args (after
   the existing seed):  `pcr_rate` (default `NA`/off), `pcr_lambda`
   (default `1`, only used if `pcr_rate` is set), `adapter_contamination`
   (default `FALSE`), `adapter_sequence` (default `""`, meaning "use
   polyester's own default sequence" when adapter_contamination is on).
   Forward as named args to `simulate_experiment_countmat(...)` only when
   enabled, so a layer-1-4 rerun with no new args is byte-for-byte the same
   invocation as before (backward compatible).
   *Tier: cheap model -- mechanical CLI/arg-forwarding.*
2. **Pilot parameter choice**: `pcr_rate = 0.1`, `pcr_lambda = 1` -- 10% of
   fragments seeded, each getting `Poisson(1)+1` (mean 2) total copies, so
   roughly +20% library inflation from duplicates, in the range of
   real-world PCR-duplication rates reported by tools like Picard
   MarkDuplicates (commonly 5-30% depending on library complexity/input
   amount). A modelling assumption, not fit to the benchmark data -- flagged
   for the user, revisit if the benchmark contrast looks wrong.
   `adapter_contamination = TRUE` with the package default sequence, for
   completeness (see caveat above about expected near-zero effect at
   current fraglen).
3. **Re-run only the read-generation + realignment steps** (not
   `build_experiment.py` -- layer 5 doesn't touch the count matrix, only
   how reads are generated from it): polyester with the new flags -> STAR
   realignment (same index/settings as layers 1-4) -> `dsignal.extract` ->
   `sim/psi_recovery_check.py` -> `sim/baseline_model.py`, scored against
   `npz_layer4fix` (the RNG-fix baseline), not the older buggy `npz_layer4`.
   *Tier: cheap model for driving, strong model for interpreting the
   result.*
4. **Check duplicate rate actually shows up in the aligned BAMs** as a
   sanity check independent of the classifier score -- e.g. `samtools
   flagstat` or counting exact-duplicate read pairs (same start coordinates)
   pre-markdup, to confirm the injected ~20% duplication rate is visible
   post-STAR-realignment (STAR doesn't dedupe by default, so injected PCR
   duplicates should show up as literal repeated alignments).
5. Commit incrementally (code -> pilot results); update the checkpoint with
   the noise-layer-5 result and the adapter-contamination caveat.

## Open questions / assumptions flagged to the user

- `pcr_rate=0.1`/`pcr_lambda=1` is a guess, not fit to real duplication
  rates in the benchmark BAMs -- could instead measure the benchmark's own
  duplicate fraction (e.g. via `samtools markdup` stats) and match it, but
  that's more work than the pilot warrants right now.
- Per the confirmed mechanism above, adapter contamination is expected to
  be a near-no-op at our current fraglen=250/fragsd=25 settings. Not
  treating this as a problem to fix (shortening fraglen artificially just
  to force a visible adapter effect would be tuning the simulation to
  produce a result rather than modelling reality) -- documenting it as an
  expected, verified-correct outcome instead.

## Status

Implemented in-session per explicit user instruction ("tackle RNG first
then move to noise layer 5" -- continuing the session's established
implement-now pattern from layers 2-4).
