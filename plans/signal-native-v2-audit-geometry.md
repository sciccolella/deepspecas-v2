# signal-native-v2 — noise-calibration audit + window-geometry realism

Successor to `plans/signal-native-v2-resume.md`. Approved 2026-08-16; read the
resume doc first for the state this plan starts from.

## Context

v2 transfers to the 140-event curated benchmark far worse than v1 (62.1% vs
87.9% collapsed; v1 reproduced locally on 2026-08-16), despite v1 using a cruder
simulator. More simulated data makes transfer *worse* — sim-CV and transfer are
anticorrelated across every comparison available — so the scale-up lever is
exhausted and the remaining explanation is **distribution shift between the
simulation and the real data**, not insufficient training signal.

The resume doc's live hypothesis is that v2's added realism is *miscalibrated,
not insufficient*: every noise layer was validated only by showing it degraded
the baseline, never by checking whether the *amount* matches reality. Step 0
found the first hard counter-instance in the opposite direction — simulated CSE
events carry ~4x the skip-junction support that curated ES events do.

**A second, larger shift was found while planning this work** and is quantified
below: the extraction-window geometry. Both are pursued in parallel; they do not
contend for the same resource (the audit is CPU/local, the geometry work needs
pasteur re-extraction plus the local GPU).

### The geometry finding (new — from `sim/experiment/batches/b*/design.tsv` vs `benchmark/*.txt`)

| | sim (4,800) | curated benchmark (140) |
|---|---|---|
| window width, log10 mean / sd | 3.822 / 0.560 | 3.445 / 0.702 |
| median width | 6,110 bp | 3,063 bp |
| event span ÷ window width, median | **0.567** (RI 0.22, A3 0.42, A5 0.49, NN 0.71, CSE 0.94) | **≈ 1.0**, all classes |

`benchmark/events.bed` is identical to the region CSVs: **the curated window *is*
the event span**. Per-class benchmark widths (A3 1732, A5 1545, ES 6886, IR 944)
track the sim *event spans* (A3 1400, A5 1579, CSE 7332, RI 560) almost exactly.

So at inference the event always fills its window; in training it fills a
label-dependent 4–98%, and windows are ~2x too wide. `sim/window.py`'s
benchmark-calibrated width draw matched the *marginal* width distribution while
breaking the *relationship* between window and event that the benchmark has.

**Decision taken (user, this session): match it, and measure the coupling
honestly.** Matching re-introduces a width→label coupling that the
window-geometry fix deliberately destroyed — but that coupling is present in the
deployment distribution too, so the honest test is whether a geometry-only model
*also transfers*. If it does, the coupling is shared real structure, not a
simulation artifact, and that is itself a finding worth writing down.

Out of scope by decision: the v1-images/ResNet50 representation crossover
(resume step 3); more simulation batches; more noise layers; Phase-4 fine-tuning.

---

## Track A — noise-calibration audit (local, CPU; pasteur for the sim side)

### A1. `dsignal/audit_noise.py` — one measurement pass, two modes

`measure` mode takes a `region,bam1,bam2` CSV (the format both the benchmark
CSVs and every batch's `extract.csv` already use) and writes one row per window.
`compare` mode reads two such TSVs and prints a sim-vs-real table with a
ratio column and a shortlist of quantities off by more than ~2x.

Every quantity is **annotation-free and window-local**, so the same code runs
unchanged on real and simulated BAMs. Reuse, don't reimplement:
`dsignal.extract.extract_condition` / `extract_event`,
`dsignal.features.normalize_record`, and `diagnose_cse_a3._supported` /
`_best_shared_pair` / `_best_se_triplet` (import them; that script is the
worked example of this whole method end to end).

| quantity | how | sim knob it audits |
|---|---|---|
| `intron_cov_frac` | median coverage strictly inside each supported junction's intron ÷ mean coverage over the flanking exonic anchors; report the median over supported junctions | layer 2: `build_experiment --premrna-frac-low/high` 0.01 / 0.10 |
| `novel_junction_rate`, `novel_junction_read_frac` | share of supported junctions with raw `jM < 20`, and their share of junction reads | layer 4: `--novel-frac-low/high` 0.005 / 0.03, `--novel-nn-frac-high` 0.20 |
| `dup_rate` | fraction of read pairs sharing `(reference_start, cigarstring, next_reference_start)`, counted **per source library** | layer 5: `PCR_RATE=0.1`, `PCR_LAMBDA=1` |
| `background_frac`, `antisense_frac` | share of window coverage on bases spanned by no supported junction and outside its exonic anchors; NH-weighted share of junction reads whose motif strand disagrees with the window's `strand_vote` | layer 3: `--neighbor-depth-low/high` 100 / 3000 |
| `cov_mean`, `cov_median`, `top_junction_count`, `n_supported` | already in `ConditionRecord.stats` / `normalize_record` | `--gene-mean-depth 2000` |
| `a3/a5/se_delta`, `*_minor_count` | the three `diagnose_cse_a3` helpers, unmodified | `select_events._sample_psi_pair`: `|dPSI| ~ U(0.3, 0.8)` |

Two things the implementation must get right, both verified during planning:

1. **`extract.py` masks the annotated-junction bit off** (`motif_raw % 20`), so
   the novel-junction rate is *not* recoverable from the stored `.npz`. The
   audit tool must walk the BAM itself. Both sides carry the bit: the benchmark
   BAM's reads show `jM=[0,22]` (22 ≥ 20 = annotated), and the sim BAMs were
   aligned against a STAR index built from `Homo_sapiens.GRCh38.99.gtf`.
2. **`benchmark/c1.bam` is `samtools view -L events.bed` of a 3-run merge.**
   Off-window reads are absent, so `background_frac` / `antisense_frac` are a
   *lower bound* on the real side and must be printed with that caveat, not
   silently compared. The merge also means duplicate detection must be grouped
   by library: query names carry the run prefix (`SRR1513329.` / `330.` /
   `331.`), so key duplicates within a prefix only, or cross-run coordinate
   collisions inflate the real dup rate.

*Model tier: strong.* The metric definitions and the two caveats above are the
whole value of the step; a mechanical implementation would produce a table that
looks fine and means nothing.

### A2. Run it and compare

- Real, locally: over `benchmark/{A3,A5,ES,IR}.csv` → `runs/noise_audit_real.tsv`.
- Sim, on pasteur (needs the BAMs): over `sim/batches/b*/extract.csv`, optionally
  subsampled to ~1,000 rows for speed → `runs/noise_audit_sim.tsv`. **Only the
  TSV comes back** — never BAMs or FASTA.
- Locally: `compare` mode → `runs/noise_audit_compare.log`.

Deliverable: the sim-vs-real table plus the >2x shortlist. Expected to confirm
the known `|dPSI|` / minor-branch-depth offender (median 132 vs 31 reads) and to
say for the first time whether layers 2–5 are over- or under-injected.

*Model tier: cheap* (running commands, moving one TSV).

---

## Track B — window-geometry realism (pasteur re-extraction + local GPU)

### B1. Decisive cheap diagnostic first: does geometry-only *transfer*?

`runs/scaleup_geometry.json` records the geometry-only control at 0.360 sim-CV
accuracy (vs a 0.25 NN-majority floor) — but that model was never saved, so it
was never scored on the benchmark. `dsignal.predict.Predictor` restores
`feature_subset` from the checkpoint, so `score_benchmark` needs no change:

```
python -m dsignal.train --npz-dir sim/experiment/npz_scaleup \
  --labels sim/experiment/labels_scaleup.tsv --feature-subset geometry \
  --out-dir runs/scaleup_geometry_ctl --json-out runs/scaleup_geometry_ctl.json
python -m dsignal.score_benchmark --net runs/scaleup_geometry_ctl/fold*.pt \
  --out runs/benchmark_events_geometry_ctl.tsv
```

~15 min of GPU. Read it as: geometry-only well above chance *on the benchmark*
⇒ window geometry is shared real structure and B3 is justified; at chance ⇒ the
geometry shift costs the model nothing directly and B3's value drops to
"removes a train/test mismatch", which is weaker but still worth testing.

*Model tier: cheap.* Two commands and a number.

### B2. Measure the real flank distribution from the reads

The benchmark window equals its event span, so there is no padding to read off
the CSVs — measure it from the alignments instead. Per benchmark window: find
the outermost supported junction coordinates and record the left/right gap to
the window edges. That is the empirical "padding a curator's window leaves
around the splicing structure", and it is the distribution B3 fits. Run the same
measurement on the sim windows for the side-by-side. Fold this in as a
subcommand of `audit_noise.py` rather than a new script.

*Model tier: medium.*

### B3. Rewrite `sim/window.py` as an event-derived sampler

`win = [event_start − L, event_end + R]`, with `L`, `R` drawn independently from
the B2 flank distribution in **absolute bp**, label-independent. Keep the
`sample_window_detail` signature and the chromosome-clipping behaviour;
`clamped` loses its meaning and should be repurposed or removed deliberately,
not left lying.

Do not delete the module docstring's history — it records why the multiple-of-span
sampler and then the benchmark-width sampler were each rejected, and this change
is the third entry in that sequence, not a reversal to the first. Same for
`sim/width_leak_check.py`: its recorded numbers (0.457/0.235 → 0.432/0.199)
must be *appended to*, with the new sampler's number, not overwritten.

*Model tier: strong.* This is the design decision the whole track rests on.

### B4. Re-extract b1–b5 on pasteur — no re-simulation

Re-run `sim.select_events` per batch at the same seed and the same
`--exclude-genes` priors. `select_events.main`'s RNG note guarantees this: `rng`
drives row selection and is never touched by window sampling (`rng_window` is
seeded independently), so the regenerated design must reproduce the identical
rows and genes with only new window columns.

**Gate:** diff the regenerated `design.tsv` against the stored one on every
column except `win_start`/`win_end` — must be byte-identical. If it is not, stop:
something else moved and the comparison to the 4,800-event run is void.

Then `sim.build_extract_csv` + `dsignal.extract` against the existing
`sim/batches/b*/aligned/c{1,2}.bam`. Gate: npz count == `extract.csv` rows per
batch (the bug that silently truncated three batches last run). Copy back the
`npz` dirs (53 MB total for 4,800) + labels. Nothing else.

*Model tier: medium* — mechanical, but the design-diff gate needs care.

### B5. Merge, train, score

```
python sim/merge_batches.py ...            # -> npz_geomwin / labels_geomwin.tsv
sim/run_measurements.sh sim/experiment/npz_geomwin \
  sim/experiment/labels_geomwin.tsv geomwin \
  sim/experiment/labels_geomwin.nn15.tsv
```

`run_measurements.sh` already runs, in order: the two leak controls, the GBM,
the transformer without augmentation, the shipped all-augmentation model, the
benchmark transfer, and the prior sweep — for the base labels and each extra
mix. ~20 min per training config on the RTX 4080.

The numbers to beat, both collapsed on the benchmark: **62.1%** (scaleup, 25% NN)
and **69.3%** (scaleup nn15, v2's best). Train the nn15 mix too so the
comparison is best-vs-best. Report the prior-corrected number alongside the raw
one — the uniform-prior correction is free and settled.

*Model tier: medium.*

---

## Verification

Track A: the audit is its own verification — the deliverable is the sim-vs-real
table. The one thing to self-check is that `measure` mode gives identical
`cov_mean` / `n_supported` to the stored `.npz` `stats` for a handful of sim
windows; if it does not, the BAM walk disagrees with `dsignal.extract` and every
other number is suspect.

Track B, in order, each a stop-the-line gate:
1. regenerated `design.tsv` identical to the stored one outside the window columns;
2. per-batch npz count == extract.csv row count;
3. `width_leak_check` + both geometry controls reported with their new values,
   **including the case where they go up** — that is the expected, accepted
   consequence of matching the benchmark, and suppressing it would defeat the
   point of B1;
4. benchmark collapsed accuracy vs 62.1% / 69.3%, plus the per-class A3/A5/ES/IR
   breakdown and the NN false-positive count (32 at 4,800; 25 at nn15).

## Notes carried forward

- Never `git push`; never create or switch branches unprompted.
- Build → verify → commit each step before starting the next.
- Training is local (GPU); simulation/alignment/extraction is on pasteur, and
  only `.npz` + labels + TSVs come back.
- Do not `pip install` into `/datasets/deepSpecas-v2/envs/DSnet` — it broke the
  env once already.
- pasteur `/data` is at 83%; the re-extraction adds ~11 MB per batch, negligible.
- Beware `pgrep -f <pattern>` in a wrapper whose own command line contains the
  pattern — it matches itself and waits forever (cost 4.3 h last run).
