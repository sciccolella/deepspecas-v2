# signal-native-v2 — window-geometry leak fix (Phase 2 follow-up)

Fixes the item flagged at the end of `plans/signal-native-v2-checkpoint.md`
("Fix the window-geometry leak in `sim/window.py`"). Branch:
`signal-native-v2`.

## Diagnosis (done before this plan, numbers reproducible)

The checkpoint says window geometry alone reaches 46.3% acc / 0.415 macro-F1
and calls it a window-sampler artifact. Measured more precisely:

1. **The accuracy is just window width.** A `DecisionTreeClassifier(max_depth=4)`
   on window width *alone*, 5-fold over `npz_scale1260` (widths read straight
   off the `.npz` filenames): **0.463 acc / 0.253 macro-F1** — the accuracy is
   identical to the Transformer geometry control's 0.463. Per-class F1:
   NN 0.656, RI 0.440, A3 0.141, A5 0.030, CSE 0.000. So the shortcut is
   carried by NN and RI; the positive classes get nothing from width.

2. **The control's extra macro-F1 is not geometry.**
   `dsignal/dataset.apply_feature_subset("geometry")` zeroes feature *columns*
   but leaves the *number of junction tokens* and their relative spacing
   intact. Junction count is structural information, not geometry, so the
   control as written is not the control the plan's Verification item 2 asks
   for. That, not width, is the gap between 0.253 (true width-only) and 0.415
   (reported control).

3. **Width↔label correlation is partly real biology and does transfer.** The
   curated benchmark shows the same ordering as the simulation:

   | | RI / IR | A5 | A3 | CSE / ES |
   |---|---|---|---|---|
   | benchmark region width, median | 943 | 1,544 | 1,731 | 6,885 |
   | sim window width, median (scale1260) | 1,451 | 2,727 | 2,563 | 17,763 |

   Pooled benchmark widths (n=140, all four CSVs): min 1, q10 336, q25 1,208,
   median 3,062, q75 7,544, q90 19,122, max 181,765; `log10` mean 3.442,
   sd 0.713.

   The genuine artifacts are therefore narrower than "windows correlate with
   label":
   - **NN windows come from whole gene spans** — median 22,734 bp, q90
     247,638 bp. No curator produces that, and NN is exactly where the
     shortcut's accuracy lives.
   - **Sim windows are ~2x wider than benchmark ones** across the board,
     because flank is drawn as a multiple of the event span.

## Steps

### Step 1 — recalibrate `sim/window.py` (span-independent width)
*Model tier: strong (distributional judgment, clamp semantics, RNG care).*

Replace the "flank = log-uniform [0.1x, 10x] of event span" rule with:

- draw a target window **width** from a log-normal calibrated on the pooled
  benchmark distribution above (`log10` mean 3.442, sd 0.713), truncated to
  the observed `[min, max]` range — pooled across all four CSVs so the draw
  carries no label information;
- if the drawn width is smaller than the event span, clamp to the span (plus
  a small minimum flank) — unavoidable, since the window must contain the
  event. **Count and report the clamp rate**; it is the residual, irreducible
  span→width coupling and belongs in the checkpoint;
- place the event uniformly at random inside the window (keeps the existing
  off-center property, which was working and should be preserved), clipped to
  `[1, chrom_len]`.

Keep the signature and the 1-based-inclusive return convention, and keep
`rng` as the sole randomness source. **RNG care**: `sim/select_events.py`
seeds `rng_window` as a *separately seeded* generator (`seed + 1_000_000`),
deliberately not spawned from a shared `SeedSequence` — see the window
randomization update in the checkpoint. Do not change that. Changing the
number of draws per row inside `sample_window` is fine (it only affects the
window stream), but verify the row-*selection* columns of the regenerated
`design.tsv` are byte-identical to the current ones before going further —
same check that caught this class of bug last time.

### Step 2 — stop NN windows inheriting whole-gene spans
*Model tier: cheap/mechanical.*

In `sim/select_events.py`, for `kind in {nn_plain, nn_novel}` (and check
`nn_hard` — it uses a real event span, so it is probably already fine), the
row's `region_start`/`region_end` currently span the entire gene. Replace it
with a random sub-interval of the gene whose length is drawn from the pooled
*positive* event-span distribution, so NN stops being "the wide class". The
gene's own coordinates still bound it.

Note: `region_start`/`region_end` for NN rows feed nothing but window
sampling and the extraction CSV (NN rows have no forced event and are not in
`psi_recovery_check.py`), so this is a contained change — confirm that before
implementing.

### Step 3 — make the geometry control honest
*Model tier: strong (must not perturb the real training path).*

`dsignal/dataset.apply_feature_subset("geometry")` must also neutralize
junction count, so the control tests geometry and nothing else. Simplest
approach: under `subset == "geometry"`, replace the token set with a fixed
number of identical dummy tokens (count independent of the record) carrying
only the geometry columns, leaving the CLS geometry features intact. The
`"all"` and `"no-geometry"` paths must be untouched — verify by re-running a
`no-geometry` fold and confirming the metrics match the checkpoint's table
exactly.

### Step 4 — re-extract and re-measure
*Model tier: cheap for the pipeline driving, strong for reading the results.*

**Scope decided with the user: re-extract only, no full re-simulation.** Keep
the existing `scale1260` BAMs on pasteur; regenerate `design.tsv` (new
windows) → `sim/build_extract_csv.py` → `dsignal/extract.py` → pull the new
`.npz` set back locally as `sim/experiment/npz_windowfix/` +
`labels_windowfix.tsv`. Minutes, not hours.

**Known and accepted mismatch**: the neighbour background gene set on those
BAMs was chosen under the *old* windows, so some new windows will contain
genes that were never simulated as background (less layer-3 noise than
intended) and vice versa. That is fine for measuring whether the geometry
shortcut is gone; a fully consistent re-simulation happens at the next
scale-up, which is a re-run of the existing recipe with no new code.

Then, locally on the `DSnet` env (`/datasets/deepSpecas-v2/envs/DSnet`):

1. width-only decision-tree check (the diagnosis-1 script) — should now be
   near chance (0.20 acc-equivalent on a 5-class, prior-weighted design);
2. `dsignal/baseline_gbm.py --feature-subset geometry` and `train.py
   --feature-subset geometry` — the actual Verification item 2 control, which
   should now sit at chance with Step 3 in place;
3. `baseline_gbm.py` + `train.py` with `no-geometry` (the standing reported
   numbers) — expect them roughly unchanged; a large move means the window
   change did something unintended;
4. `--feature-subset all` — **re-open the question the checkpoint closed.**
   Geometry was excluded by default because it was an artifact. If width is
   now benchmark-calibrated, `all` may be legitimately usable; if `all` still
   beats `no-geometry` by more than noise, keep `no-geometry` as the default
   and say why;
5. `dsignal/score_benchmark.py` — the 50.0% zero-shot number is the thing
   this fix might actually move, since the 30/140 false-NN predictions are
   the dominant real error and NN's absurd window widths were the biggest
   artifact. Report per-CSV and the confusion matrix as before.

### Step 5 — write up
*Model tier: cheap.*

Append an update section to `plans/signal-native-v2-checkpoint.md` covering:
the three diagnosis findings, the clamp rate from Step 1, the new control
numbers, the `all` vs `no-geometry` decision, the new benchmark score vs
50.0%, and the accepted neighbour-set mismatch from the re-extract-only
scope. Commit (no push).

## Verification checklist

- [ ] regenerated `design.tsv` selection columns byte-identical to current
- [ ] window width distribution per label vs the benchmark table above
- [ ] NN window widths no longer the widest class by an order of magnitude
- [ ] clamp rate reported
- [ ] width-only tree at chance
- [ ] geometry-only control at chance (with the Step-3 token-count fix)
- [ ] `no-geometry` numbers reproduce the checkpoint's table within noise
- [ ] benchmark score reported against the 50.0% baseline
