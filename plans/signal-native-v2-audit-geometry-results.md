# signal-native-v2 — audit + geometry: results (2026-08-16)

Implementation report for `plans/signal-native-v2-audit-geometry.md`. Both
tracks executed in full, in one session. Branch `signal-native-v2`, nothing
pushed.

**Read this instead of `plans/signal-native-v2-resume.md` for direction.** The
resume doc's state description is still accurate; its diagnosis is now partly
superseded.

---

## The headline

**Zero-shot transfer to the curated benchmark went from 62.1% to 78.6%
collapsed, and the only thing that changed was the extraction window.** Same
simulated reads, same alignments, same 4,800 events, same 4,800 genes, same
model, same hyperparameters. No re-simulation.

| benchmark, collapsed | v2 windows | v3 windows |
|---|---|---|
| uncorrected | 62.1% | **78.6%** |
| uniform prior correction | 67.1% | **79.3%** |
| best over class mixes | 69.3% (nn15) | **79.3%** (25% NN) |
| NN false positives (of 140) | 32 | 13 |
| sim-CV macro-F1 | 0.800 | 0.867 |
| A3 / A5 / ES / IR, collapsed | 46 / 50 / 74 / 65 | 82 / 70 / 80 / 82 |

v1 remains ahead at 87.9%. The gap is now 8.6 points, was 25.8.

Two secondary observations:

- **sim-CV and transfer moved together for the first time.** Every earlier
  comparison had them anticorrelated, which is why the resume doc said not to
  trust sim-CV. That anticorrelation may have been an artifact of the window
  mismatch rather than a general property.
- The prior sweep's oracle (`nn_weight=0`, i.e. told the benchmark has no NN
  events) reaches **87.1%** against v1's 87.9%. **Not a like-for-like claim** —
  v1 needs no such oracle, having emitted only 2 false NN calls. What it does
  say: the residual gap to v1 is now mostly NN false positives, not confusion
  among the positive classes.

## Why the window was wrong

`benchmark/events.bed` is byte-identical to the region CSVs, and matching all
140 curated windows against the SUPPA catalogue the simulation draws its events
from (`python -m dsignal.audit_noise padding`) shows the curated window **is**
the event span plus a per-type constant:

| csv | matched | pad_left med [q10,q90] | pad_right med [q10,q90] |
|---|---|---|---|
| A3 | 28/28 | −1 [−89, −1] | −1 [−1, −1] |
| A5 | 29/30 | −1 [−1, −1] | −1 [−1, −1] |
| ES | 55/65 | 100 [100, 100] | 100 [100, 100] |
| IR | 13/17 | 0 [0, 0] | 0 [0, 0] |

Median event span / window width: **1.000**. 120/125 matched windows agree on
both edges within 200 bp. (The unmatched ones are rMATS-only calls absent from
SUPPA's strict catalogue.)

The v2 sampler matched the benchmark's *marginal width distribution* while
destroying this *relationship*: the simulated event filled a label-dependent
4–98% of its window (median 0.567 — RI 0.22, A3 0.42, A5 0.49, NN 0.71, CSE
0.94) inside a window ~2× too wide (log10 mean 3.82 vs 3.44). v3 draws
`win = event span + U[0,100] bp per edge`, label-independently: log10 mean
3.34–3.40 per batch, span/width 0.95.

The per-type constants are deliberately **not** reproduced — a per-class padding
constant would be label leakage with no counterpart at inference, where nobody
knows the class.

### The cost, reported not suppressed

v3 knowingly reintroduces a width→label coupling. Like-for-like at 4,800 events
and 25% NN, sim-CV:

| control | v2 windows | v3 windows | floor |
|---|---|---|---|
| width-only decision tree | 0.279 / 0.221 | 0.349 / 0.293 | 0.250 / 0.080 |
| LightGBM geometry-only | 0.319 | 0.446 | — |
| Transformer geometry-only | 0.360 | 0.486 | — |

This is accepted rather than explained away, on evidence gathered before the
change: the geometry-only control, trained on the 4,800-event set and scored on
the **curated benchmark** for the first time, reaches 31.4% 5-class / 35.0%
collapsed — above a ~20% random floor, below the 46.4% you get by always
answering CSE. So geometry is not a strong general signal, but it *is* a strong
RI signal: **13/17 curated IR events are recovered from window geometry alone,
on real data**, where RI is 12% of the set. The coupling is therefore partly
shared with the deployment distribution, not purely a simulation artifact.

And the model is not merely riding it: it gains 16.5 points of transfer where
geometry alone gains far less, and the shipped model trains with
`--feature-subset no-geometry`, which zeroes those columns outright.

## Track A — the noise-calibration audit

`dsignal/audit_noise.py`, 1,500 simulated windows (300 per batch) vs the 140
curated ones. **Every injected quantity is over-injected except layer 4 — and
the differential signal the model is supposed to learn is under-injected.**

| quantity | sim/real | knob |
|---|---|---|
| `intron_cov_frac` | **7.43× over** | layer 2, `--premrna-frac 0.01–0.10` |
| `se_minor_count` | **6.75× over** | `_sample_psi_pair` |
| `antisense_frac` | 6.36× over † | layer 3, `--neighbor-depth 100–3000` |
| `a3_minor_count` | 5.28× over | `_sample_psi_pair` |
| `cov_mean` | 5.20× over | `--gene-mean-depth 2000` |
| `top_junction_count` | 4.46× over | depth |
| `dup_rate` | 2.77× over ‡ | layer 5, `PCR_RATE=0.1` |
| `background_frac` | 2.43× over † | layer 3 |
| `novel_junction_rate` | **0.52× under** | layer 4, `--novel-frac 0.005–0.03` |
| `a3_delta` | **0.56× under** | `_sample_psi_pair` |
| `se_delta` | **0.55× under** | `_sample_psi_pair` |

† real side is a lower bound: `benchmark/c1.bam` is a `samtools view -L
events.bed` subset, so genuinely off-window background is absent. Direction
only. ‡ a coordinate-duplicate rate rises with depth on its own; read next to
`cov_mean`, which is itself 5.2× off.

### The finding that inverts the resume doc

The resume doc's concern was that `|dPSI| ~ U(0.3, 0.8)` at fixed depth "never
produces a subtle event". Measured **on the reads**, the realised per-condition
change is 0.12–0.15 in simulation against 0.22–0.27 in the curated benchmark.

Simulated events are **deep but bland**: ~5× the read support carrying half the
differential change. Curated events are **shallow but sharp**: few reads, large
PSI change. A model trained to read a large difference off a large pile of reads
meets a small pile with a sharper difference at inference. That is one coherent
mechanism for both the transfer gap and for more simulated data making it worse.

The requested `|dPSI|` is not the problem — the *recovered* one is. This is
consistent with the batches' own PSI-recovery numbers (0.126–0.211).

### Layer 2 inverts per class

Pooled, layer 2 over-injects 7.4×. For RI specifically it **under**-injects
4.2×: real IR windows have intron/exon coverage 0.558, simulated RI 0.133.
Simulated retained introns are not retained nearly as thoroughly as real ones.

## What was verified

Track A: the tool's own BAM walk reproduces `dsignal.extract`'s junction table
exactly — same keys, same counts — on benchmark windows. It needs a separate
pass only because `extract.py` masks off STAR's `+20` annotated-junction bit,
which is the whole of layer 4's readout.

Track B, every gate passed:
1. all five regenerated designs byte-identical to the scale-up outside
   `win_start`/`win_end` (seeds 1–5, verified empirically rather than assumed);
2. npz count == extract.csv rows, 960 per batch, 4,800 total;
3. leak controls re-run and reported at their new, higher values;
4. transfer measured for both class mixes, raw and prior-corrected.

## Two bugs found in this session's own work

1. **The regen script built `--exclude-genes` flags and never expanded them
   into the command.** Batches b2–b5 came out with disjointness broken. The
   row-identity gate caught it, which is what it was for.
2. **`_library_of` split read names on `.` to scope duplicate detection.** That
   is right for `benchmark/c1.bam` (three merged SRA runs, `SRR1513329.6348221`)
   and wrong for simulated reads, named `read2270765` with no dot — every read
   became its own library and the measured duplicate rate was ~0. The first
   audit therefore reported layer 5 as doing nothing; corrected, it over-injects
   2.77×. Any conclusion drawn from a "sim dup_rate ≈ 0" reading is void.

Also: a `touch DONE` sentinel placed after a loop that could fail fired on a
failed run, and a wait loop trusted it. Sentinels must be written only on
success.

## Where the class mix lever went

Under v2 windows, rebalancing to 15% NN helped (62.1% → 69.3%). Under v3
windows it **hurts** slightly: 78.6% at 25% NN vs 77.1% at nn15. The lever
reversed. Combined with the earlier step-4 finding that calibration beats
rebalancing, the position is now: leave the mix at 25% NN and apply the uniform
prior correction, which is free.

## Suggested next steps

1. **A calibration re-simulation batch**, now justified by numbers rather than
   suspicion. One batch, fixing the audit's shortlist together: drop
   `--gene-mean-depth` ~5×, widen it into a distribution rather than a constant,
   raise layer 4's novel-junction rate ~2×, cut layer 2's pre-mRNA fraction
   except for RI rows, cut layer 3 and layer 5 by ~2.5×. Do not fix one
   variable at a time — the audit gives the whole list at once.
2. **Understand the `delta` shortfall** before or alongside (1). The realised
   differential change is half the real one despite a requested `|dPSI|` of
   0.3–0.8; the abundance model or the PSI recovery is losing it. This is the
   single largest lever left and it is not a knob, it is a defect.
3. Re-run the audit after (1) — it is the acceptance test now, cheap and
   annotation-free.
4. The representation crossover (v1 images on v2 data) is still open and still
   out of scope; the case for it is weaker now that the signal-native model is
   within 8.6 points.

## Artifacts

| path | what |
|---|---|
| `dsignal/audit_noise.py` | the audit tool (`measure` / `compare` / `padding`) |
| `runs/noise_audit_{real,sim_b*}.tsv`, `runs/noise_audit_compare.log` | Track A data + table |
| `runs/window_padding.log` | curated window vs SUPPA event span |
| `sim/window.py` | v3 sampler; docstring carries all three iterations |
| `sim/experiment/npz_geomwin/`, `labels_geomwin{,.nn15}.tsv` | 4,800 re-extracted records |
| `sim/experiment/batches_geomwin/b*/` | per-batch npz, labels, regenerated design |
| `runs/geomwin_*.{json,log}` | controls + models under v3 windows |
| `runs/benchmark_geomwin{,_nn15}.log`, `runs/prior_sweep_geomwin*.tsv` | transfer |
| `runs/benchmark_events_geometry_ctl.tsv` | geometry-only control on the benchmark |
| pasteur `sim/batches/b*/{design.geomwin.tsv,npz_geomwin/}` | remote side |
| pasteur `sim/{regen_designs.sh,reextract_geomwin.sh}` | the two re-run scripts |
