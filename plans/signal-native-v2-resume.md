# signal-native-v2 — START HERE (2026-08-16)

Resume document. Branch `signal-native-v2`, HEAD `b9e4986`, clean tree, nothing
pushed (per repo convention — pushing is the user's call).

**Read this before `plans/signal-native-v2-checkpoint.md`.** That file is a
1,300-line chronological log of Phases 1-3 and every noise layer; it is still
the reference for *how* things were built and why, but its framing up to the
final section assumes a premise that the last run disproved. Read it for
mechanism, not for direction.

---

## The one thing to know

**v2 transfers to real data far worse than v1, and v1 uses a much cruder
simulator.** Both are trained on simulated data only and evaluated zero-shot on
the same 140 curated benchmark events.

| | v1 | v2 (4,800 events) | v2 best (nn15) |
|---|---|---|---|
| collapsed accuracy | **87.9%** (123/140) | 62.1% | 69.3% |
| A3 / A5 / ES / IR | 75.0 / 90.0 / 90.8 / 94.1 | 46.4 / 50.0 / 73.8 / 64.7 | — |
| NN predictions (all false) | 2 | 32 | 25 |

v1's number was **reproduced locally on 2026-08-16** (not quoted from the
manuscript): `runs/v1/pred_*.csv`, produced with the released
`data/trained_nets/*` via `dsEnsemble.py --zoom -1 --a --cse`, in the dedicated
`DSv1` env. It matches 123/140 exactly.

Caveat that keeps the comparison fair: v1's nets are `--a --cse`, so v1 **cannot
separate A3 from A5 at all**. The table above is v2's collapsed number, so the
comparison is like-for-like, but v2 is attempting a strictly harder task and
losing on the easier metric.

**A correction to be aware of**: earlier sessions in this thread asserted v1 was
trained on real data. That was wrong — `runDS.smk` has
`SIMDIR="/data/deepSpecas/simulations/simulated_reads/"`. The false claim was
propping up a reading in which v2's number was a reasonable Phase-4 baseline. It
is not.

## The second thing to know

**More simulated data makes transfer worse.** Design held fixed at 25% NN:

| training set | sim-CV macro-F1 | benchmark (collapsed) |
|---|---|---|
| 960 (batch 1) | 0.667 | **67.9%** |
| 3,840 | 0.759 | 65.7% |
| 4,800 | **0.800** | 62.1% |

Isolated without the volume/mix confound by the step-4 conditions: at identical
class mix, 1,591 positives transfer *better* than 3,600 (64.3% vs 61.4%). Sim-CV
and transfer are anticorrelated across every comparison available. **Do not run
more simulation batches** — that lever is exhausted and points the wrong way.

---

## What is done and trustworthy

- **Phases 1-3** — extractor, simulation with five noise layers, feature builder,
  LightGBM baseline, `JunctionSetTransformer`. All verified; see the checkpoint.
- **Window-geometry leak** — fixed; controls at chance, re-confirmed at the new
  scale and the new class mix.
- **Scale-up run** (`plans/signal-native-v2-scaleup-7500.md`) — executed in full,
  all five batches passed all four gates. 4,800 events, 4,800 distinct genes,
  zero cross-batch overlap. STAR 96.4-97.7% uniquely mapped; PSI recovery per
  label 0.126-0.211.
- **Step 0 (CSE→A3 diagnosis)** — the plan's hypothesis (curated ES windows
  genuinely contain competing A3 events, so the label is the curator's choice)
  is **refuted**. SUPPA catalogues an A3 event in only 18% of ES windows and
  *fewer* in the misclassified subgroup; A3-consistent structure is
  near-ubiquitous but *weaker* where the model says A3; `p_A3` tracks A3
  evidence at r = −0.24. Real cause: insensitivity to thin skip evidence.
  Tool: `dsignal/diagnose_cse_a3.py`.
- **Step 4 (design vs calibration)** — **redundant, and calibration wins.**
  Positive-matched pair (identical 1,591 positives, NN 1,200 vs 530):
  rebalancing 43%→25% gains +2.9 points; calibrating the 43% model gains +5.7
  and lands higher (67.1% vs 65.7%). Calibration is free; rebalancing costs a
  re-simulation. **Future designs need not be rebalanced.** Ship the uniform
  prior correction — it is free and never hurts.

## The live hypothesis for the next session

v2's added realism is **miscalibrated, not insufficient**. Every noise layer was
validated only by showing it *degraded* the baseline model, which was read as
proof it added signal-relevant noise. Nobody ever checked whether the *amount*
matches the real data. Step 0 found the first hard counter-instance, in the
opposite direction: simulated CSE events carry **~4x** the skip-junction support
that curated ES events do (median minor-branch reads 132 vs 31), because
`sim/select_events.py::_sample_psi_pair` draws `|dPSI| ~ U(0.3, 0.8)` at a fixed
2,000-read gene depth and never produces a subtle event.

So the simulation is plausibly *noisier* along the axes we added and *cleaner*
along the ones we never varied — a larger distribution shift than v1's simpler
simulator had, not a smaller one.

> **Superseded by `plans/signal-native-v2-audit-geometry.md`** (approved
> 2026-08-16). That plan keeps the audit as its Track A, drops the
> representation crossover, and adds a Track B for a *second, larger* shift
> found while planning it: the curated benchmark's window **is** the event span
> (`benchmark/events.bed` == the region CSVs), while sim events fill a
> label-dependent 4-98% of a window ~2x too wide. The list below is kept for its
> reasoning; take the order from the newer plan.

## Proposed next steps, in order

1. **Noise-calibration audit** (the main line). For each quantity the simulation
   injects, measure it in `benchmark/c1.bam`/`c2.bam` and compare against what
   the simulation produces:
   - intronic / pre-mRNA coverage fraction (layer 2)
   - neighbouring & antisense gene expression (layer 3)
   - unannotated-junction rate and abundance (layer 4)
   - PCR duplicate rate (layer 5 — `samtools markdup` gives the real one)
   - coverage depth distribution, and `|dPSI|` of real differential events
   `dsignal/diagnose_cse_a3.py` demonstrates the method end to end (extract →
   canonicalize → measure motif/coverage statistics → compare sim vs real);
   reuse its `_supported` / `_best_shared_pair` / `_best_se_triplet` helpers.
   Expected output: a table of sim-vs-real per quantity, and a shortlist of the
   ones that are off by more than ~2x.
2. **Window randomization is a specific suspect.** v1 trains and tests on
   event-derived windows; v2 randomizes (`sim/window.py`). The randomization
   correctly removed a width shortcut, but it may also have moved training
   windows away from what a curator actually draws. Cheap test: re-extract the
   existing BAMs with event-derived windows and re-score transfer — no
   re-simulation needed, and there is precedent (the windowfix run did exactly
   this kind of re-extraction-only change).
3. **Consider the representation.** v1's images + ResNet50 may simply be more
   robust to this shift than a transformer keyed on exact junction identity and
   counts. Worth testing rather than assuming signal-native is strictly better.
4. **`|dPSI|`/depth realism run** — one-line change to `_sample_psi_pair` plus
   varying `--gene-mean-depth`. Now a *sub-item of the audit* rather than a
   standalone run: fix it together with whatever else the audit flags, in one
   batch, rather than one variable at a time.

Not recommended: more scale-up; more noise layers; Phase 4 fine-tuning (it would
paper over the transfer gap rather than explain it, and the gap is the finding).

---

## Environments and paths

**Local (has the GPU, RTX 4080; all training happens here)**
- `/datasets/deepSpecas-v2/envs/DSnet` — v2 work. torch 2.5.1 **conda** build
  with CUDA, lightgbm, sklearn, pysam.
  **Do not `pip install` into this env.** On 2026-08-16 a `pip install
  torchvision` pulled its own `torch 2.13.0+cu130` — CPU-only on this driver —
  overwrote the conda torch and broke lightgbm's libstdc++ resolution. Repaired
  with `mamba install --force-reinstall pytorch-gpu=2.5.1 pytorch=2.5.1
  libtorch=2.5.1`. Use a separate env for anything needing new packages.
- `/datasets/deepSpecas-v2/envs/DSv1` — **new**, for running v1. python 3.11,
  pytorch-gpu 2.5.1, torchvision 0.20.1, py-opencv, pysam, seaborn, rich.
  Created because v1 needs torchvision/cv2 and DSnet must not be touched.
- `/datasets/deepSpecas-v2/envs/DSpoly2` — older env, **no torch**; not useful.

**Pasteur (simulation/alignment only — no GPU)**
- `ssh pasteur`, workdir `/data/conda/ciccolella/deepSpecas-v2`
- envs at `/data/conda/ciccolella/envs/`: `DSpoly2v2` (python, STAR-avx2,
  samtools, SUPPA2), `DSsim` (R, ASimulatoR polyester fork, gffread)
- `sim/batches/b{1..5}/` — 42 GB of BAMs/FASTA/reads, kept for re-extraction.
  `/data` is at 83% (1.7 TB free). Delete `sim/batches/*/reads/` first if space
  is needed; the BAMs are the reusable part.
- See `memory/reference_pasteur_signal_native.md`.

## Key artifacts

| path | what |
|---|---|
| `sim/experiment/npz_scaleup/` + `labels_scaleup.tsv` | the 4,800-event training set |
| `labels_scaleup.{nn43,posmatched25,nn15}.tsv` | step-4 conditions (same npz dir) |
| `sim/experiment/batches/b{1..5}/` | per-batch npz, labels, design, psi_recovery |
| `runs/scaleup_allaug/fold*.pt` | the 4,800-event model (5 folds) |
| `runs/scaleup_nn15_allaug/fold*.pt` | v2's **best-transferring** model (69.3%) |
| `runs/v1/pred_*.csv` | reproduced v1 predictions, 123/140 |
| `runs/cse_a3_diagnosis.{tsv,log}` | step-0 diagnosis + sim-vs-real evidence table |
| `runs/benchmark_events_*.tsv` | per-event predictions with probabilities |

## Tooling added this run

- `sim/run_batch.sh` — the whole simulation recipe, resumable, one batch per
  invocation. Previously existed only as prose in the checkpoint.
- `sim/order_fasta.py` — reorders the FASTA to the count matrix **positionally**
  (`simulate_experiment_countmat` matches by row index, not name). The easiest
  step in the pipeline to get silently wrong.
- `sim/merge_batches.py` — re-keys batches as `<tag>_<ix>`; `--nn-frac` /
  `--from-labels` build class-mix conditions.
- `sim/run_measurements.sh` — controls → models → benchmark → prior sweep.
- `dsignal/diagnose_cse_a3.py` — annotation-free junction-motif diagnostics,
  sim-vs-real comparison, SUPPA catalogue overlap.

## Bugs fixed this run (do not re-introduce)

1. `run_batch.sh` resume check used `ls "$dir" 2>/dev/null | wc -l`; under
   `set -e -o pipefail` the non-zero exit silently killed the script after
   alignment on three batches. Now an explicit `-d` guard + `find`.
2. **Second ASimulatoR polyester fork bug** (the first was the `class(readmat)`
   check): `get_reads()` trims short fragments exceeding `readlen` after adapter
   padding, but when that selection is *empty* `subseq<-` calls `xscat()` with
   zero sequences and aborts. Seed-dependent — b1-b4 survived, b5 died after 25
   min. `sim/run_polyester.R` patches `polyester:::get_reads`, asserts the
   substitution matches exactly once, and was verified byte-identical when the
   selection is non-empty.
3. Watch out for `pgrep -f <pattern>` in a wrapper whose own command line
   contains the pattern — it matches itself and waits forever. Cost 4.3 hours
   this run.

## Conventions

- Never `git push`; never create/switch branches unprompted.
- Code changes go through a plan file under `plans/` + a separate implementation
  session, unless the user explicitly says to implement in-session.
- Build → verify → commit each step before starting the next (the user has
  confirmed this preference explicitly).
- Training is local (GPU); simulation/alignment is on pasteur. Only `.npz` +
  labels come back — never BAMs/FASTA.
