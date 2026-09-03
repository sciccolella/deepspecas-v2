# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`deepSpecas` classifies alternative splicing (AS) events from RNA-seq BAM files. There are **two lines of work in this repository**, and which one you are touching determines almost everything else:

- **v1 — image-based** (`master`, published). Pileup/coverage data from paired BAM files is rendered into one of six image representations, and a CNN (ResNet50) classifies the image into a splicing event type (`A3`, `A5`, `CE`/`SE` or collapsed `CSE`, `RI`, `NN`, optionally collapsed `A3`+`A5` → `A`). Root-level modules: `deepSpecas.py`, `dsEnsemble.py`, `plot2s.py`, `plot.py`, `DSutils.py`, `mc_mplp.py`, `jitimg.py`.
- **v2 — signal-native** (branch `signal-native-v2`, in development, unpublished). **No image rendering at all**: junctions and base-resolution coverage are extracted straight from the BAMs and classified by a junction-set Transformer. Training data is simulated end-to-end (SUPPA event catalogue → PSI-targeted abundances → polyester reads → STAR). Lives entirely in `dsignal/` and `sim/`.

**Branch `signal-native-v2` is the active line of work.** It was branched from `master` at `379cbba`, so it contains all of v1 too. If you are resuming v2 work, read **`plans/signal-native-v2-resume-9.md` first** — it is the current state-of-play document; `plans/README.md` indexes the rest.

## Environment setup

No requirements.txt/pyproject — dependencies are installed via conda/mamba.

**v1** (the published env recipe; note it is not what current work runs in):

```bash
mamba create -n DSpoly2 -c bioconda -c conda-forge -c pytorch -c nvidia pysam snakemake matplotlib rich pytorch torchvision torchaudio pytorch-cuda=11.8 scikit-learn pandas seaborn opencv
conda activate DSpoly2
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

**v2 compute split.** Training is local (RTX 4080); simulation, alignment and extraction run on the remote machine `pasteur`, which has no GPU.

| where | path | envs | notes |
|---|---|---|---|
| local | `/datasets/deepSpecas-v2/envs/DSnet` | `DSnet` | all v2 training. **Never `pip install` into it.** |
| local | `/datasets/deepSpecas-v2/envs/DSv1` | `DSv1` | running v1 (has torchvision + cv2) |
| remote | `ssh pasteur`, `/data/conda/ciccolella/deepSpecas-v2` | `DSpoly2v2`, `DSsim`, `DSrender` | simulation / polyester / STAR / extraction; `DSrender` has matplotlib+PIL+torch for `plot2s.py` |

Only `.npz` records, labels and TSVs come back from pasteur — never BAMs or FASTA. Pasteur's `dsignal/` holds only the modules shipped there; rsync a new one before running it.

### Running jobs on pasteur — read before launching anything parallel

**pasteur is a shared machine.** Other people's jobs run on it, and a careless
launch takes it away from them. On 2026-08-17 an unthrottled 32-way render drove
the load average to **230 on 64 cores** and made new login shells impossible for
~40 minutes. `crash_pasteur.md` has the exact reproduction; the rules that follow
come from it.

1. **Always wrap remote jobs in `limitram`.** `limitram <LIMIT> <command> …` is
   `/usr/bin/limitram`, a `systemd-run --user --scope -p MemoryMax=` wrapper:
   ```bash
   nohup limitram 32G bash sim/render2.sh squishstack 12 50000 > log 2>&1 &
   ```
2. **Cap the thread pools.** `limitram` bounds memory, not CPU, and memory was
   never the binding constraint in that incident — runnable threads were. numpy
   and torch each size a thread pool to the machine's core count, so N processes
   silently become N×64 threads. Export all four, every time:
   ```bash
   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
   ```
   Even then `plot2s.py` runs at ~174% CPU per process, so budget ~2 cores each.
3. **`nice -n 19`** every background job, and keep the shard count modest
   (≤12 for renders). Check `cat /proc/loadavg` before launching and a minute
   after; if the 1-minute figure goes past ~50, throttle.
4. **Measure throughput early** — sample the output twice a few minutes apart.
   Both render failures showed up as *zero* rows produced while the CPU was busy.
5. **Never `pkill -f <pattern>`** when the pattern also appears in the ssh
   command line carrying it — it kills your own session. Use `pkill -f "[p]attern"`.

There is no pytest suite, linter, or CI config, and no build step — scripts are run directly with `python`. The verification gates that stand in for tests are:

- `python -m dsignal.verify` — extractor coverage/junction parity against `plot2s.get_coords_from_bam`
- `python -m dsignal.verify_features` — canonicalization involution, exact A3↔A5 flip, coverage off-by-one
- `python -m sim.width_leak_check`, `python -m sim.psi_recovery_check` — simulation-design gates
- `python -m dsignal.audit_noise compare` — **the acceptance test after any re-simulation**: annotation-free, runs the identical code path on real and simulated BAMs

## v1 pipeline / data flow

1. **`plot2s.py`** — turns a genomic region + two BAM files (two experimental conditions) into an image representation. Defines the `Event2sample` / `Event2sampleReads` / `Event2sampleComb` classes (each implementing `EventAbstract`: `build_zoom`, `strip`, `plot`), one per image-representation "mode": `pyplot2s`, `colmplp2s`, `squishstack`, `squish4d`, `comb`, `comb4d`. Also runnable standalone as a CLI to batch-render a CSV of events into an image directory (used by the `make_plots` Snakemake rule).
2. **`plot.py`** — an earlier/parallel single-sample (not two-condition) plotting module with its own `Event`/`Event2sample`/`Event2sampleReads` classes and `pyplot_plt`/`colmplp_plt` renderers. `jitimg.py` is a small ad-hoc/debug driver built on `plot.py` + `DSutils.py` (not part of the Snakemake pipeline).
3. **`DSutils.py`** — GTF parsing (`parse_gtf`, `parse_multi_gtf`) and exon/transcript-mask building (`get_transcript`), used by the single-sample path.
4. **`mc_mplp.py`** — `Channels`/`Channels2d`/`ChannelsMat2d` helper classes for turning per-base coverage/skip arrays into stacked image-channel matrices; used internally by the plotting modules.
5. **`deepSpecas.py`** — the main model CLI (`EventDataset`, `_build_model` (ResNet50, first conv layer patched to accept 3 or 4 channels depending on plot mode), `_build_transformer`, `train_model`, `kfold_crossvalidation`, `validate`, `predict`). Three subcommands:
   - `train` — trains a net on a CSV of (image path, label), writes weights (`-o/--net-out`) and a confusion-matrix heatmap (`--hm`).
   - `validate` — loads a trained net, runs it against a validation CSV, reports precision/recall/fscore and a heatmap.
   - `predict` — given a CSV of `region,BAM_path1,BAM_path2`, renders images on the fly (via `plot2s.py`) and classifies them (no train/val CSV of pre-rendered images needed).
6. **`dsEnsemble.py`** — loads up to 6 trained nets (one per image representation, each passed as a separate `--<mode>` flag pointing at a `.pth` file), reuses `deepSpecas._build_model`/`_build_transformer`, renders each representation on the fly per event, combines their outputs and prints the ensemble prediction. This is the recommended way to run inference with the released pretrained models (see README). It also carries `--shard i/N` (event-level parallelism; every CSV line is independent and reassemblable by its `ix` prefix) and `--gpu-render` (opt-in device-side render path, **not** bit-exact with the default PIL path).
   - **How the ensemble actually combines:** it sums the six nets' **raw logits** and softmaxes once (`allout = sum(outs)`; `probs = F.softmax(allout, dim=1)`), *not* their softmax outputs. A net with larger logit magnitude therefore dominates regardless of calibration. This is a known open issue, recorded in `report.md`.

Six image "plot modes" are used consistently as an enum-like string across the whole codebase: `pyplot2s`, `colmplp2s`, `squishstack`, `squish4d`, `comb`, `comb4d`. 3-channel modes (`pyplot2s`, `squishstack`, `comb`) load as RGB; 4-channel modes (`colmplp2s`, `squish4d`, `comb4d`) load as CMYK. Each mode has a fixed `RESIZE_SIZE` (duplicated as a dict in both `DSutils`-adjacent `deepSpecas.py` and `plot2s.py`) that images are resized to before being fed to the net — scaled by `--resizescale` when given.

### Full experiment pipeline

`runDS.smk` is the Snakemake file that reproduces the paper's experiments end-to-end: `make_plots` → `train_test_split_data` → `train_net`/`crossvalidate_net` → `validate_net` → `predict_benchmark`/`ensemble_benchmark`. It hardcodes absolute data paths (`/data/deepSpecas/...`) that only exist on the original compute environment — treat it as documentation of the pipeline shape and wildcard/filename conventions (e.g. `{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth`) rather than something runnable as-is.

## v2 pipeline / data flow (`dsignal/` + `sim/`)

Two halves: `sim/` manufactures labelled training data, `dsignal/` extracts features from BAMs and trains/runs the model. The five classes are `("A3", "A5", "CSE", "NN", "RI")` — fixed order, defined once in `dsignal/dataset.py`.

### `dsignal/` — extraction → features → model → evaluation

1. **`extract.py`** — the signal extractor. Junction set + base-resolution coverage per `region,bam1,bam2` row, straight from the BAMs. **Never reads a GTF**: strand comes from STAR's `jM` tag (with the `+20` annotated bit masked off via `% 20`) or a reference-dinucleotide fallback. Writes one `.npz` per event.
2. **`features.py`** — the single source of truth for the representation, consumed by *both* the GBM baseline and the Transformer so they are compared on identical information. Produces a per-junction token matrix `[J, 48]` and an event-level vector `[32]` (window geometry, coverage moments, pooled topology summaries, the dropped-junction tail). Minus-strand records are canonicalized by mirroring within the window, so "donor" always means the transcription-direction donor and the left-right flip is an exact A3↔A5 map.
3. **`dataset.py`** — torch `Dataset`, the five augmentations (`flip`, `condition_swap`, `subsample`, `junction_dropout`, `jitter`, all off by default), `FeatureScaler` (fit on the training fold only, stored in the checkpoint), and `collate`.
4. **`model.py`** — `JunctionSetTransformer`. A junction set is a *set*, not a sequence, so this is a pre-norm encoder with two learned additive attention biases carrying the structure that matters: bucketized signed log-distance between junction midpoints, and a per-head scalar for each discrete pairwise relation (`identical`/`shares_donor`/`shares_acceptor`/`nested`/`overlapping`/`disjoint`). The CLS token is *projected from the event-level vector* rather than being a free parameter, and is never masked — so a zero-junction event is a valid CLS-only forward pass.
5. **`train.py`** — 5-fold `GroupKFold` **grouped by gene**, with an explicit no-leak assertion; out-of-fold predictions pooled into one report. Writes one checkpoint per fold.
6. **`predict.py`** — inference. Feature config, scaler, classes and feature subset all come from the checkpoint, never from CLI defaults. Several fold checkpoints are averaged. Also holds `PriorCorrection` for a train/deploy class-prior mismatch — **applied per fold before the folds are averaged**; correcting the averaged posterior is a different and wrong estimator.
7. **`score_benchmark.py`** — scores checkpoints against the 140-event curated benchmark (see below). Reports 5-class and A3+A5→A collapsed accuracy, both confusion matrices, accuracy vs region width, and wall clock.
8. **`baseline_gbm.py`** — the LightGBM baseline, and (importantly) the **shared evaluation harness**: `grouped_folds`, `report`, `report_collapsed`, `print_vs_reference`. Route any new model through these rather than reimplementing, or the numbers stop being comparable to the layer-by-layer history.
9. **`audit_noise.py`** — noise-calibration audit, subcommands `measure` / `compare` / `padding`. Annotation-free and window-local, so the identical code path runs on real and simulated BAMs.
10. **`prior_sweep.py`** — one benchmark extraction pass, N prior settings evaluated analytically from cached per-fold posteriors. **`verify.py` / `verify_features.py`** — the gates listed above. **`diagnose_cse_a3.py`** — annotation-free structural motif scoring. **`tabulate_runs.py`** — side-by-side table of `train.py --json-out` files.

### `sim/` — the simulator

Pipeline order:

```
gtf_gene_spans / gtf_transcript_lengths / gtf_exon_structure   (GTF-derived tables)
  → catalogue.py            SUPPA .ioe files → Event objects (SE→CSE; MX/AF/AL excluded)
  → select_events.py        balanced design: positives + nn_plain / nn_hard / nn_novel
    + window.py             the extraction window around each event span
  → gene_neighbors, novel_junctions, build_premrna_fasta      (noise layers 2-4 inputs)
  → build_experiment.py     transcript × sample count matrix + the noise-layer knob panel
  → order_fasta.py          polyester matches matrix to FASTA by row index, not by name
  → run_polyester.R → align_reads.sh (STAR)      [run_batch.sh drives a whole batch]
  → build_extract_csv.py → dsignal.extract → merge_batches.py → training
```

Things that are easy to get wrong here:

- **One gene per event, and gene-disjointness across batches.** `select_events.py --exclude-genes` prevents cross-batch leakage that `GroupKFold` cannot see, because it groups within one labels file.
- **Independent RNG streams** per window draw, negative-span draw and noise layer, so adding a layer does not perturb earlier ones.
- **Window geometry is load-bearing.** `sim/window.py` is on its *third* sampler; its docstring deliberately retains all three as a decision record. Only v3 is live: `window = event span + U[0, 100] bp per edge`, label-independently. Switching this was worth +16.5 points of zero-shot transfer.

### The benchmark, and what "zero-shot" means here

`benchmark/{A3,A5,ES,IR}.csv` — 140 manually-curated real events over `benchmark/c1.bam`/`c2.bam`. **It contains no NN events**, so every NN prediction is a false positive and NN recall is unmeasurable. The v2 model is trained purely on simulation and scored here with no fine-tuning, which is the point of the comparison.

**It is split.** `benchmark/split.tsv` (committed, 70 dev / 70 test, stratified, cluster-disjoint at 50 kb) — everything tuned is read off **dev**, test is scored once per milestone. Use `--half {dev,test,both}` on `score_benchmark`/`prior_sweep`, and `python -m dsignal.benchmark_split compare A.tsv B.tsv --half dev` for the paired McNemar. At n=70 a real difference needs 8–10 events, against 5–6 at n=140.

**Which v1 number to compare against.** v1's famous **87.9%** (123/140) is the *six-net ensemble*. A **single** v1 representation (`squishstack`) trained on v1's own data scores **75.7%**, and trained on v2's simulated data scores **74.3%** (`runs/v1_crossover/`, `runs/v1_pretrained/`) — so a single v2 model should be measured against ~76%, not 87.9%. The older 78.6% figure is **retracted**: its epoch was chosen on the fold it reported.

**Split-wise (`benchmark/split.tsv`), the numbers are** (`dsignal.v1_reference`, `plans/signal-native-v2-ensemble-results.md`): v1 six-net ensemble **60/70 dev, 63/70 test** (60+63=123/140, consistent with the all-140 figure above); v1 single `squishstack` net 56/70 dev, 50/70 test. v2's ensemble (`runs/ensembles/e3_bagged.json`, 16 members, `--ensemble`) reaches **77.1% dev (54/70) / 85.7% test (60/70)**; a single v2 model (no ensembling) reaches 75.7% dev (53/70) / 82.9% test (58/70) (`runs/geomwin_iv30_refit` + TTA + `nn_weight` 0.1). **Test was scored once** and the pre-registered parity/win criterion against v1 was **not** met (McNemar p=0.549, discordant 4 v2-only / 7 v1-only) — any further v1-vs-v2 comparison on this benchmark needs a fresh held-out half. An earlier ad hoc analysis quoted v1 at 62/69 test (122/139 all); that figure is **retracted** — it deduplicated by region alone rather than `(csv, region)`, dropping one of the two rows for the region that appears in both `A3.csv` and `A5.csv`.

Honesty caveats: the NN operating point is **settled** — pre-registered from the deployment argument in `dsignal/prior_sweep.py`'s docstring, then scored on dev. Still open: `DPSI_BETA`/`PSI_LEVEL_BETA` in `sim/select_events.py` are fitted on all 140 benchmark events, so anything using `--psi-dist beta` is domain-calibrated along that axis; if it is used again, refit on dev only.

Every benchmark number this branch has produced is in `runs/benchmark_*.log`; grep `aggregate collapsed`.

## Running with pretrained models

Pretrained weights ship in `data/trained_nets/<mode>/trained.pth` for all six modes. Example CSVs for the manually-curated benchmark dataset (`A3.csv`, `A5.csv`, `ES.csv`, `IR.csv`) live in `benchmark/`, alongside the source BAMs (`c1.bam`/`c2.bam`, two experimental conditions) and IGV screenshots (`benchmark/igv_images/`) of the curated truth-set events — see `benchmark/README.md` for how that truth set was built (rMATS + SUPPA consensus, PRJNA255099).

Ensemble inference on the full benchmark:
```bash
python dsEnsemble.py \
--csv benchmark/A3.csv \
--pyplot2s data/trained_nets/pyplot2s/trained.pth \
--colmplp2s data/trained_nets/colmplp2s/trained.pth \
--squishstack data/trained_nets/squishstack/trained.pth \
--squish4d data/trained_nets/squish4d/trained.pth \
--comb data/trained_nets/comb/trained.pth \
--comb4d data/trained_nets/comb4d/trained.pth \
--zoom -1 --a --cse
```

Single-representation inference:
```bash
python deepSpecas.py predict --csv benchmark/IR.csv --zoom -1 -n data/trained_nets/squishstack/trained.pth --a --cse -p squishstack
```

Input CSVs for both are `region,BAM_path_condition1,BAM_path_condition2` (no header).

Add `--shard i/N` to split a CSV across parallel processes, and `--gpu-render` for the device-side render path (faster, not bit-exact).

## Running v2

The single-model recipe below (75.7% dev / 82.9% test collapsed) is still the base for every ensemble member. **The test-validated headline is the ensemble** (`runs/ensembles/e3_bagged.json`, 16 members, 77.1% dev / 85.7% test) — see `plans/signal-native-v2-ensemble-results.md`. Score it with:

```bash
python -m dsignal.score_benchmark --ensemble runs/ensembles/e3_bagged.json \
  --half dev --tta-condition-swap --target-prior nn-weight --nn-weight 0.1
```

**Dev-tuned, not test-validated, current best: 87.1% dev collapsed** (`runs/ensembles/e4_union2.json`, 6 members trained on `npz_union2`, 12,480 events — 1.44× `npz_geomwin`'s size — see `plans/signal-native-v2-scale-union2-results.md`). Single-model scale on `npz_union2` alone plateaus at 56–57/70 dev past ~8,640 events (in-sim macro-F1 keeps climbing, zero-shot transfer does not); reapplying Stage 2's view-diversity ensemble on top of the largest pool reaches 61/70 dev collapsed, +7 events over the best single run there, not significant at n=70 (McNemar p=0.13). **Test is spent** (Stage 6) and this number has no test exposure — do not compare it against v1 or promote it to "the deliverable" without a fresh held-out benchmark half.

Reproducing the single-model best (75.7% dev / 82.9% test collapsed), in the local `DSnet` env:

```bash
# train (5 folds, grouped by gene) -- ~4,800 simulated events under v3 windows.
# --refit and --inner-val-frac matter: the epoch is chosen on a gene-disjoint
# slice of each *training* fold, never on the fold being reported.
python -m dsignal.train \
  --npz-dir sim/experiment/npz_geomwin --labels sim/experiment/labels_geomwin.tsv \
  --augment all --feature-subset no-geometry --refit --inner-val-frac 0.30 \
  --out-dir runs/geomwin_iv30_refit --json-out runs/geomwin_iv30_refit.json

# score zero-shot against the curated real events -- dev while iterating,
# test once per milestone
python -m dsignal.score_benchmark --net runs/geomwin_iv30_refit/fold*.pt \
  --half dev --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
  --train-labels sim/experiment/labels_geomwin.tsv \
  --out runs/benchmark_events_stack_dev.tsv

# paired McNemar against another run
python -m dsignal.benchmark_split compare \
  runs/benchmark_events_stack_dev.tsv runs/benchmark_events_<new>.tsv --half dev
```

Inference on an arbitrary `region,bam1,bam2` CSV — same input format as v1, so the two are a drop-in comparison:

```bash
python -m dsignal.predict --csv benchmark/A3.csv --net runs/geomwin_allaug/fold*.pt
```

`--feature-subset` defaults to `no-geometry`. The other two values are `all` and `geometry` (the geometry-only control).

## Workflow conventions (how to collaborate on this repo)

### Git
- Commits are fine to create as work is completed, but **never `git push`** — pushing is always the user's call.
- **Never create or switch branches on your own initiative.** Branching decisions belong to the user. If you think a new branch is warranted (e.g. a risky or large change), propose it and wait for a decision instead of creating one.

### Planning before writing code
Whenever a task requires writing or modifying code:
1. Produce a **step-by-step plan** and discuss it with the user first. Do not start implementing.
2. For each step, note **which model tier should implement it**, based on the step's difficulty (e.g. mechanical/boilerplate steps → a cheaper/faster model; steps requiring judgment, architecture decisions, or tricky debugging → a stronger model).
3. Wait for explicit approval of the plan before doing any implementation work.
4. Once approved, **save the plan to a file** under `plans/` (e.g. `plans/<short-slug>.md`) and **stop** — do not implement it in the current session. A separate session will pick up the saved plan file and do the implementation.
5. Alongside the saved plan, give the user a **short prompt** they can manually paste into that new implementation session to kick it off (should reference the plan file path and summarize the goal in a sentence or two).

### After implementation
Once a plan has been implemented (in the later session that does the actual work):
- Write a concise report back to the user summarizing what was done, deviations from the plan (if any), and what was verified.
- Commit the work (no push, per above).

## Conventions worth knowing before editing

### v2 (`dsignal/`, `sim/`)

- **Every comparison on the 140-event benchmark needs a significance test.** Differences of 4–6 events are noise at this size; several past runs differ by exactly that much. McNemar on the paired per-event outcomes is what the branch uses throughout.
- **Never delete a measured number from a docstring or plan doc.** Retracted findings stay, with the retraction next to them. That is what makes the plan chain readable, and two findings have already been retracted.
- Build → verify → commit each step before starting the next.
- `--a`/`--cse`-style class collapsing does not exist in v2; the five classes are fixed and `score_benchmark.py` reports the A3+A5→A collapse as a derived number so it stays comparable to v1's 123/140.
- Feature-subset changes touch **three** call sites: `baseline_gbm.apply_feature_subset`, `dataset.apply_feature_subset`, and `dataset.apply_feature_subset_meta` (the last is the easy one to forget — it keeps `s_rel`/`e_rel` consistent with the dummy token count under the geometry control). The geometry group itself lives as private names in `baseline_gbm.py` and is imported by `dataset.py`.
- `N_TOKEN_FEATURES`/`N_EVENT_FEATURES` are baked into `ModelConfig` defaults and into every saved checkpoint. Adding a feature does not break loading (config comes from the checkpoint) but will mismatch the stored scaler.

### v1 (root-level modules)

- `--a` collapses classes `A3`+`A5` into `A`; `--cse` collapses `CE`+`SE` into `CSE`. Both flags must match between how a net was trained and how it's used for validate/predict — `EventDataset._build_classes` and the hardcoded `classes` list in `deepSpecas.main`/`dsEnsemble.main` must be built identically or the loaded `state_dict`'s final `fc` layer shape (and label order, since `classes` is always `sorted()`) will mismatch.
- `--zoom -1` means "no zoom" (use the full region); positive integers are flanking-region-trim widths applied around a coverage gap (see `build_zoom` in `plot2s.py`).
- `dsEnsemble.py` and `deepSpecas.py predict` currently hardcode `device = "cuda:0"` (no CPU fallback), unlike `train`/`validate`/`plot.py` paths which check `torch.cuda.is_available()`.
- The `--explain` (CAM/class-activation-map) option in `deepSpecas.py predict` is unimplemented (`raise NotImplementedError`); the implementation is stubbed out in comments referencing `_getCAM`/`_saveCAM`.
