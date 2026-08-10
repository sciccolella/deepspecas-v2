# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`deepSpecas` is a research tool for classifying alternative splicing (AS) events from RNA-seq BAM files using image-based deep learning: pileup/coverage data from paired BAM files is rendered into one of several image representations, and a CNN (ResNet50) classifies the image into a splicing event type (`A3`, `A5`, `CE`/`SE` or collapsed `CSE`, `RI`, `NN`, optionally collapsed `A3`+`A5` → `A`).

## Environment setup

No requirements.txt/pyproject — dependencies are installed via conda/mamba:

```bash
mamba create -n DSpoly2 -c bioconda -c conda-forge -c pytorch -c nvidia pysam snakemake matplotlib rich pytorch torchvision torchaudio pytorch-cuda=11.8 scikit-learn pandas seaborn opencv
conda activate DSpoly2
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

There is no test suite, linter, or CI config in this repo. There are no build steps — scripts are run directly with `python`.

## Core pipeline / data flow

1. **`plot2s.py`** — turns a genomic region + two BAM files (two experimental conditions) into an image representation. Defines the `Event2sample` / `Event2sampleReads` / `Event2sampleComb` classes (each implementing `EventAbstract`: `build_zoom`, `strip`, `plot`), one per image-representation "mode": `pyplot2s`, `colmplp2s`, `squishstack`, `squish4d`, `comb`, `comb4d`. Also runnable standalone as a CLI to batch-render a CSV of events into an image directory (used by the `make_plots` Snakemake rule).
2. **`plot.py`** — an earlier/parallel single-sample (not two-condition) plotting module with its own `Event`/`Event2sample`/`Event2sampleReads` classes and `pyplot_plt`/`colmplp_plt` renderers. `jitimg.py` is a small ad-hoc/debug driver built on `plot.py` + `DSutils.py` (not part of the Snakemake pipeline).
3. **`DSutils.py`** — GTF parsing (`parse_gtf`, `parse_multi_gtf`) and exon/transcript-mask building (`get_transcript`), used by the single-sample path.
4. **`mc_mplp.py`** — `Channels`/`Channels2d`/`ChannelsMat2d` helper classes for turning per-base coverage/skip arrays into stacked image-channel matrices; used internally by the plotting modules.
5. **`deepSpecas.py`** — the main model CLI (`EventDataset`, `_build_model` (ResNet50, first conv layer patched to accept 3 or 4 channels depending on plot mode), `_build_transformer`, `train_model`, `kfold_crossvalidation`, `validate`, `predict`). Three subcommands:
   - `train` — trains a net on a CSV of (image path, label), writes weights (`-o/--net-out`) and a confusion-matrix heatmap (`--hm`).
   - `validate` — loads a trained net, runs it against a validation CSV, reports precision/recall/fscore and a heatmap.
   - `predict` — given a CSV of `region,BAM_path1,BAM_path2`, renders images on the fly (via `plot2s.py`) and classifies them (no train/val CSV of pre-rendered images needed).
6. **`dsEnsemble.py`** — loads up to 6 trained nets (one per image representation, each passed as a separate `--<mode>` flag pointing at a `.pth` file), reuses `deepSpecas._build_model`/`_build_transformer`, renders each representation on the fly per event, sums softmax outputs across nets, and prints the ensemble prediction. This is the recommended way to run inference with the released pretrained models (see README).

Six image "plot modes" are used consistently as an enum-like string across the whole codebase: `pyplot2s`, `colmplp2s`, `squishstack`, `squish4d`, `comb`, `comb4d`. 3-channel modes (`pyplot2s`, `squishstack`, `comb`) load as RGB; 4-channel modes (`colmplp2s`, `squish4d`, `comb4d`) load as CMYK. Each mode has a fixed `RESIZE_SIZE` (duplicated as a dict in both `DSutils`-adjacent `deepSpecas.py` and `plot2s.py`) that images are resized to before being fed to the net — scaled by `--resizescale` when given.

### Full experiment pipeline

`runDS.smk` is the Snakemake file that reproduces the paper's experiments end-to-end: `make_plots` → `train_test_split_data` → `train_net`/`crossvalidate_net` → `validate_net` → `predict_benchmark`/`ensemble_benchmark`. It hardcodes absolute data paths (`/data/deepSpecas/...`) that only exist on the original compute environment — treat it as documentation of the pipeline shape and wildcard/filename conventions (e.g. `{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth`) rather than something runnable as-is.

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

- `--a` collapses classes `A3`+`A5` into `A`; `--cse` collapses `CE`+`SE` into `CSE`. Both flags must match between how a net was trained and how it's used for validate/predict — `EventDataset._build_classes` and the hardcoded `classes` list in `deepSpecas.main`/`dsEnsemble.main` must be built identically or the loaded `state_dict`'s final `fc` layer shape (and label order, since `classes` is always `sorted()`) will mismatch.
- `--zoom -1` means "no zoom" (use the full region); positive integers are flanking-region-trim widths applied around a coverage gap (see `build_zoom` in `plot2s.py`).
- `dsEnsemble.py` and `deepSpecas.py predict` currently hardcode `device = "cuda:0"` (no CPU fallback), unlike `train`/`validate`/`plot.py` paths which check `torch.cuda.is_available()`.
- The `--explain` (CAM/class-activation-map) option in `deepSpecas.py predict` is unimplemented (`raise NotImplementedError`); the implementation is stubbed out in comments referencing `_getCAM`/`_saveCAM`.
