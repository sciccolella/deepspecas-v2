# deepSpecas

`deepSpecas` classifies alternative splicing (AS) events from RNA-seq BAM files. Coverage
and read data from two experimental conditions are rendered into an image representation,
and a ResNet50 classifies that image into a splicing event type (`A3`, `A5`, `CE`/`SE`,
`RI`, `NN`; `A3`+`A5` can be collapsed into `A` and `CE`+`SE` into `CSE`).

Six image representations are supported, and are named consistently across every tool:
`pyplot2s`, `colmplp2s`, `squishstack`, `squish4d`, `comb`, `comb4d`.

> This README documents the released tool. Every command below accepts `--help`; the
> option lists here are summaries, not verbatim help output.

# Build env

```bash
mamba create -n DSpoly2 -c bioconda -c conda-forge -c pytorch -c nvidia pysam snakemake matplotlib rich pytorch torchvision torchaudio pytorch-cuda=11.8 scikit-learn pandas seaborn opencv
conda activate DSpoly2
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

# Input format

Both inference tools take a headerless CSV, one event per line, with **two BAM files —
one per experimental condition**:

```
Genomic_position,Path_to_BAM_for_condition1,Path_to_BAM_for_condition2
```

Example files for the manually-curated benchmark dataset ship in
`benchmark/[A3|A5|ES|IR].csv`; see [`benchmark/README.md`](benchmark/README.md) for how
that truth set was built.

# Run `deepSpecas` using trained nets

Alongside the tool we release the trained networks used in the manuscript, so
`deepSpecas` can be used directly without retraining all the different representations.
Weights ship in `data/trained_nets/<mode>/trained.pth` for all six modes.

## Ensemble

`dsEnsemble.py` loads one trained net per image representation, renders each
representation on the fly per event, and prints the ensemble prediction.

Key options:

| option | meaning |
|---|---|
| `--csv CSV` | events to predict (format above) |
| `--zoom ZOOM` | **required**; `-1` means no zoom (use the full region) |
| `--pyplot2s`, `--colmplp2s`, `--squishstack`, `--squish4d`, `--comb`, `--comb4d` | trained `.pth` for that representation; pass as many as you want to ensemble |
| `--a` / `--cse` | collapse `A3`+`A5` → `A` / `CE`+`SE` → `CSE`. Must match how the nets were trained |
| `--mincov MINCOV` | minimum coverage to plot |
| `--imgout DIR` | save the rendered images |
| `--shard i/N` | process only shard `i` of `N`; every CSV line is independent, so shards can run in parallel and be concatenated |
| `--gpu-render` | opt-in device-side render path. Much faster, but **not bit-exact** with the default PIL path — borderline predictions can flip |

Example run on A3 for the benchmark dataset:

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

`--explain` and `--version` are accepted but currently ignored.

## Single representation

To run a single image representation, use `deepSpecas.py` in `predict` mode:

| option | meaning |
|---|---|
| `--csv CSV` | events to predict (format above) |
| `-p, --plot MODE` | one of the six representations |
| `-n, --net NET` | trained net file |
| `--zoom ZOOM` | `-1` for no zoom |
| `--a` / `--cse` | class collapsing, must match training |
| `--resizescale F` | rescale images before the net (default `1.0`) |
| `--wd DIR` | working directory for the paths inside the CSV |

Example run on intron retention for the benchmark dataset using the `squishstack`
representation:

```bash
python deepSpecas.py predict --csv benchmark/IR.csv --zoom -1 -n data/trained_nets/squishstack/trained.pth --a --cse -p squishstack
```

# Train `deepSpecas`

## Build images from samples

`plot2s.py` renders a CSV of events into an image directory and writes the matching
`image_path,label` CSV that training consumes.

| option | meaning |
|---|---|
| `-i, --input CSV` | CSV of events |
| `-o, --outdir DIR` | output image directory |
| `--outcsv CSV` | output CSV of events |
| `-m, --mode MODE` | one of the six representations |
| `-d, --datadir DIR` | root directory containing the BAMs |
| `--zoom [ZOOM ...]` | default `-1` |
| `--mincov MINCOV` | minimum coverage to plot |
| `--resize` | resize images on output |

## Train the nets

`deepSpecas.py train` trains one net on one representation.

| option | meaning |
|---|---|
| `--train CSV` | training CSV of `image_path,label` |
| `--val CSV` | validation CSV; if omitted, `--train` is split randomly |
| `-p, --plot MODE` | one of the six representations |
| `-o, --net-out FILE` | where to write the trained weights |
| `--epochs`, `--batch-size` | training loop |
| `--crossvalidation` | k-fold cross-validation instead of a single fit |
| `--sample-frac F` | downsample the training set |
| `--a` / `--cse` | class collapsing |
| `--resizescale F` | rescale images before the net |
| `--hm FILE` | confusion-matrix heatmap output (default `hm.png`) |

`deepSpecas.py validate` loads a trained net, scores it against a validation CSV, and
reports precision/recall/F-score plus a heatmap.

# Replicate experiments

The entire pipeline is available as a [snakefile](runDS.smk): `make_plots` →
`train_test_split_data` → `train_net`/`crossvalidate_net` → `validate_net` →
`predict_benchmark`/`ensemble_benchmark`.

Note that `runDS.smk` hardcodes absolute data paths (`/data/deepSpecas/...`) that only
exist on the original compute environment. Treat it as documentation of the pipeline
shape and of the wildcard/filename conventions rather than as something runnable as-is.
