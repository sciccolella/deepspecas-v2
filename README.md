# deepSpecas v2

`deepSpecas v2` classifies an alternative-splicing (AS) event from two RNA-seq BAM
files (two experimental conditions) as one of `A3`, `A5`, `CSE` (cassette/skipped
exon), `RI` (retained intron) or `NN` (no event). Unlike
[deepSpecas v1](https://github.com/sciccolella/deepSpecas), which renders the reads
into images and classifies the image with a CNN, v2 renders **nothing**: it extracts the
junction set and base-resolution coverage straight from the BAMs and classifies them
with a junction-set Transformer. The released model is trained **only on simulated
data** (SUPPA event catalogue → PSI-targeted abundances → polyester reads → STAR) and
is used zero-shot on real data.

On the pre-registered `benchmark_new` test half it gets 143/243 (58.8%) against the v1
six-net ensemble's 110/243 (45.3%), McNemar p < 0.0001, at ~0.1 s/event. **It
recognises almost none of the no-event (NN) regions: 3 of 80 on that test half.**
Read [`RESULTS.md`](RESULTS.md) before relying on any number here.

## Install

```bash
mamba env create -f envs/inference.yml     # CPU torch, pysam, numpy, pandas, scikit-learn
conda activate deepspecas-v2
```

This is enough for inference and for `dsignal.verify*`. A GPU is not needed (about
0.13 s/event on CPU with the six-member ensemble). Simulation and training need the
pasteur-style environments in `envs/sim_*.yml` and a CUDA torch; see
[`workflow/README.md`](workflow/README.md). Fetch the benchmark BAMs with
`scripts/fetch_data.sh` ([`DATA.md`](DATA.md)); the model weights are in git
(`models/e4_union2/`, 162 MB).

## Input

A headerless CSV, one event per line, **the same format as v1**:

```
region,bam_condition1,bam_condition2
16:24906764-24907175,benchmark_new/c1.bam,benchmark_new/c2.bam
```

`region` is `chrom:start-end` with chromosome names spelled as in the BAM header. BAMs
must be coordinate-sorted and indexed. The extractor never reads a GTF: strand comes
from STAR's `jM` tag, or from reference dinucleotides when no tag is present.

## Run

With no model given, `dsignal.predict` runs the released ensemble
(`models/e4_union2/manifest.json`: 6 members × 5 folds) at the operating point that was
frozen for the test spend: condition-swap test-time augmentation, and prior correction
`--target-prior nn-weight --nn-weight 0.1`.

```bash
python -m dsignal.predict --csv events.csv --out predictions.tsv
```

Standard output has one line per event: `region`, predicted class, the five class
probabilities, and seconds. `--out` writes a TSV with columns
`region  prediction  p_A3  p_A5  p_CSE  p_NN  p_RI` (probabilities are the ensemble
average). Run from the repository root, or pass `--ensemble /path/to/manifest.json`.

| option | effect |
|---|---|
| `--min-confidence 0.5` | withhold the call (`prediction = NA`) when the top probability is below the threshold, instead of forcing one. Reported as lost coverage |
| `--no-tta` | turn off condition-swap TTA (half the inference time) |
| `--target-prior none` | no class-prior correction (see below) |
| `--net runs/x/fold*.pt` | one training run's folds instead of the ensemble (every option opt-in) |
| `--device cpu` | force CPU (the default is CUDA if available) |
| `--min-anchor`, `--max-nh`, `--min-mapq` | read filters of the extractor (defaults 5, 10, 0) |

The prior correction re-weights the posterior from the training class mix (NN is 25% of the
training events; 15% for the `nn15` member) to a deployment prior in which NN has weight 0.1 relative to
an event class (`nn_weight 0.1`, pre-registered in `dsignal/prior_sweep.py`). It trades
NN recall for accuracy on the event classes.

## What it is not good at

**NN (no splicing event) is essentially unrecognised on curated real data:** 3/80
(3.8%) on the `benchmark_new` test half, 11/79 on dev; 77 of the model's 100 test
errors are NN regions called as CSE, A3, A5 or RI. v1 scores 0/80 on the same events,
so this is a limitation shared with the image model, not a regression, but the
classifier should not be used to *rule out* an event. Excluding NN truth the accuracy
is 85.9% (140/163). Use `--min-confidence` if you would rather abstain than guess.

Other limits, in [`RESULTS.md`](RESULTS.md) §5: the simulated training pool shares genes
with the test set; no held-out curated half remains, so further comparisons are
exploratory; accuracy was measured on curated human events only (`benchmark/`, from
PRJNA255099, and `benchmark_new/`), not across organisms or library types.

## Verifying a checkout

```bash
python -m dsignal.verify --csv benchmark/A3.csv benchmark/A5.csv benchmark/ES.csv benchmark/IR.csv
python -m dsignal.verify_features --npz-dir <npz pool>   # needs a simulated pool, DATA.md
python scripts/check_results.py                           # recompute RESULTS.md's tables
```

## Where things are

| | |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every claimable number, its source file, caveats, retracted figures |
| [`DATA.md`](DATA.md) | what is not in git (BAMs, simulated pools, other checkpoints) and where it lives |
| [`plans/README.md`](plans/README.md) | the decision record: every stage's plan, pre-registration and result |
| [`HISTORY.md`](HISTORY.md) | how this repository was split out of v1's, and the old→new commit map |
| `dsignal/` | extraction, features, model, training, inference, scoring |
| `sim/` | the simulator (GTF tables → event design → polyester → STAR → extraction) |
| `benchmark*/` | curated truth sets (CSVs, splits, curation ledger) |
| `workflow/` | the reproduction driver |

## Credit

This project is the successor of **deepSpecas v1** (image-based):
<https://github.com/sciccolella/deepSpecas>. The manually curated benchmark of v1
(built from Best et al., Nat Commun 2014, PRJNA255099, with an rMATS + SUPPA consensus)
is reused here as `benchmark/`; `benchmark/README.md` documents how it was built.
Citation details for v2 will be added when the manuscript exists.
