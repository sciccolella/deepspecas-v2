#!/usr/bin/env bash
# Step 3 + 4 of plans/signal-native-v2-scaleup-7500.md, on the local GPU box.
#
# Everything the checkpoint reports for a data set, in the order it is read:
# the two leak controls first (they are cheap and must be re-confirmed at each
# new scale, never assumed), then the real models, then zero-shot transfer to
# the curated benchmark raw and prior-corrected, then the NN-fraction ablation.
#
# Usage:
#   sim/run_measurements.sh <npz_dir> <labels_tsv> <tag> [extra_labels_tsv ...]
#
# Extra labels files (e.g. the NN-resampled ones from merge_batches.py) get the
# full Transformer + benchmark treatment too, which is what makes step 4's four
# numbers comparable.
set -euo pipefail

NPZ=${1:?usage: run_measurements.sh <npz_dir> <labels_tsv> <tag> [extra_labels ...]}
LABELS=${2:?}
TAG=${3:?}
shift 3
EXTRA=("$@")

PY=${PY:-/datasets/deepSpecas-v2/envs/DSnet/bin/python}
RUNS=${RUNS:-runs}
mkdir -p "$RUNS"

step() { echo -e "\n[$(date +%H:%M:%S)] === $* ===" >&2; }

# --- leak controls ----------------------------------------------------------
step "$TAG: width-only leak test"
"$PY" sim/width_leak_check.py --npz-dir "$NPZ" --labels "$LABELS" \
  2>&1 | tee "$RUNS/${TAG}_width_leak.log"

step "$TAG: LightGBM geometry control"
"$PY" -m dsignal.baseline_gbm --npz-dir "$NPZ" --labels "$LABELS" \
  --feature-subset geometry 2>&1 | tee "$RUNS/${TAG}_gbm_geometry.log"

step "$TAG: Transformer geometry control"
"$PY" -m dsignal.train --npz-dir "$NPZ" --labels "$LABELS" \
  --feature-subset geometry --json-out "$RUNS/${TAG}_geometry.json" \
  2>&1 | tee "$RUNS/${TAG}_geometry.log"

# --- the real models --------------------------------------------------------
step "$TAG: LightGBM (no-geometry)"
"$PY" -m dsignal.baseline_gbm --npz-dir "$NPZ" --labels "$LABELS" \
  2>&1 | tee "$RUNS/${TAG}_gbm.log"

step "$TAG: Transformer, no augmentation"
"$PY" -m dsignal.train --npz-dir "$NPZ" --labels "$LABELS" \
  --json-out "$RUNS/${TAG}_noaug.json" \
  2>&1 | tee "$RUNS/${TAG}_noaug.log"

# --- the model we actually ship, plus transfer ------------------------------
for lab in "$LABELS" "${EXTRA[@]}"; do
  name=$TAG
  [[ "$lab" != "$LABELS" ]] && name="${TAG}_$(basename "$lab" .tsv | sed 's/.*\.//')"
  step "$name: Transformer, all augmentations"
  "$PY" -m dsignal.train --npz-dir "$NPZ" --labels "$lab" \
    --augment all --out-dir "$RUNS/${name}_allaug" \
    --json-out "$RUNS/${name}_allaug.json" 2>&1 | tee "$RUNS/${name}_allaug.log"

  step "$name: benchmark, uncorrected"
  "$PY" -m dsignal.score_benchmark --net "$RUNS/${name}_allaug"/fold*.pt \
    --out "$RUNS/benchmark_events_${name}.tsv" 2>&1 | tee "$RUNS/benchmark_${name}.log"

  step "$name: prior sweep"
  "$PY" -m dsignal.prior_sweep --net "$RUNS/${name}_allaug"/fold*.pt \
    --train-labels "$lab" --out "$RUNS/prior_sweep_${name}.tsv" \
    2>&1 | tee "$RUNS/prior_sweep_${name}.log"
done

step "$TAG: all measurements done"
