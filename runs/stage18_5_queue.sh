#!/usr/bin/env bash
# Stage 18.5: base seeds 1/2 on union2 n12480 (seed 0 = runs/scale_union2_n12480),
# then covshape and covshape+delcov at seeds 0/1/2. Each run is trained with the
# single-model recipe and scored on both dev sets.
set -uo pipefail
export PATH=/datasets/deepSpecas-v2/envs/DSnet/bin:$PATH
cd /home/ciccolella/code/deepSpecas
NPZ=sim/experiment/npz_union2; LAB=sim/experiment/labels_union2.n12480.tsv
BNEW="--split benchmark_new/split.tsv --csv benchmark_new/A3.csv benchmark_new/A5.csv benchmark_new/ES.csv benchmark_new/IR.csv benchmark_new/NN.csv"
run() {  # name seed extra-train-args...
  local name=$1 seed=$2; shift 2
  if [[ ! -f runs/$name/fold4.pt ]]; then
    python -m dsignal.train --npz-dir $NPZ --labels $LAB --augment all --feature-subset no-geometry \
      --refit --inner-val-frac 0.30 --seed $seed "$@" --out-dir runs/$name --json-out runs/$name.json > runs/$name.log 2>&1 || { echo "TRAIN FAILED $name"; return; }
  fi
  python -m dsignal.score_benchmark --net runs/$name/fold*.pt --half dev $BNEW --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $LAB --out runs/bnew_dev_$name.tsv > runs/bnew_dev_$name.log 2>&1
  python -m dsignal.score_benchmark --net runs/$name/fold*.pt --half dev --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $LAB --out runs/benchmark_events_${name}_dev.tsv > runs/benchmark_${name}_dev.log 2>&1
  echo "DONE $name: bnew $(grep -o 'aggregate collapsed: [0-9/]*' runs/bnew_dev_$name.log)  old $(grep -o 'aggregate collapsed: [0-9/]*' runs/benchmark_${name}_dev.log)"
}
run union2_base_s1 1
run union2_base_s2 2
run stage18_covshape_s0 0 --event-groups covshape
run stage18_covshape_s1 1 --event-groups covshape
run stage18_covshape_s2 2 --event-groups covshape
run stage18_covdel_s0 0 --event-groups covshape delcov
run stage18_covdel_s1 1 --event-groups covshape delcov
run stage18_covdel_s2 2 --event-groups covshape delcov
echo QUEUE_DONE
