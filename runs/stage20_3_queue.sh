#!/usr/bin/env bash
# Stage 20.3: arms B (head-only), C (full fine-tune), A (mixed GRL) x seeds 0/1/2.
# Flags are exactly plans/signal-native-v2-stage20-real-finetune.md's "Fixed
# before running" section -- do not tune anything here. Modelled on
# runs/stage18_5_queue.sh.
set -uo pipefail
export PATH=/datasets/deepSpecas-v2/envs/DSnet/bin:$PATH
cd /home/ciccolella/code/deepSpecas

REAL_NPZ=sim/experiment/npz_real3; REAL_LAB=sim/experiment/labels_real3.tsv
MIX_NPZ=sim/experiment/npz_union2_real3; MIX_LAB=sim/experiment/labels_union2_real3.tsv
PARENT_LAB=sim/experiment/labels_union2.n12480.tsv
BNEW="--split benchmark_new/split.tsv --csv benchmark_new/A3.csv benchmark_new/A5.csv benchmark_new/ES.csv benchmark_new/IR.csv benchmark_new/NN.csv"
ARENA="--csv ptbp1_arena/candidates/A3.csv ptbp1_arena/candidates/A5.csv ptbp1_arena/candidates/ES.csv ptbp1_arena/candidates/IR.csv ptbp1_arena/candidates/NN.csv"

parent_of() { case $1 in 0) echo runs/scale_union2_n12480;; 1) echo runs/union2_base_s1;; 2) echo runs/union2_base_s2;; esac; }

score_one() {  # name train-labels
  local name=$1 train_lab=$2
  python -m dsignal.score_benchmark --net runs/$name/fold*.pt --half dev $BNEW --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $train_lab \
    --out runs/bnew_dev_$name.tsv > runs/bnew_dev_$name.log 2>&1
  python -m dsignal.score_benchmark --net runs/$name/fold*.pt --half dev --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $train_lab \
    --out runs/benchmark_events_${name}_dev.tsv > runs/benchmark_${name}_dev.log 2>&1
  python -m dsignal.score_benchmark --net runs/$name/fold*.pt $ARENA --half both --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $train_lab \
    --out runs/arena_$name.tsv > runs/arena_$name.log 2>&1
  echo "SCORED $name: bnew $(grep -o 'aggregate collapsed: [0-9/]*' runs/bnew_dev_$name.log) old $(grep -o 'aggregate collapsed: [0-9/]*' runs/benchmark_${name}_dev.log) arena $(grep -o 'aggregate collapsed: [0-9/]*' runs/arena_$name.log)"
}

run_B() {  # seed
  local s=$1 parent=$(parent_of $s) name=stage20_B_s$s
  if [[ ! -f runs/$name/fold4.pt ]]; then
    python -m dsignal.train --npz-dir $REAL_NPZ --labels $REAL_LAB --augment all --feature-subset no-geometry \
      --seed $s --init-from $parent --freeze encoder --select-on none --epochs 20 --lr 1e-4 \
      --balance-real-cells 0 --class-weights none \
      --out-dir runs/$name --json-out runs/$name.json --dump-oof runs/$name.oof.tsv \
      > runs/$name.log 2>&1 || { echo "TRAIN FAILED $name"; return; }
  fi
  score_one $name $PARENT_LAB
}

run_C() {  # seed
  local s=$1 parent=$(parent_of $s) name=stage20_C_s$s
  if [[ ! -f runs/$name/fold4.pt ]]; then
    python -m dsignal.train --npz-dir $REAL_NPZ --labels $REAL_LAB --augment all --feature-subset no-geometry \
      --seed $s --init-from $parent --freeze none --select-on none --epochs 10 --lr 3e-5 \
      --warmup-epochs 2 --avg-last-k 3 --balance-real-cells 0 --class-weights none \
      --out-dir runs/$name --json-out runs/$name.json --dump-oof runs/$name.oof.tsv \
      > runs/$name.log 2>&1 || { echo "TRAIN FAILED $name"; return; }
  fi
  score_one $name $PARENT_LAB
}

run_A() {  # seed
  local s=$1 name=stage20_A_s$s
  if [[ ! -f runs/$name/fold4.pt ]]; then
    python -m dsignal.train --npz-dir $MIX_NPZ --labels $MIX_LAB --augment all --feature-subset no-geometry \
      --seed $s --balance-real-cells 0.10 --class-weights none --domain-lambda 1.0 --grl-schedule ramp \
      --select-on inner --inner-val-frac 0.30 --refit \
      --out-dir runs/$name --json-out runs/$name.json --dump-oof runs/$name.oof.tsv \
      > runs/$name.log 2>&1 || { echo "TRAIN FAILED $name"; return; }
  fi
  score_one $name $MIX_LAB
}

# sensitivity re-score of B/C's own real-only training labels (reported, not gated)
score_sens() {  # name (must exist already)
  local name=$1
  python -m dsignal.score_benchmark --net runs/$name/fold*.pt --half dev $BNEW --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $REAL_LAB \
    --out runs/bnew_dev_${name}_sens.tsv > runs/bnew_dev_${name}_sens.log 2>&1
  python -m dsignal.score_benchmark --net runs/$name/fold*.pt --half dev --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $REAL_LAB \
    --out runs/benchmark_events_${name}_sens_dev.tsv > runs/benchmark_${name}_sens_dev.log 2>&1
  echo "SENS $name: bnew $(grep -o 'aggregate collapsed: [0-9/]*' runs/bnew_dev_${name}_sens.log) old $(grep -o 'aggregate collapsed: [0-9/]*' runs/benchmark_${name}_sens_dev.log)"
}

for s in 0 1 2; do run_B $s; done
for s in 0 1 2; do run_C $s; done
for s in 0 1 2; do run_A $s; done
for s in 0 1 2; do score_sens stage20_B_s$s; done
for s in 0 1 2; do score_sens stage20_C_s$s; done
echo QUEUE_DONE
