#!/usr/bin/env bash
set -uo pipefail
export PATH=/datasets/deepSpecas-v2/envs/DSnet/bin:$PATH
cd /home/ciccolella/code/deepSpecas
ARENA="--csv ptbp1_arena/candidates/A3.csv ptbp1_arena/candidates/A5.csv ptbp1_arena/candidates/ES.csv ptbp1_arena/candidates/IR.csv ptbp1_arena/candidates/NN.csv"
LAB=sim/experiment/labels_union2.n12480.tsv
score_arena() {  # name net...
  local name=$1; shift
  python -m dsignal.score_benchmark --net "$@" $ARENA --half both --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 --train-labels $LAB \
    --out runs/arena_$name.tsv > runs/arena_$name.log 2>&1
  echo "DONE $name: $(grep -o 'aggregate collapsed: [0-9/]*' runs/arena_$name.log)"
}
score_arena union2_base_s0 runs/scale_union2_n12480/fold*.pt
score_arena union2_base_s1 runs/union2_base_s1/fold*.pt
score_arena union2_base_s2 runs/union2_base_s2/fold*.pt
echo BASELINES_DONE
