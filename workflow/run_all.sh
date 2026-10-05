#!/usr/bin/env bash
# DOCUMENTATION-GRADE driver for the whole v2 pipeline, in order. It records the
# commands, flags and gates that produced the released `e4_union2` ensemble; it
# has NOT been run end to end as one script (the real work ran in stages over
# weeks, on two machines, driven by hand -- see workflow/README.md). Treat it as
# the map, like v1's runDS.smk. Stages are functions; run one at a time:
#
#     workflow/run_all.sh <stage>     # design | simulate | merge | train | score
#
# Hard-coded paths are the original machines'. Edit the variables below.
set -euo pipefail

PASTEUR_ROOT=${PASTEUR_ROOT:-/data/conda/ciccolella/deepSpecas-v2}   # remote, no GPU
ENVS=${ENVS:-/data/conda/ciccolella/envs}
DSNET_PY=${DSNET_PY:-/datasets/deepSpecas-v2/envs/DSnet/bin/python}   # local GPU box

# The 13 batches behind npz_union2 (12,480 events, 12,480 distinct genes), in the
# exclusion-chain order: each batch excludes the genes of ALL earlier designs.
# Their design.tsv/labels.tsv live in sim/experiment/<dir>/<tag>/ -- not in git
# (DATA.md); the chain cannot be rebuilt without them.
UNION2_BATCHES=(
  geomwin/b1 geomwin/b2 geomwin/b3 geomwin/b4 geomwin/b5      # sim/experiment/batches_geomwin/
  bag2/b8 bag2/b9 bag2/b10 bag2/b11                           # sim/experiment/batches_bag2/ (seeds 8-11)
  union2/b12 union2/b13 union2/b14 union2/b15                 # sim/experiment/batches_union2/ (seeds 12-15)
)

design_and_simulate() {            # ON PASTEUR. One call = one batch of ~960 events.
  # sim/run_batch.sh: select_events -> background genes -> build_experiment ->
  # order_fasta -> run_polyester.R -> align_reads.sh (STAR) -> index ->
  # dsignal.extract -> psi_recovery_check. Every step skips itself if its output
  # exists, so a dead batch is resumed by re-running the same command.
  #
  # RULES FROM crash_pasteur.md -- the machine is shared; a 32-way render once took it
  # to load 230 and locked everyone out of new logins:
  #   * wrap in limitram, cap every thread pool, nice -n 19
  #   * cat /proc/loadavg before launching and a minute after; throttle past ~50
  #   * never `pkill -f <pattern>` that also matches the ssh command line carrying it
  local tag=$1 seed=$2; shift 2     # remaining args: design.tsv of every earlier batch
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  ( cd "$PASTEUR_ROOT" && cat /proc/loadavg &&
    THREADS=8 PSI_DIST=uniform nohup limitram 32G nice -n 19 \
      sim/run_batch.sh "$tag" "$seed" "$@" > "logs/$tag.log" 2>&1 & )
  # Before spending polyester/STAR time, inspect the design: nn_hard rows have
  # psi1 == psi2 exactly, select_events.log reports 0 gene overlap with the
  # exclusion set, and every class reached its target (RI is the one that comes up short).
  #
  # Acceptance gates per batch, relative to already-accepted batches:
  #   python -m sim.psi_recovery_check          (run by run_batch.sh)
  #   python -m sim.width_leak_check
  #   python -m dsignal.audit_noise compare      <- the acceptance test after any re-simulation
  # Bring back only npz/, labels.tsv, design.tsv, psi_recovery.tsv, logs/ -- never BAMs/FASTA.
}

merge_pool() {                     # LOCAL. Records are hard-linked.
  python sim/merge_batches.py \
    --batch b1=sim/experiment/batches_geomwin/b1 --batch b2=sim/experiment/batches_geomwin/b2 \
    --batch b3=sim/experiment/batches_geomwin/b3 --batch b4=sim/experiment/batches_geomwin/b4 \
    --batch b5=sim/experiment/batches_geomwin/b5 \
    --batch b8=sim/experiment/batches_bag2/b8 --batch b9=sim/experiment/batches_bag2/b9 \
    --batch b10=sim/experiment/batches_bag2/b10 --batch b11=sim/experiment/batches_bag2/b11 \
    --batch b12=sim/experiment/batches_union2/b12 --batch b13=sim/experiment/batches_union2/b13 \
    --batch b14=sim/experiment/batches_union2/b14 --batch b15=sim/experiment/batches_union2/b15 \
    --out-npz sim/experiment/npz_union2 --out-labels sim/experiment/labels_union2.tsv \
    --subsample-events 4800 --subsample-events 8640 --subsample-events 12480
  # nn15 member: same records, NN thinned to 15% by whole genes
  python sim/merge_batches.py --from-labels sim/experiment/labels_union2.n12480.tsv \
    --out-labels sim/experiment/labels_union2.n12480.tsv --nn-frac 0.15
  # -> sim/experiment/labels_union2.n12480.nn15.tsv
  # Invariant to assert: 12,480 npz records, 12,480 distinct genes.
}

train_members() {                  # LOCAL, GPU. ~7 min per member on an RTX 4080 (5 folds).
  # The headline recipe verbatim; --refit/--inner-val-frac keep the epoch choice off
  # the reported fold. Flags differ per member exactly as in runs/ens_union2_*.json.
  local T="$DSNET_PY -m dsignal.train --npz-dir sim/experiment/npz_union2 --feature-subset no-geometry --refit --inner-val-frac 0.30"
  local L=sim/experiment/labels_union2.n12480.tsv
  $T --labels $L --augment all --out-dir runs/scale_union2_n12480   --json-out runs/scale_union2_n12480.json
  $T --labels $L --augment all --feature-subset all \
     --out-dir runs/ens_union2_allfeat    --json-out runs/ens_union2_allfeat.json
  $T --labels $L --augment all --two-head \
     --out-dir runs/ens_union2_twohead    --json-out runs/ens_union2_twohead.json
  $T --labels $L --augment all --depth-norm scale \
     --out-dir runs/ens_union2_depthscale --json-out runs/ens_union2_depthscale.json
  $T --labels $L --augment flip condition_swap subsample \
     --out-dir runs/ens_union2_augsub     --json-out runs/ens_union2_augsub.json
  $T --labels sim/experiment/labels_union2.n12480.nn15.tsv --augment all \
     --out-dir runs/ens_union2_nn15       --json-out runs/ens_union2_nn15.json
  # (a later --feature-subset on the command line overrides the one inside $T)
  # Ensemble spec (equal weight, mean_prob): runs/ensembles/e4_union2.json
}

score_dev() {                      # LOCAL. DEV HALVES ONLY. Test halves are spent.
  $DSNET_PY -m dsignal.score_benchmark --ensemble runs/ensembles/e4_union2.json \
    --half dev --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
    --csv benchmark/A3.csv benchmark/A5.csv benchmark/ES.csv benchmark/IR.csv
  $DSNET_PY -m dsignal.score_benchmark --ensemble runs/ensembles/e4_union2.json \
    --half dev --split benchmark_new/split.tsv --tta-condition-swap \
    --target-prior nn-weight --nn-weight 0.1 \
    --csv benchmark_new/A3.csv benchmark_new/A5.csv benchmark_new/ES.csv benchmark_new/IR.csv benchmark_new/NN.csv
  # paired test between two runs' per-event TSVs:
  #   python -m dsignal.benchmark_split compare A.tsv B.tsv --half dev
  # Reference to reproduce: runs/bnew_dev_e4_union2.tsv (142/242 collapsed, 58.7%).
}

case "${1:-}" in
  design|simulate) shift; design_and_simulate "$@";;
  merge)           merge_pool;;
  train)           train_members;;
  score)           score_dev;;
  *) sed -n 2,10p "$0"; exit 2;;
esac
