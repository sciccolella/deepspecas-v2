# Stage 20.3 — real supervision done right: pre-registration

Written 2026-09-14, **before any arm has been trained or scored.** Parent
plan: `plans/signal-native-v2-stage18-22-accuracy-plan.md` §20.3–20.4.
State-of-play: `plans/signal-native-v2-resume-10.md`. Everything in
"Fixed before running" is frozen at the commit that adds this file; the
Results section is appended afterwards and nothing above it is edited
except to add retractions next to the numbers they retract.

## Why this stage exists (one paragraph)

Stages 14/16/17 mixed *one* real cohort (AQR) into 12.5k simulated rows,
unweighted, majority-CSE, and every attempt failed its gate on the same
shape: pooled accuracy regressed beyond seed noise on `benchmark_new` dev
while arena NN recall rose. Stage 18 then showed the sim→real gap is
*diffuse* (real-vs-sim AUC 0.999, still 0.987 after removing the 30 most
informative feature families) and that **curated NN is not expressible
from the current features** (NN-vs-positive AUC 0.58 with 3.2k real
labels) whereas **arena NN is** (0.88 within-arena, 0.83–0.85 across
arenas). Stage 20.1 built two more gene-disjoint real cohorts (PRPF4/HepG2,
U2AF2/K562) and Stage 20.2 gave `train.py` a fine-tuning path with
bit-identity checks. This stage runs the three fine-tuning regimes the
parent plan pre-specified, with the three-cohort pool, and gates them on
what Stage 18 says can move: **arena-type NN and positives**, with curated
NN reported but not expected to move.

## Fixed before running

### Inputs

| item | value |
|---|---|
| parents (seed-matched) | seed 0 `runs/scale_union2_n12480`, seed 1 `runs/union2_base_s1`, seed 2 `runs/union2_base_s2` — all single-model recipe on `npz_union2` / `labels_union2.n12480.tsv`, `--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30` |
| real training pool | `sim/experiment/real_{pos,nn}_{aqr,prpf4,u2af2}` — 510+474, 285+447, 1,109+480 = **3,305 rows**, six `kind`s, all `dsignal.verify`-clean, gene-disjoint from `ptbp1_arena`, `benchmark/`, `benchmark_new` (Stage 14/16 exclusion set + Stage 20.1 `exclude_genes_{prpf4,nn_prpf4,u2af2,nn_u2af2}.tsv`) |
| **never in any training pool** | `real_{pos,nn}_ptbp1` (92 genes shared with `benchmark_new` dev), `runs/real_dev_npz`, `runs/real_tra2_npz` |
| primary evaluation | `benchmark_new` dev (242; 16 A3 / 12 A5 / 107 CSE / 28 RI / 79 NN) |
| regression guard | `benchmark/` dev (70; 14 A3 / 15 A5 / 33 CSE / 8 RI) |
| confirmatory | `ptbp1_arena` all 2,422 candidates (152 A3 / 113 A5 / 1,073 ES / 284 IR / 800 NN) |
| test | `benchmark_new` test (243) **untouched**; `benchmark/` test spent (Stage 6) |
| seeds | train seeds 0, 1, 2; the arm's seed also selects its parent |

Real arena labels are ~67% pure (pair 1's measured purity; pairs 2–4
unmeasured, assumed similar). Every number below is reported with that
caveat.

### Pool construction (mechanical, one-off, committed)

Same pattern as Stage 16 (`f1ad3233`): an uncommitted symlink shim
`sim/experiment/_shim_union2/{npz -> ../npz_union2, labels.tsv ->
../labels_union2.n12480.tsv}`, then `sim/merge_batches.py`:

```bash
# real-only pool (arms B, C)
python -m sim.merge_batches \
  --batch real_pos_aqr=sim/experiment/real_pos_aqr   --batch real_nn_aqr=sim/experiment/real_nn_aqr \
  --batch real_pos_prpf4=sim/experiment/real_pos_prpf4 --batch real_nn_prpf4=sim/experiment/real_nn_prpf4 \
  --batch real_pos_u2af2=sim/experiment/real_pos_u2af2 --batch real_nn_u2af2=sim/experiment/real_nn_u2af2 \
  --out-npz sim/experiment/npz_real3 --out-labels sim/experiment/labels_real3.tsv
# mixed pool (arm A) = the same six batches + the union2 n12480 shim
python -m sim.merge_batches --batch union2=sim/experiment/_shim_union2 <six real batches as above> \
  --out-npz sim/experiment/npz_union2_real3 --out-labels sim/experiment/labels_union2_real3.tsv
```

Expected: `labels_real3.tsv` 3,305 rows; `labels_union2_real3.tsv`
15,785 rows, real share 20.9%. `kind` is carried through unchanged (it is
what `--balance-real-cells` and `domain_probe.cohort_of` key on). The
LOCO pools (§ below) are the same command with one cohort's two batches
dropped: `labels_real3_no_{aqr,prpf4,u2af2}.tsv`. **Before training
anything**: assert the gene set of every pool is disjoint from the ENSG
ids in `ptbp1_arena/consensus.tsv`, `benchmark_new` dev and `benchmark/`
(the Stage 20.1 exclusion already guarantees it; the assertion is the
visible check, same style as `sim/select_events.py`).

### The three arms (exact flags)

Common to all: `--npz-dir <pool> --labels <pool labels> --augment all
--feature-subset no-geometry --seed S --out-dir runs/stage20_<arm>_s<S>
--json-out runs/stage20_<arm>_s<S>.json --dump-oof
runs/stage20_<arm>_s<S>.oof.tsv`. Folds: 5, `GroupKFold` by gene as
always (real rows group by their ENSG prefix in `event_id`).

- **A — mixed, cell-balanced, ramped GRL** (trained from scratch; the
  parent is *not* loaded, the seed just matches):
  `--npz-dir sim/experiment/npz_union2_real3 --labels
  sim/experiment/labels_union2_real3.tsv --balance-real-cells 0.10
  --class-weights none --domain-lambda 1.0 --grl-schedule ramp --select-on
  inner --inner-val-frac 0.30 --refit`
- **B — head-only fine-tune** of the seed-matched parent:
  `--init-from <parent> --freeze encoder --select-on none --epochs 20 --lr
  1e-4 --npz-dir sim/experiment/npz_real3 --labels
  sim/experiment/labels_real3.tsv --balance-real-cells 0 --class-weights
  none`
- **C — full fine-tune, low LR** of the seed-matched parent:
  `--init-from <parent> --freeze none --select-on none --epochs 10 --lr
  3e-5 --warmup-epochs 2 --avg-last-k 3 --npz-dir sim/experiment/npz_real3
  --labels sim/experiment/labels_real3.tsv --balance-real-cells 0
  --class-weights none`

No other hyper-parameter is tuned. If an arm is unstable (loss NaN, or an
OOF macro-F1 on its own held-out real folds below the parent's zero-shot
number on the same rows) that is reported as the arm's result, not fixed
by sweeping.

`--balance-real-cells 0` in B/C means the twelve real `(kind, label)`
cells carry equal total weight (mean weight 1); in A the real rows sum to
0.10 × the sim total. `--class-weights none` in all three avoids
balancing twice.

### Scoring protocol (identical for every arm and seed)

Three `score_benchmark` calls per run, `--net runs/stage20_<arm>_s<S>/fold*.pt
--tta-condition-swap --target-prior nn-weight --nn-weight 0.1`:

1. `--half dev --split benchmark_new/split.tsv --csv benchmark_new/{A3,A5,ES,IR,NN}.csv --out runs/bnew_dev_stage20_<arm>_s<S>.tsv`
2. `--half dev --out runs/benchmark_events_stage20_<arm>_s<S>_dev.tsv` (default `benchmark/`)
3. `--csv ptbp1_arena/candidates/{A3,A5,ES,IR,NN}.csv --half both --out runs/arena_stage20_<arm>_s<S>.tsv`

`--train-labels` is **explicit and pre-registered**: arm A passes its own
`labels_union2_real3.tsv` (the prior it was trained from, as Stage 14/16
did with their mixed files); arms B and C pass the parent's
`labels_union2.n12480.tsv` — the checkpoint's stored `train_class_counts`
is the 3.3k fine-tune set's, which is not the deployment prior. A
sensitivity re-score of B/C with `labels_real3.tsv` is reported in the
Results table but is not a gate input.

Baselines needed that do not exist yet (free, scored first): the three
parents on `ptbp1_arena` with the same protocol →
`runs/arena_union2_base_s{0,1,2}.tsv`. `runs/bnew_dev_union2_base_s{0,1,2}.tsv`
and `runs/benchmark_events_union2_base_s{0,1,2}_dev.tsv` exist.

Paired comparisons use `dsignal.benchmark_split compare A.tsv B.tsv
--half {dev,both} --by-class`, always **seed-matched** (arm seed S vs
parent seed S).

### Baselines (seed-matched single-model parents; e4 ensemble for reference)

| set | metric | s0 | s1 | s2 | e4 (16-member ref.) |
|---|---|---:|---:|---:|---:|
| `benchmark_new` dev | collapsed | 130/242 | 128/242 | 134/242 | 142/242 |
| | A3 / A5 / CSE / RI / NN recall | 12/16, 7/12, 86/107, 18/28, 7/79 | 12/16, 7/12, 83/107, 16/28, 8/79 | 13/16, 8/12, 87/107, 18/28, 8/79 | 14/16, 9/12, 89/107, 19/28, 11/79 |
| `benchmark/` dev | collapsed | 56/70 | 56/70 | 59/70 | 61/70 |
| `ptbp1_arena` | collapsed / NN recall | *to score* | *to score* | *to score* | 1175/2422; A3 90/152, A5 68/113, CSE 669/1073, RI 213/284, NN 135/800 |

Noise ceilings (branch-measured seed spread): **10/242** on `benchmark_new`
dev, **5/70** on `benchmark/` dev. On the arena (n=800 NN, n=2,422 total)
the paired McNemar carries the decision, not a fixed event count.

### Go / no-go gate (mechanical; applied per arm after all three seeds)

An arm **PASSES** only if all of the following hold. Anything else is a
FAIL, and a FAIL is recorded with the same detail as a pass.

1. **Arena NN moves** (the NN condition, per Stage 18 rule 3): seed-matched
   McNemar on the 800 `ptbp1_arena` NN rows, arm vs parent, p < 0.05 with
   the arm ahead, in **≥ 2/3 seeds**, and the median-over-seeds NN recall is
   above the parents' median.
2. **No beyond-noise regression** on `benchmark_new` dev collapsed
   (arm − parent ≥ −10 in every seed) **and** on `benchmark/` dev collapsed
   (≥ −5 in every seed). A regression at the ceiling in all three seeds is
   also a FAIL (three identical −10s are not noise).
3. **Per-class recall** (the Stage 16 shape is a FAIL regardless of pooled
   accuracy): on the arena, no class whose recall drops with McNemar
   p < 0.05 in ≥ 2/3 seeds; on `benchmark_new` dev, no class whose recall
   drops by ≥ 3 events in all three seeds (A5 n=12 and RI n=28 are too
   small to test, so this is a count rule, applied to all five classes).
4. **Leave-one-cohort-out** (run only for arms that satisfy 1–3, all three
   seeds, three leave-outs = 9 extra runs per arm, same flags with
   `labels_real3_no_<cohort>.tsv`): in **each** of the three leave-outs,
   the arena NN gain over the seed-matched parent keeps the same sign in
   ≥ 2/3 seeds. The held-out cohort's rows are also scored zero-shot by
   the leave-out model (they are gene-disjoint from it) and reported —
   that number is the only *real-to-real* transfer figure this stage
   produces and is informative even if the gate fails.

Curated NN recall on `benchmark_new` dev is **reported for every run and
is not a gate input** — Stage 18 rule 3 says the features cannot express
it, and a movement there would be surprising enough to need its own
follow-up, not a pass.

### What is decided by the outcome

- **Any arm passes** → 20.4: apply that arm's flags to all six e4 members
  (seed-matched parents are the members themselves) →
  `runs/ensembles/e6_realft.json`, scored on dev only against
  `runs/ensembles/e4_union2.json` (142/242, 61/70, 1175/2422). No test
  spend from this stage. If two arms pass, the one with the larger median
  arena NN gain goes to the ensemble; the other is recorded.
- **All arms fail** → real supervision on the current features is closed
  for the second time with a stronger negative than Stage 14–17 (three
  cohorts, balanced cells, three regimes), and Stage 22b (self-supervised
  pretraining on real windows) proceeds as the next real-data device, per
  Stage 18 rule 1. Stage 21 (selective calling, v1+v2 rule) is independent
  and runs regardless.
- The arms' own 5-fold OOF on the real pool (`--dump-oof`) is a free
  within-real number; it is reported per cohort and per class but decides
  nothing.

### Cost and order

Baselines (3 arena scores, ~3 min each) → B × 3 seeds (minutes each, head
only) → C × 3 seeds (~10 min each) → A × 3 seeds (~1 h each, from scratch
on 15.8k rows) → scoring → gate → LOCO for passing arms. One queue script
`runs/stage20_3_queue.sh` modelled on `runs/stage18_5_queue.sh`, logging to
`runs/stage20_3_queue.log`, ending in `QUEUE_DONE`. Everything is local
(`DSnet`, RTX 4080); nothing runs on pasteur. Checkpoints stay gitignored;
JSON, logs, per-event TSVs and OOF TSVs are committed.

### Not fixed here (and deliberately so)

- `nn_weight` stays 0.1 (Stage 21.3 owns any re-sweep, after this stage).
- No epoch, LR or `FRAC` sweep. If B/C look under- or over-trained on
  their own OOF curves, that goes in the write-up as a follow-up, not into
  this stage's numbers.
- No new manual curation, no new arenas.

---

## Results

*(appended after the runs; nothing above this line changes)*
