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

Written 2026-09-14, after all nine arm/seed runs and the mechanical gate.
**Verdict: all three arms FAIL.** No arm satisfies gate rules 1–3
simultaneously, so LOCO (rule 4) was not run for any arm, per "run only for
arms that satisfy 1–3."

### Pool construction: one mechanical finding

The pre-registered gene-disjointness assertion (real rows' `gene_id` vs
`ptbp1_arena/consensus.tsv` direct + the cumulative
`exclude_genes_u2af2.tsv`/`exclude_genes_nn_u2af2.tsv` exclusion sets, which
are supersets of `exclude_genes_stage16.tsv`/`exclude_genes_prpf4.tsv`) failed
on the first pass: **11 of 3,305 rows (0.33%)** leaked gene overlap that the
per-cohort exclusion builds missed — 4 rows overlapped `ptbp1_arena`'s own
gene set directly, the other 7 were genes excluded from PRPF4/U2AF2's *own*
candidate pools for clustering-near-curated-region reasons that AQR's
older (Stage 14/16) exclusion build never re-checked against. All 11 were
real NN or positive rows (4 `real_nn_aqr`, 1 `real_pos_prpf4`, 1
`real_pos_u2af2`, 5 `real_nn_u2af2`), dropped from both pools before any
training. Final pools: `labels_real3.tsv` **3,294** rows (A3 207, A5 133, CSE
1,299, RI 263, NN 1,392), `labels_union2_real3.tsv` **15,774** rows (real
share 20.9%, unchanged). Sim rows in `npz_union2` are *not* checked for
disjointness from `ptbp1_arena` — that was never true of the base pool and is
the same pre-existing condition the `union2_base` parents already carry into
their own arena scores below. Log: `runs/stage20_3_gene_disjoint.log`.

### Baselines (parents, zero-shot on `ptbp1_arena`; free, not previously measured there)

| seed | 5-class | collapsed | A3 | A5 | CSE | NN | RI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1077/2422 = 44.5% | 1091/2422 = 45.0% | 67/152 | 53/113 | 645/1073 | 94/800 | 218/284 |
| 1 | 1068/2422 = 44.1% | 1085/2422 = 44.8% | 76/152 | 55/113 | 636/1073 | 104/800 | 197/284 |
| 2 | 1120/2422 = 46.2% | 1136/2422 = 46.9% | 71/152 | 52/113 | 656/1073 | 120/800 | 221/284 |

`benchmark_new` dev and `benchmark/` dev baselines are the pre-registered
128–134/242 and 56–59/70 from `plans/signal-native-v2-resume-10.md`
(unchanged, not re-scored).

### Arm results, all three evals, all three seeds

Collapsed accuracy; `bnew`/`old` are arm − parent deltas (rule 2's arithmetic
check), `arena McN p` is the aggregate-collapsed McNemar p-value (not the
per-class one used for rule 1/3).

| arm | seed | bnew dev | Δbnew | old dev | Δold | arena | arena McN p |
|---|---|---:|---:|---:|---:|---:|---:|
| B (head-only) | 0 | 126/242 | −4 | 55/70 | −1 | 1140/2422 = 47.1% | 0.0047 |
| B | 1 | 125/242 | −3 | 61/70 | +5 | 1122/2422 = 46.3% | 0.0334 |
| B | 2 | 129/242 | −5 | 60/70 | +1 | 1164/2422 = 48.1% | 0.1189 |
| C (full, low LR) | 0 | 134/242 | +4 | 64/70 | +8 | 1396/2422 = 57.6% | 0.0000 |
| C | 1 | 135/242 | +7 | 65/70 | +9 | 1390/2422 = 57.4% | 0.0000 |
| C | 2 | 135/242 | +1 | 64/70 | +5 | 1392/2422 = 57.5% | 0.0000 |
| A (mixed, GRL) | 0 | 151/242 | +21 | 65/70 | +9 | 1568/2422 = 64.7% | 0.0000 |
| A | 1 | 147/242 | +19 | 64/70 | +8 | 1582/2422 = 65.3% | 0.0000 |
| A | 2 | 149/242 | +15 | 67/70 | +8 | 1590/2422 = 65.6% | 0.0000 |

Every arm clears rule 2 (no beyond-noise regression: bnew ≥ −10, old ≥ −5,
every seed) — C and A clear it by a wide, positive margin on both dev sets.
This is the part of the result that looks like a win, and it is why rules 1
and 3 (which is what the pooled number hides, per the Stage 16 shape) carry
the gate.

Sensitivity re-score of B/C with `--train-labels labels_real3.tsv` (their own
fine-tune set's prior, not the deployment prior; reported, not a gate input):
B bnew 103/103/97 of 242, old 56/59/57 of 70; C bnew 116/117/114 of 242, old
61/62/60 of 70 — both lower than the `labels_union2.n12480.tsv`-scored numbers
above in every case, i.e. the deployment prior correction is doing real work
and is the right one to gate on, as pre-registered.

### Rule 1 — arena NN (the pre-registered decisive condition)

Per-class McNemar on the 800 arena NN rows, arm vs seed-matched parent:

| arm | seed | parent NN recall | arm NN recall | direction | McNemar p |
|---|---|---:|---:|---|---:|
| B | 0 | 94/800 = 11.8% | 15/800 = 1.9% | **arm behind** | 0.0000 |
| B | 1 | 104/800 = 13.0% | 13/800 = 1.6% | **arm behind** | 0.0000 |
| B | 2 | 120/800 = 15.0% | 19/800 = 2.4% | **arm behind** | 0.0000 |
| C | 0 | 94/800 = 11.8% | 62/800 = 7.8% | **arm behind** | 0.0003 |
| C | 1 | 104/800 = 13.0% | 72/800 = 9.0% | **arm behind** | 0.0004 |
| C | 2 | 120/800 = 15.0% | 75/800 = 9.4% | **arm behind** | 0.0000 |
| A | 0 | 94/800 = 11.8% | 178/800 = 22.2% | arm ahead | 0.0000 |
| A | 1 | 104/800 = 13.0% | 182/800 = 22.8% | arm ahead | 0.0000 |
| A | 2 | 120/800 = 15.0% | 192/800 = 24.0% | arm ahead | 0.0000 |

**B and C move arena NN, but backwards** — real-only fine-tuning (head-only or
full) makes the model call NN *less* often on the confirmatory arena, not
more, despite raising pooled accuracy on every other eval. Both hard-fail
rule 1 (0/3 seeds with the arm ahead; the requirement is ≥2/3). **A passes
rule 1 cleanly**: ahead in 3/3 seeds, all p < 0.0001, median NN recall 22.8%
(A) vs 13.0% (parent).

### Rule 3 — per-class recall regression

**Arena**: for A, every one of the five classes *improves* in every seed
(McNemar p < 0.0001 each), so the "no class regresses" clause is satisfied
trivially — there is no regression to catch. (B and C already fail on rule 1
and were not further pursued against this clause, though for the record B
regresses RI in all 3 seeds — 218→193, 197→178, 221→200, p ≤ 0.0002 each —
and C's own NN regression is exactly rule 1's finding.)

**`benchmark_new` dev, count rule (±3 events, all three seeds), arm A vs parent:**

| class | n | parent (s0/s1/s2) | A (s0/s1/s2) | Δ (s0/s1/s2) |
|---|---:|---|---|---|
| A3 | 16 | 12/14/13 | 15/15/15 | +3/+1/+2 |
| A5 | 12 | 7/7/8 | 11/11/11 | +4/+4/+3 |
| CSE | 107 | 86/83/87 | 99/98/99 | +13/+15/+12 |
| **NN** | **79** | **7/8/8** | **4/1/3** | **−3/−7/−5** |
| RI | 28 | 18/16/18 | 22/22/21 | +4/+6/+3 |

Four of five classes gain by ≥2 events in every seed. **NN drops by ≥3 events
in all three seeds** — the literal condition rule 3 states ("applied to all
five classes"). By that reading, **arm A fails rule 3.**

**This is the pivotal call, and it is not clean — flagged rather than
resolved quietly.** The plan's closing paragraph says curated NN recall on
`benchmark_new` dev "is reported for every run and is not a gate input,"
citing Stage 18 rule 3 (curated NN is not expressible from current features,
AUC 0.58). Read in isolation, that sentence would exempt this exact NN row
from rule 3 and flip arm A to a **PASS** (all four conditions clear, since
rule 3's arena clause and rule 2 are already satisfied and rule 1 is
satisfied outright). Read against rule 3's own text, which explicitly scopes
the count rule to "all five classes" as the mechanism for catching exactly
the Stage 16 failure shape (pooled-accuracy win hiding a per-class
regression), the more specific and more on-point text governs, and the
mechanical answer is **FAIL**. Both readings are applied and reported rather
than one being picked:

- **Under the literal all-five-classes reading (adopted as the gate's
  outcome): arm A fails.**
- **Under the "curated NN is not a gate input" reading: arm A would pass**,
  and would proceed to 20.4 as the ensemble candidate.

Given the ambiguity was not resolved before running (a pre-registration
failure worth naming, not just the arms'), the conservative reading is
adopted — a fine-tune that improves arena NN by design (GRL on real-vs-sim)
while curated NN quietly drops in every seed is exactly the shape Stage 16
warned about, even if the specific mechanism (features cannot express
curated NN, so *any* move there is close to noise) argues the drop may not
mean what rule 3 assumes it means. This tension is real and unresolved by
this stage; a follow-up that either re-registers rule 3 to explicitly exempt
curated NN, or measures whether arm A's curated-NN drop is itself within the
noise Stage 18 already characterized, would settle it cheaply without a new
training run (the OOF/dev predictions already exist).

### Within-real OOF (free, decides nothing, 5-fold pooled over the 3 seeds' own folds)

Accuracy on each arm's own real-only held-out folds, real rows only, macro
over the 3 seeds:

| kind | A | B | C |
|---|---:|---:|---:|
| real_nn_aqr | 909/1410 = 64.5% | 635/1410 = 45.0% | 847/1410 = 60.1% |
| real_nn_prpf4 | 988/1341 = 73.7% | 724/1341 = 54.0% | 965/1341 = 72.0% |
| real_nn_u2af2 | 859/1425 = 60.3% | 599/1425 = 42.0% | 746/1425 = 52.4% |
| real_pos_aqr | 1152/1530 = 75.3% | 871/1530 = 56.9% | 1061/1530 = 69.3% |
| real_pos_prpf4 | 642/852 = 75.4% | 561/852 = 65.8% | 603/852 = 70.8% |
| real_pos_u2af2 | 2660/3324 = 80.0% | 2207/3324 = 66.4% | 2553/3324 = 76.8% |

A > C > B on every cohort — full-context gradient flow (A trains from
scratch with real rows in the mix and domain-adversarial pressure; C
fine-tunes the whole encoder; B only trains a head on top of frozen sim
features) buys within-real fit, and the ranking matches the gate: A's
zero-shot numbers are the strongest, and B's head-only fine-tune is worst
everywhere, consistent with a frozen encoder being unable to relocate its
decision boundary far enough to help arena NN at all.

### Gate verdict

| arm | rule 1 (arena NN) | rule 2 (no regression) | rule 3 (per-class) | PASS? |
|---|---|---|---|---|
| B | FAIL (0/3 seeds ahead) | pass | not evaluated further | **FAIL** |
| C | FAIL (0/3 seeds ahead) | pass | not evaluated further | **FAIL** |
| A | pass (3/3 seeds ahead) | pass | FAIL (bnew NN, literal reading) | **FAIL** |

**All three arms fail.** Per the plan's pre-registered consequence: real
supervision on the current features is closed for the second time, with a
stronger negative than Stage 14–17 (three cohorts, balanced real cells, three
fine-tuning regimes, one of which — A — gets everything else right and is
blocked on a single ambiguous sub-clause). LOCO was not run for any arm.
Stage 22b (self-supervised pretraining on real windows) is next, per Stage 18
rule 1. Stage 21 (selective calling) is independent and runs regardless. The
one open item this stage leaves behind: resolve the rule-3/curated-NN
ambiguity above — cheaply, from data already on disk — before treating "all
arms fail" as final.
