# signal-native-v2 — START HERE (2026-08-19)

Resume document. Branch `signal-native-v2`, clean tree (after this commit),
**nothing pushed** (per repo convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-3.md` as the entry point.
`plans/signal-native-v2-ensemble.md` (the plan) and
`plans/signal-native-v2-ensemble-results.md` (the full stage-by-stage record)
are both **fully implemented** — read the results doc for detail this file
compresses.

---

## The one thing to know

**The ensemble closed part of the gap; it did not close it. The mechanism
that actually moved the number was more training data, not more models.**

resume-3 closed with "the mechanism the evidence supports is ensembling."
That was tested properly this session — six pre-registered ensembles, a
combination-rule sweep, a real bagging-vs-scale control — and the honest
answer is smaller than hoped: ensembling is worth **+1–2 events, free**, but
the single most effective thing tried was training one model on twice the
data (`npz_geomwin` + `npz_bag2` merged, `npz_union`, 8,640 events), which
beat every ensemble in this plan.

## The corrected v1 target — now final, not provisional

`plans/signal-native-v2-ensemble.md`'s own opening correction (60/70 dev,
62/69 test) has itself been superseded by Stage 0.1's cleaner `(csv, region)`
keying, consistent with every other number this branch produces:

| model | dev | test | all 140 |
|---|---|---|---|
| **v1 six-net ensemble** | **60/70 = 85.7%** | **63/70 = 90.0%** | 123/140 = 87.9% |
| v1 `squishstack` single net | 56/70 = 80.0% | 50/70 = 71.4% | 106/140 = 75.7% |
| **v2 headline (E3, image-free ensemble)** | **54/70 = 77.1%** | **60/70 = 85.7%** | — |
| v2 single-run (no ensemble, prior headline) | 53/70 = 75.7% | 58/70 = 82.9% | — |

Use **60/70 dev, 63/70 test, 123/140 all** for v1 from here on. `CLAUDE.md`
has been updated to match; the old 62/69 / 122/139 figures stay in
`plans/signal-native-v2-ensemble.md` and this file, retracted in place, per
convention.

**Result on the pre-registered test criterion:** neither parity (v2's test
accuracy ≥ v1's) nor a win (McNemar p<0.05) was reached. 60/70 vs 63/70,
discordant 4 v2-only / 7 v1-only, p=0.549. **Test has been spent** — any
further v1-vs-v2 comparison on this benchmark needs a fresh held-out half
(re-split, or a new curated set).

## The headline configuration

**60/70 = 85.7% collapsed on the held-out test half**, scored once, E3
selected on dev (which ties E2 at 54/70 = 77.1%, and is the fuller,
pre-registered superset):

```bash
python -m dsignal.score_benchmark --ensemble runs/ensembles/e3_bagged.json \
  --half test --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
  --out runs/benchmark_events_e3_test.tsv
```

`runs/ensembles/e3_bagged.json` lists all 16 member run directories (equal
weight, `mean_prob` — Stage 4 showed the rule/weighting choice does not
matter at this member count). Every member's checkpoints already exist under
`runs/ens_*`; nothing needs retraining to reproduce this.

**The best single number in the whole plan, not currently in any shipped
ensemble:** `runs/ens_bag_union` alone (one 5-fold run on `npz_union`) scores
**55/70 = 78.6% dev** — better than E1, E2, E3, and the bagged-pair-only
ensemble. It was a Stage 3 diagnostic run, not a Stage 6 candidate (see the
honesty ledger in the results doc for why), so it was not spent against test.
**It is the clearest next lever**, not another ensemble axis.

## What is now settled, and what it means for the next session

**Ensembling has a real but small, saturating effect.** View diversity
(Stage 2) was worth +1 dev event for free. Data bagging (Stage 3, disjoint
gene sets, separate models, averaged) added **nothing** once view diversity
was already present — E3 == E2, 0 discordant events. The combination rule
and per-run F1-weighting (Stage 4) moved **zero** predictions across 6
pre-registered cells. Sixteen members of broadly similar quality (sim
macro-F1 0.851–0.895) saturate an image-free ensemble's ceiling quickly; this
line is not worth pushing further with more member variety.

**Scale is the lever that actually worked.** `npz_union` (union of
`npz_geomwin` + the four new Stage 3 batches, 8,640 events, one model, no
ensembling) beat every ensemble in this plan by training on twice the data.
This was framed going in as a control to separate "bagging helped" from
"more data helped" — it answered cleanly, and scale won.

**The image view still holds real complementary signal.** Stage 5's ceiling
check: averaging E3 with a *single* v1 crossover net (already-computed
predictions, no new inference) reaches 87.1% dev — above the v1 six-net
ensemble's own 60/70=85.7%. This is not shippable (the deliverable is
image-free), but it is evidence the remaining gap is not "the signal-native
representation is missing information the image view also lacks."

**What this suggests for a next plan, if there is one:**

* Scale `npz_union`-style data further — more gene-disjoint batches, one
  model, not more ensemble members. This is the one lever this session found
  that is still unexplored past a single doubling.
* Do **not** spend more effort on ensemble breadth (Stage 4's null result) or
  on data bagging specifically (Stage 3's control resolved it).
* If the image-view signal (Stage 5) is ever worth chasing for real, it needs
  a plan of its own — distilling whatever the image representation captures
  that junction/coverage features do not, not literally shipping a hybrid.
* Test is spent. Any further v1-vs-v2 milestone needs either a fresh
  benchmark split or accepting that all remaining tuning happens on dev only,
  reported honestly as such.

## Open honesty caveats (carried forward, still open)

1. **The NN operating point — settled**, unchanged this session
   (`nn_weight=0.1`, pre-registered in `dsignal/prior_sweep.py`'s docstring).
2. **`DPSI_BETA`/`PSI_LEVEL_BETA` are still fitted on all 140 benchmark
   events.** Stage 3's new batches used `PSI_DIST=uniform` throughout, so
   they stay clean; this caveat is unchanged from resume-3.
3. **`union-alone`'s 78.6% dev has had no exposure to test**, unlike E1/E2/E3.
   If it is trained again or scaled up, it becomes eligible for a Stage-6-style
   one-shot test spend the same as any other candidate — but that spend has
   not happened yet, and this document is not claiming it as a result.

## What is new in the code

| file | what |
|---|---|
| `dsignal/v1_reference.py` | v1 prediction CSVs → score_benchmark-format per-event TSVs, `(csv, region)`-keyed |
| `dsignal/predict.py` | `Predictor` validates `feature_config` across checkpoints (H1); `build_prior_correction` builds a per-fold correction from each checkpoint's own `train_class_counts` when no `--train-labels` given (H2); `combine()` accepts either a shared or a per-fold `PriorCorrection` |
| `dsignal/ensemble.py` | **new** — `EnsembleSpec`, `MultiRunPredictor`, `combine()` (`mean_prob`/`mean_logprob`/`sum_logit`), the posterior cache (`--cache-out`/`--cache-in`) |
| `dsignal/score_benchmark.py` | `--ensemble <spec.json>` (mutually exclusive with `--net`), `--rule` |
| `dsignal/train.py` | `--class-weight-boost CLASS FACTOR` (repeatable), used for `ens_cse` |
| `sim/merge_batches.py` | unchanged, driven for `npz_bag2` and `npz_union` |

## Artifacts

| path | what |
|---|---|
| `plans/signal-native-v2-ensemble-results.md` | the full stage-by-stage record |
| `runs/ensembles/{single_iv30,e1_seeds,e2_views,e3_bagged,e3_f1weighted,bag_pair_only}.json` | every ensemble spec, committed and reproducible |
| `runs/ens_seed{1,2,3,4}/`, `runs/ens_avglastk5/`, `runs/ens_iv45/` | Stage 1 members |
| `runs/ens_{allfeat,twohead,depthscale,augsub,nn15,cse}/` | Stage 2 members |
| `runs/ens_bag_{geomwin20,geomwin21,bag2_0,bag2_1,union}/` | Stage 3 members |
| `sim/experiment/{npz_bag2,npz_union}/`, `labels_{bag2,union}.tsv`, `batches_bag2/{b8,b9,b10,b11}/` | the new gene-disjoint data |
| `runs/benchmark_events_{e1,e2,e3}_dev.tsv`, `runs/benchmark_events_e3_test.tsv` | per-event ensemble scoring |
| `runs/benchmark_events_v1_{ensemble,squishstack}.tsv` | the corrected v1 references |
| `runs/ens_cache_dev.npz` | Stage 4's cached posteriors (16 members × dev) |
| `runs/stage5_hybrid_script.py`, `runs/stage5_hybrid_ceiling.log` | the ceiling measurement |

## Running jobs on pasteur — read `CLAUDE.md` first

Unchanged from resume-3: `limitram`, all four thread env vars at 1, `nice
-19`, hold the ssh session (Linger=no kills backgrounded `&` jobs — a `run
in background` tool call that keeps the local ssh process alive works;
`nohup cmd &` inside the ssh command does not reliably survive). This session
saw pasteur's own baseline load sit at ~20/64 from other users throughout,
with a transient spike to ~90 during Stage 3's later batches traced to
another user's job burst (confirmed via `ps aux` before continuing rather
than assumed) — our own footprint (`THREADS=8` vs the default 24) never
needed throttling.

## Conventions

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar; 4–6 events is noise at n=140,
  8–10 at n=70.
- Never delete a measured number from a docstring or plan doc — retractions
  go next to the number they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/rendering on
  pasteur.
