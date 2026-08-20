# signal-native-v2 — Stage 7: scale the training set (`npz_union2`)

*Plan written 2026-08-20. Successor to `plans/signal-native-v2-ensemble.md` /
`-ensemble-results.md`; entry point remains `plans/signal-native-v2-resume-4.md`.
Approved 2026-08-20; this is the canonical copy.*

---

## Context

The ensemble plan (Stages 0–6) ran to completion and answered its question in a
direction nobody expected. Ensembling works but saturates: view diversity was
worth **+1 dev event**, data bagging added **exactly nothing** on top of it
(E3 == E2, 0 discordant), and the combination-rule × weighting sweep moved
**zero predictions across all six pre-registered cells**. The one thing that
actually moved the number was **more data**: a single model on `npz_union`
(8,640 events, `npz_geomwin` + `npz_bag2`) scored **55/70 = 78.6% dev**,
beating the full 16-member E3 (54/70) and every other configuration in the
plan. That run was a Stage 3 diagnostic, never eligible for the test spend, and
resume-4 names it explicitly as "the clearest next lever".

This plan pulls that lever once more, and it is the **last time it can be
pulled cleanly**. Measured against the SUPPA catalogue:

| class | distinct genes | already consumed | unused |
|---|---|---|---|
| A3 | 9,013 | | 4,055 |
| A5 | 8,523 | | 3,708 |
| SE | 14,409 | | 8,121 |
| **RI** | **3,935** | | **908** |

9,024 genes are already spent across `batches/`, `batches_geomwin/`,
`batches_beta/`, `batches_bag2/`. At 180 RI events per batch under the
one-gene-per-event invariant, **RI supports only ~5 more batches, ever**. So:
four batches, `npz_union2` at 12,480 events (1.44× the current best), spend
~720 of the remaining 908 RI genes, keep ~190 in reserve.

**Intended outcome.** A three-point scale curve on gene-disjoint subsamples of
one pool, answering whether dev accuracy is still rising at 12.5k or has
flattened — and, if it is still rising, one final ensemble on top to see
whether scale and view diversity stack. **Everything is read off dev.** Test
was spent in Stage 6 (E3 60/70 vs v1 63/70, McNemar p=0.549) and this plan
makes **no v1-vs-v2 claim**; if the curve lands clearly above 78.6% dev, a
follow-up plan curates a fresh held-out half and spends it once.

## Decisions taken (user, this session)

1. **4 gene-disjoint batches**, invariant intact. No gene reuse.
2. **Dev-only reporting.** No test spend. Fresh-benchmark curation is a
   separate follow-up plan, gated on the result.
3. **Scale curve first, one ensemble second.** Do not confound the two.

---

## Stages

### Stage 7.0 — subsample utility (local, cheap model tier)

`sim/merge_batches.py` already has `_drop_genes_to(rows, keep_n, seed)`, which
drops whole genes so `GroupKFold` grouping is never split — exactly what a
scale curve needs. It is currently reachable only via `--nn-frac`.

Add `--subsample-events N` (repeatable) + reuse of the existing
`--from-labels` path, writing `labels_union2.n{N}.tsv`. No new logic: it is
`_drop_genes_to` over all rows plus the same `order`-preserving write that
`resample_to_nn_frac` already does. Refactor the shared tail into one helper
rather than copying it.

*Verify:* each output labels file has ≈N rows, its gene set is a subset of the
next larger one's (nesting matters — a non-nested curve confounds size with
gene identity), and class fractions stay within ~1pp of the parent.

**Model tier: cheap.** Mechanical, one file, existing helper.

### Stage 7.1 — four new batches on pasteur (heavy, mixed tier)

Batches `b12`–`b15`, seeds 12–15, `sim/run_batch.sh` unchanged, exactly the
Stage 3 settings (`PSI_DIST=uniform` — keeps the `DPSI_BETA` honesty caveat
clean), each passing `--exclude-genes` for **every** prior design.tsv in the
chain: `batches_geomwin/b{1..5}`, `batches/b{1..5}`, `batches_beta/b6beta`,
`batches_bag2/b{8..11}`, plus the earlier members of this batch chain.

Per `CLAUDE.md` and resume-4, non-negotiable:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
THREADS=8 nohup limitram 32G nice -n 19 sim/run_batch.sh b12 12 <prior design.tsv ...> > logs/b12.log 2>&1 &
```

`cat /proc/loadavg` before launching and a minute after; throttle past ~50.
Hold the ssh session alive (a backgrounded Bash tool call), Linger=no.

**Inspect the design before any polyester/STAR time is spent** — this is the
check that caught the null-dPSI double-draw bug: confirm `nn_hard` rows have
`psi1 == psi2` exactly (`NULL_DPSI_SIGMA=0`), confirm each
`logs/select_events.log` reports **0 gene overlap** with the exclusion set, and
confirm all four label classes actually reached 180 (**RI is the one that can
silently come up short as the pool thins — if any batch under-fills RI, stop
and report rather than proceeding**).

Bring back only `npz/`, `labels.tsv`, `design.tsv`, `psi_recovery.tsv` and
`logs/` — never BAMs or FASTA.

*Gates, per batch, before it is accepted:* `python -m sim.psi_recovery_check`
(run by `run_batch.sh`), `python -m sim.width_leak_check`, and
`python -m dsignal.audit_noise compare` — the acceptance test after any
re-simulation — each clean relative to the already-accepted `bag2` batches.

**Model tier: strong** for the exclusion-chain construction, design inspection
and gate interpretation; the launching itself is mechanical.

### Stage 7.2 — merge and build the curve (local)

```bash
python sim/merge_batches.py \
  --batch <all 9 existing geomwin+bag2 batches> \
  --batch b12=... --batch b13=... --batch b14=... --batch b15=... \
  --out-npz sim/experiment/npz_union2 \
  --out-labels sim/experiment/labels_union2.tsv
```

Records hardlink, so this costs nothing on top of the per-batch dirs.
Then the nested subsamples at **4,800 / 8,640 / 12,480**.

*Verify:* 12,480 npz records; `labels_union2.tsv` gene set == union of the 13
designs, size 12,480 distinct genes (the invariant, asserted not assumed).

**Model tier: cheap.**

### Stage 7.3 — the scale curve (local, `DSnet`)

Three runs, **identical config**, the headline recipe verbatim — only
`--labels` differs:

```bash
python -m dsignal.train \
  --npz-dir sim/experiment/npz_union2 --labels sim/experiment/labels_union2.n{4800,8640,12480}.tsv \
  --augment all --feature-subset no-geometry --refit --inner-val-frac 0.30 \
  --out-dir runs/scale_union2_n{N} --json-out runs/scale_union2_n{N}.json
```

Scored on **dev only**, the settled operating point:

```bash
python -m dsignal.score_benchmark --net runs/scale_union2_n{N}/fold*.pt \
  --half dev --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
  --train-labels sim/experiment/labels_union2.n{N}.tsv \
  --out runs/benchmark_events_scale_n{N}_dev.tsv
```

Report via the shared harness (`dsignal.baseline_gbm.report` /
`report_collapsed`), tabulate with `dsignal.tabulate_runs`, and run **paired
McNemar** on every adjacent pair via `dsignal.benchmark_split compare`. At
n=70 a real difference needs 8–10 events: expect most adjacent deltas to be
non-significant, and **report direction + magnitude honestly rather than
promoting a 2-event rise to a finding**.

Corroborating already-measured points to place alongside (different gene sets,
so context not curve): `runs/geomwin_iv30_refit` 53/70 at 4.8k,
`runs/ens_bag_union` 55/70 at 8.6k.

**Pre-registered read, written down before any number is seen:**
- n12480 ≥ 57/70 → scale still paying; triggers the follow-up curation plan.
- n12480 in 54–56/70 → flat-to-marginal; scale is saturating, record it and
  stop pulling this lever.
- n12480 ≤ 53/70 → the `npz_union` result was noise; say so plainly and
  retract nothing (numbers stay, per convention).

**Model tier: strong** — this is the measurement the plan exists for.

### Stage 7.4 — one ensemble on the largest set (local, conditional)

**Only if 7.3 is not the ≤53/70 case.** Six members on
`labels_union2.n12480.tsv`, reusing Stage 2's proven-diverse families exactly
(no new axes — Stage 4's null says breadth is done): baseline, `ens_allfeat`
(`--feature-subset all`), `ens_twohead`, `ens_depthscale`, `ens_augsub`,
`ens_nn15` (via `--nn-frac 0.15`). Spec committed as
`runs/ensembles/e4_union2.json` (equal weight, `mean_prob` — Stage 4 showed
rule/weighting is a null), scored on dev with `--ensemble`.

The question is narrow and worth stating up front: **does the free +1–2 from
view diversity survive at 1.44× the data, or does scale absorb it?** McNemar
E4 vs the best single 7.3 run answers it.

**Model tier: strong** for member config and interpretation; launching is
mechanical.

### Stage 7.5 — write-up and commit

`plans/signal-native-v2-scale-union2-results.md` in the stage-by-stage style of
the ensemble results doc, and a new `plans/signal-native-v2-resume-5.md`
superseding resume-4. Update `CLAUDE.md`'s v2 headline **only if** the dev
number improves, and label it dev-tuned. Update the honesty ledger: the
`DPSI_BETA` caveat carries forward unchanged (uniform batches stay clean), and
a new entry records that **the RI gene pool is now ~190 genes from exhausted**
— this lever cannot be pulled a third time without changing the
one-gene-per-event design.

Commit per stage (build → verify → commit). **Never `git push`**; never create
or switch branches.

---

## Critical files

| file | change |
|---|---|
| `sim/merge_batches.py` | `--subsample-events N` on the `--from-labels` path, reusing `_drop_genes_to`; factor out the shared write tail |
| `sim/run_batch.sh` | **unchanged** — invoked with a longer `--exclude-genes` chain |
| `sim/experiment/batches_union2/b{12..15}/` | new (on pasteur, `npz/`+`labels.tsv`+`design.tsv`+logs rsynced back) |
| `sim/experiment/npz_union2/`, `labels_union2*.tsv` | new merged pool + nested subsamples |
| `runs/scale_union2_n{4800,8640,12480}{,.json,.log}` | the curve |
| `runs/ensembles/e4_union2.json` | Stage 7.4 spec (committed; `runs/*/` is gitignored, `runs/ensembles/` is not) |
| `plans/signal-native-v2-scale-union2-results.md`, `plans/signal-native-v2-resume-5.md` | new |

No changes to `dsignal/` are anticipated — `train.py`, `score_benchmark.py`,
`ensemble.py`, `benchmark_split.py` and `baseline_gbm.py`'s shared harness
already do everything this plan needs.

## Verification

1. **Utility:** `sim/merge_batches.py --subsample-events` output — row counts,
   gene-set nesting across the three sizes, class fractions within ~1pp.
2. **Per batch, pre-expensive-steps:** `select_events.log` shows 0 gene
   overlap; `nn_hard` rows have `psi1 == psi2`; all four positive classes
   reached 180.
3. **Per batch, post-simulation:** `sim.psi_recovery_check`,
   `sim.width_leak_check`, `dsignal.audit_noise compare` — all clean vs the
   accepted `bag2` batches. A failing gate blocks the batch from the merge.
4. **Extractor invariants unchanged:** `python -m dsignal.verify`,
   `python -m dsignal.verify_features` still pass (no `dsignal/` change is
   planned, so these are a regression check, not a new gate).
5. **Merge:** 12,480 npz records, 12,480 distinct genes, no `ix` collisions.
6. **Training:** `train.py`'s existing no-leak assertion (gene-disjoint folds)
   must fire clean on the merged pool.
7. **Scoring:** dev only, `--half dev`, `runs/benchmark_events_scale_n*_dev.tsv`
   each 70 rows; every adjacent comparison through
   `dsignal.benchmark_split compare` (McNemar), never a bare accuracy delta.

## Explicitly out of scope

- Any test-half scoring, and any v1-vs-v2 claim.
- More ensemble breadth beyond the six reused families (Stage 4 null).
- More data bagging (Stage 3 control resolved it).
- The Stage 5 image-view hybrid — real signal, but needs its own distillation
  plan, not a hybrid ship.
- Refitting `DPSI_BETA`/`PSI_LEVEL_BETA`; `--psi-dist beta` is not used here.
