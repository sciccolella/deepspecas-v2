# signal-native-v2 — START HERE (2026-09-14, after Stage 18 + 20.1/20.2)

Resume document. Branch `signal-native-v2`, **nothing pushed** (per repo
convention — pushing is the user's call). Supersedes
`plans/signal-native-v2-resume-9.md` as the entry point; that file holds
the Stage 12–17 state and is still the place to read for *why* the
real-data line looked closed. The current programme is
`plans/signal-native-v2-stage18-22-accuracy-plan.md` (approved
2026-09-11); Stage 18's pre-registration and full results are in
`plans/signal-native-v2-stage18-domain-probe.md`.

## What changed since resume-9 (one paragraph)

Stage 18 measured the sim→real gap instead of guessing at it. A
LightGBM real-vs-sim discriminator on the model's own features is
**perfect (AUC 0.999)** on 3,788 real windows from four cohorts vs
class-matched simulation, for every cohort, every class, every past
simulation, with or without geometry, depth-scaled or not, and with the
window pad jittered — and it stays at 0.987–0.996 after the 30 most
informative feature families are removed. The gap is *diffuse*; the
biggest single tell is the **sparse junction tail** (real windows carry
~16 distinct junctions at ~3–4 reads each, sim ~6.6 at ~68), not coverage
shape. Curated NN is **not expressible** from the current features even
with 3.2k real caller labels (NN-vs-positive AUC 0.58 on `benchmark_new`
dev), whereas arena NN is (0.88 within, 0.83–0.85 across arenas). Two
hand-built feature groups (coverage autocorrelation, deletion coverage)
were built, probed and 3-seed tested: **neither transfers**, and deletion
coverage is a pure domain tell because polyester writes no indels. In
parallel, arena pairs 3 (PRPF4/HepG2) and 4 (U2AF2/K562) were built and
extracted as gene-disjoint real cohorts, and `train.py` gained the full
fine-tuning path (`--init-from`, fixed-epoch mode, freeze, warm-up,
per-row weights, cell balancing, GRL ramp) with bit-identity checks.

## Decisions the Stage 18 rules made (pre-registered, then scored)

| rule | outcome | consequence |
|---|---|---|
| 1 — discriminator | AUC > 0.85 past k = 30 in every variant → **diffuse gap** | Stage 19's *feature-side* arm is dropped as a gap-closing device; **Stage 22b (self-supervised pretraining on real windows) moves up behind Stage 20**. Stage 19's sim-side target, if it runs, is the junction tail, not positional/GC coverage bias. |
| 2 — jittered pad | 0.998 vs 0.999 → **does not fire** | the window convention is not a tell; no sampler/TTA change |
| 3 — curated-NN expressibility | AUC 0.58 < 0.70 → **fires** | Stage 20's NN expectation is "arena-NN and positives only"; curated NN needs new information (Stage 19 groups or 22b), not more real labels on these features |
| 4 — covshape/delcov | covshape median 130 = base 130; delcov 123 (−5…−12, one seed p = 0.017) | **not adopted**; `--event-groups` stays opt-in and off |

Numbers worth carrying in your head: base **seed spread on union2 n12480
is 128/130/134 of 242 on `benchmark_new` dev and 56/56/59 of 70 on
`benchmark/` dev** (single model, TTA, `nn_weight` 0.1). The e4 ensemble's
142/242 is still the reference for ensemble-level claims.

## What exists now that did not before

- **Real npz pool** (all `kind` = `real_*`, all `dsignal.verify`-clean):
  `sim/experiment/real_{pos,nn}_{ptbp1,aqr,prpf4,u2af2}` (committed) and
  `runs/real_dev_npz` (`benchmark_new` dev, ENSG ids) / `runs/real_tra2_npz`
  (`benchmark/`, both halves) — the latter two are gitignored and rebuild in
  minutes with `dsignal.real_supervision_ceiling build`. Every cohort has an
  `npz_jitter/` twin.

  | cohort | rows | may train on? |
  |---|---:|---|
  | ptbp1 | 1,622 + 800 | **no** — shares 92 genes with `benchmark_new` dev; domain sample / confirmatory eval only |
  | aqr | 510 + 474 | yes (Stage 14/16 exclusion) |
  | prpf4 (pair 3) | 285 + 447 | yes (gene-disjoint from ptbp1/benchmark/benchmark_new) |
  | u2af2 (pair 4, K562) | 1,109 + 480 | yes (same) |
  | bnew_dev | 242 | never (evaluation) |
  | tra2 | 140 | never (evaluation) |

- **`dsignal/domain_probe.py`** (`build` / `real-vs-sim` / `real-nn`) — the
  discriminator AUC is now a standing metric next to `audit_noise`; re-run it
  after any re-simulation (`--sim-npz` swap) or feature change
  (`--event-groups`). Tables under `runs/domain_probe/`.
- **`FeatureConfig.event_groups`** (`covshape`, `delcov`; `train.py
  --event-groups`, `--feature-subset no-geometry,no-covshape`), old
  checkpoints unaffected; `verify_features` has a 7th check.
- **`train.py` fine-tune path**: `--init-from DIR`, `--select-on none`
  (fixed epochs, `--avg-last-k`), `--freeze encoder`, `--warmup-epochs`,
  per-row `weight` column, `--balance-real-cells FRAC`, `--class-weights
  none`, `--grl-schedule ramp`; `--dump-oof` fixed (+`kind`). Identity
  checks in `runs/ft_ref` (pre-change checkpoints kept). `merge_batches`
  carries `weight`.
- **`benchmark_build/build_arena_pair.sh`** — one-command arena build;
  STAR now writes `@RG` (`sim/fetch_align_benchmark_new.sh`).
- **`sim/jitter_extract_csv.py`**, `--no-exclude`/`--kind` on both real
  extract-CSV builders, `real_supervision_ceiling build --benchmark-dir/
  --half/--kind/--gene-source`.

## What is running right now

**Nothing.** Pasteur holds `prpf4_work/` and `u2af2_work/` (FASTQ + BAM,
~130 G each; never leave pasteur) next to `ptbp1_work/`/`aqr_work/`; disk
was 872 G free after the builds. Local GPU idle. All Stage 18.5
checkpoints are local-only under `runs/` (gitignored); their JSON/log/TSV
are committed.

## Next: Stage 20.3 (not started — pre-register first)

Write `plans/signal-native-v2-stage20-real-finetune.md` *before* any arm
runs, per the parent plan §20.3: parent `runs/scale_union2_n12480`
(seeds 1/2 parents now exist too: `runs/union2_base_s{1,2}`); real pool =
aqr + prpf4 + u2af2 (build one merged labels/npz dir with
`sim/merge_batches.py --batch` over the six `real_*` dirs, or point
`--npz-dir` at a symlink farm); three arms × 3 seeds:

- **A** mixed: sim + real, `--balance-real-cells 0.10 --class-weights none
  --domain-lambda 1.0 --grl-schedule ramp --select-on inner --refit`;
- **B** head-only: `--init-from --freeze encoder --select-on none
  --epochs 20 --lr 1e-4`, real-only, `--balance-real-cells 0` (cells equal);
- **C** full, low LR: `--init-from --select-on none --lr 3e-5
  --warmup-epochs 2 --epochs 10 --avg-last-k 3`, real-only.

Gate = Stage 14's three conditions **plus** leave-one-cohort-out (confirm
on PTBP1) **plus** per-class recall (a CSE-only gain with A5/RI loss is a
FAIL) — and, per Stage 18 rule 3, the NN condition is judged on **arena NN
(PTBP1 confirmatory)**, with curated NN on `benchmark_new` dev reported but
not expected to move. Score every fine-tune with an explicit
`--train-labels` (the checkpoint's `train_class_counts` is the fine-tune
set's, which is not the deployment prior). Expected cost ~1 h per run
(GPU) — the 18.5 queue script `runs/stage18_5_queue.sh` is a usable
template.

After 20.3: Stage 21 (selective calling, v1+v2 rule) is independent and
cheap; Stage 22b is now ahead of Stage 19 in the order; Stage 19, if it
runs, targets the junction tail (noise/novel junctions at 1–3 reads,
one-condition-only junctions) and must write `@RG`/indels before `delcov`
or replicate features can be used.

## Open honesty caveats (carried forward + new)

1. `nn_weight=0.1`, `DPSI_BETA`/`PSI_LEVEL_BETA` — unchanged from resume-7/8/9.
2. `benchmark_new` test (243) unspent and reserved for the programme's one
   final spend; `benchmark/` test spent (Stage 6). **Any model whose
   training pool contains a `ptbp1_arena` row is ineligible for that
   spend** (92 shared genes with `benchmark_new` dev; 298 rows with its
   consensus genes).
3. The 18.2 "real-only GBM beats the Transformer ceiling" comparison
   (131 vs 99 of 242, p = 0.001) is **not an A/B** — different training
   protocols; it says the ceiling was data-limited, nothing about
   inductive bias.
4. Pair-4 RBP choice (U2AF2 over HNRNPC/SF3B4/HNRNPK) was my call on
   mechanistic-distance grounds, recorded before fetching; it is not a
   user decision.
5. Real arena labels are ~67% pure (pair 1's measured purity); pairs 2–4
   are unmeasured and assumed similar. Report it with every Stage 20 number.

## Conventions (unchanged)

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison: McNemar **and** seed replication
  (n=70 needs 8–10 discordant, n=242 roughly 10–14).
- Never delete a measured number; retractions go next to what they retract.
- Build → verify → commit each step.
- Training local (`DSnet`); simulation/alignment/extraction/arena building
  on pasteur under `limitram`, four thread-pool vars, `nice -n 19`,
  `/proc/loadavg` before/after. Pasteur env note: `DSpoly2v2` has no
  torch, `DSrender` has torch+pysam but no pandas — build extraction CSVs
  locally, extract remotely with `DSrender`.
- Gates: `dsignal.verify`, `dsignal.verify_features` (**seven checks**),
  `sim.width_leak_check`, `sim.psi_recovery_check`, `dsignal.audit_noise
  compare --by-label`, and now `dsignal.domain_probe real-vs-sim`.
