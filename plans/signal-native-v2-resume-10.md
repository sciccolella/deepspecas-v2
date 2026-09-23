# signal-native-v2 — START HERE (2026-09-14, after Stage 18 + 20.1/20.2/20.3)

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
was 872 G free after the builds. Local GPU idle. All Stage 18.5/20.3
checkpoints are local-only under `runs/` (gitignored); their JSON/log/TSV
are committed.

## Stage 20.3 result (2026-09-14): all three arms FAIL

Full pre-registration and Results: `plans/signal-native-v2-stage20-real-finetune.md`.
Three fine-tuning regimes (B head-only, C full/low-LR, A mixed-with-GRL) x
seeds 0/1/2, trained on the gene-disjoint 3-cohort real pool
(`sim/experiment/npz_real3`/`npz_union2_real3`, 3,294/15,774 rows — the
disjointness assertion caught and dropped 11 leaking rows the per-cohort
exclusion builds missed, see Results). **B and C move arena NN recall
backwards** (parent 11.8–15.0% -> 1.6–9.4%, McNemar p < 0.001 every seed, arm
*behind* not ahead) despite raising pooled accuracy everywhere else — real-only
fine-tuning suppresses NN calls rather than improving them. **A passes rules
1 and 2 cleanly** (arena NN 22.2–24.0% vs parent 11.8–15.0%, p < 0.0001 all 3
seeds; every eval's pooled accuracy up double-digits with no regression) but
**fails rule 3's literal per-class count rule**: curated NN on `benchmark_new`
dev drops by 3–7 events in every seed while the other four classes all gain.
That call looked ambiguous against the plan's own closing note ("curated NN
is not a gate input"), but is now **resolved**: the same 6–8 curated regions
regress from NN to a confidently-predicted structural class in every one of
the 3 seeds (`p_NN` typically 0.4–0.7 -> 0.00–0.06, the new class's
probability rising to 0.5–0.99) — a reproducible, mechanistically legible
effect, not noise. The FAIL verdict for A stands, confirmed rather than
merely adopted by default. LOCO (rule 4) was not run for any arm, since none
cleared rules 1–3 together.

**Consequence, per the plan's pre-registered decision table**: real
supervision on the current features is closed for the second time, with a
stronger negative than Stage 14–17. **Stage 22b (self-supervised pretraining
on real windows) is next**, per Stage 18 rule 1. Stage 21 (selective calling,
v1+v2 rule) is independent and runs regardless — it was never blocked on
20.3 and can start any time.

## Stage 21 (2026-09-16): done, all three sub-stages

`plans/signal-native-v2-stage21-inference-levers.md` has full pre-registration
and Results for all of 21.1/21.3/21.2. One paragraph each:

- **21.1 selective calling**: `--min-confidence` (abstain → `NA`, excluded
  from accuracy) landed in `predict.py`/`score_benchmark.py`, plus
  `--coverage-curve`/`--from-tsv` (top-K%-retained table, post-hoc from
  cached `p_*`, no re-scoring). Sanity check reproduced the parent plan's
  quoted `benchmark/` dev numbers (94.3%/100.0% at top-50%/25%) exactly. A
  real, monotone lever on every eval, largest where NN isn't in the mix.
- **21.3 operating point**: `nn_weight=0.1` rechecked on 3 seeds
  (`npz_union2`/n12480) — each seed's own argmax beats 0.1, but no value is
  the argmax in more than one seed and none is McNemar-significant. **0.1
  stays.**
- **21.2 v1+v2 mean-of-posteriors**: pre-registered, scored on the arena
  (primary, n=2422) and `benchmark_new` dev (confirm, n=242). v1+v2
  significantly **underperforms v2-only** on both (arena 37.8% vs 48.5%
  collapsed p<0.0001; bnew dev 50.4% vs 58.7% p=0.0091) — v1's confident-wrong
  NN calls dominate the average and swamp a real RI gain (90.5% vs 75.0%
  collapsed recall). v1+v2 beats v1-only only marginally (arena p<0.0001 but
  +1.3 pt; bnew dev +1.6 pt, not significant). **v2-only remains the
  preferred headline**, as decided before scoring; v1+v2 reported alongside.

## Stage 22b (2026-09-21–23): done, no variant passes

Full pre-registration and Results: `plans/signal-native-v2-stage22b-ssl-pretrain.md`.
Self-supervised pretraining (MFM masked-junction-reconstruction +
NT-Xent/SimCLR contrastive, 3 variants: MFM-only, contrastive-only, both)
on a 36,000-window real pool (`sim/experiment/real_ssl_pool`, 3 cohorts,
gene-disjoint from every eval set), then 9 downstream fine-tunes (3
variants x 3 seeds, `--init-encoder-from`, otherwise identical to the
`scale_union2_n12480` recipe), scored on `benchmark_new` dev (primary),
`benchmark/` dev (guard) and `ptbp1_arena` (confirmatory). **mfm and both
improve the point estimate in all 3 seeds** against their seed-matched
base (`benchmark_new` dev collapsed: mfm +2/+9/+7, both +9/+10/+11 of 242)
but **no individual seed clears McNemar significance** (best p = 0.099,
`both` seed 2) — underpowered at n=242, the same pattern Stage 20's arm A
hit on a different rule. Contrastive-only is inconsistent (2/3 seeds
regress) and the weakest variant on every eval set. The gain in mfm/both
is almost entirely CSE recall (+~8 pt) plus a smaller A3/A5 gain — NN
recall is unchanged from the base's already-broken regime, consistent with
Stage 18's finding that curated NN isn't expressible from these features.
**Verdict: no variant passes the pre-registered gate.** Checkpoints kept as
evidence (`runs/ssl_*`, `runs/ssl_ft_*`), not promoted into any deployed
recipe. Incidental finding: `dsignal.predict.Predictor` has no missing-key
whitelist for the 22b.3-added `mask_token`/`recon_head`/`proj_head` params
(unlike `train.py`'s loader), so it can no longer strictly-load a
pre-22b.3 checkpoint — a small follow-up, not addressed here.

**Consequence**: real-signal training interventions are now closed twice
(Stage 20.3's fine-tuning, Stage 22b's pretraining) without a gate pass.
The accuracy plan's remaining evidence-order item is **Stage 19**
(re-simulation, junction-tail target), gated on nothing but its own ~25h
pasteur cost.

## Next: Stage 19 (re-simulation, junction tail)

Not pre-registered yet — needs a plan discussed and approved before any
implementation, per this repo's workflow convention. Targets noise/novel
junctions at 1–3 reads and one-condition-only junctions (Stage 18 rule 1's
sim-side target, since the sim/real gap is diffuse, not positional/GC), and
must write `@RG`/indels before `delcov` or replicate features can be used.

## Open honesty caveats (carried forward + new)

1. `nn_weight=0.1`, `DPSI_BETA`/`PSI_LEVEL_BETA` — unchanged from resume-7/8/9.
   **Stage 21.3 (2026-09-16) rechecked this**: `dsignal.prior_sweep` on the
   three seed-matched `npz_union2`/n12480 checkpoints, `benchmark_new` dev.
   Each seed's own argmax beats 0.1 (seed 0 `nn_weight=0.25` 56.6%, seed 1
   tied `uncorrected`/`0.5` 54.5%, seed 2 tied `uniform`/`1.0` 57.4%), but no
   value is the argmax in more than one seed and none crosses McNemar
   significance at n=242 (p=0.503/1.000/0.062). No 3-seed-consistent
   alternative — **0.1 stays**. Detail:
   `plans/signal-native-v2-stage21-inference-levers.md` §21.3 Results.
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
