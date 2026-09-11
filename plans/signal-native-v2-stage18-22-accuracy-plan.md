# signal-native-v2 — Stages 18–22: improving call accuracy on real data

## Context

The branch's best model (`runs/ensembles/e4_union2.json`) scores 58.7% collapsed on `benchmark_new` dev (n=242), 87.1% on `benchmark/` dev (n=70), 48.5% on the 2,422-event PTBP1 arena. Stages 14–17 (real-NN injection, multi-label real data, domain-adversarial term) all failed their gates and the resume doc leaves the line closed. The user wants a plan that moves accuracy and an explicit answer on whether a change of model would help.

**User decisions (this session):** a sim-pretrained model fine-tuned on caller-derived real labels is an acceptable headline; a v1+v2 combination may be tried but v2-only is the preferred final result; both new arenas and the gated re-simulation are authorised on pasteur; **`benchmark_new` dev is the primary target** (pooled collapsed + per-class recall), `benchmark/` dev the regression guard, the arena confirmatory.

### Diagnosis (read-only, from committed TSVs/logs/code this session)

1. **One class dominates the errors.** `benchmark_new` dev: 68 of 100 errors are curated NN called as a positive (NN→CSE 27, →A5 15, →A3 13, →RI 13). Positives alone are ~80% (clean CSE 86%, A 84%, RI 64%). Arena: 665/800 NN wrong. Not an operating-point problem: even uncorrected, only 37–55 NN calls are made against 79 NN truths and accuracy doesn't move (`plans/signal-native-v2-bnew-dev-results.md:131-151`).
2. **Curated NN is a different population from sim or arena NN.** `p_NN` separates NN from positives at AUC 0.82 on the arena (caller *negatives*, p>0.2) but 0.64 on curated data (caller *false positives*: rMATS∧SUPPA p<0.05, rejected by a human). Stage 16: arena NN transfers to arena NN (+20 pts) and not at all to curated NN (8.9%). Curated-NN recall falls with junction count (1–2 junctions 3/8 → >10 junctions 1/9): windows *with* splicing but *without a convincing differential event* read as positives. Even the model's confidence is flat (~50% accuracy across max-p 0.4–0.8 on dev).
3. **Second-order residuals**: RI (v1 89% vs v2 75% on the arena; RI→A3/A5 on every set; RI is the class hurt by `--feature-subset all` and by the coverage branch) and CSE→A5/A3 (15/107 on dev).
4. **Capacity is not the limiter.** `ens_u2_wide/deep/shallow` in-sim macro-F1 0.899–0.912 vs base 0.906; `e5_arch` 60/70 dev = base. Scale curve: in-sim F1 rises, transfer plateaus at ~8.6k events.
5. **The sim-real gap is large and the model can see it** — it learns "real BAM record" from 474 examples (Stage 15). Sim is 3.5–5.7× deeper, blander in dPSI (81.6% of real events sit under sim's 0.30 floor), smooth in coverage (`sim/run_polyester.R:100-112` passes no positional/GC bias; real coverage is ragged — the RI mechanism Track D suspected), under on novel junctions and CSE/NN intronic coverage. Nobody has measured *which features* carry the signal. The branch's sim-realism-by-knob record is "0 for 3", but all three were scored on `benchmark/` dev (no NN, n=70) and each moved a *label-relevant* axis (depth, dPSI); nuisance-axis realism has not been tried.
6. **Information a curator uses that is not featurized**: read-level alignment QC (soft-clip, `NM` mismatch, NH>1, MAPQ — the TRASH signature; tags present in all four real cohorts), replicate concordance (real read names are `SRR<run>.<n>`, `audit_noise._library_of` already resolves them; SUPPA's empirical p uses per-replicate TPM; **polyester read names do not carry the sample and sim BAMs have no `@RG`** — a re-sim must write `RG` tags), deletion coverage (`delcov_*` stored, unused), `stats_*` (stored, unused).
7. **Infrastructure gaps**: no `--init-from`/freeze/warmup/fixed-epoch mode; no per-row weights (`merge_batches.FIELDS` drops extra columns); no abstention; GBM never scored on real data; `--dump-oof` broken (`train.py:609` unpacks 5-tuples, `read_labels` returns 6).
8. **Constraints**: `batches_union2` BAMs are gone from pasteur → any extraction change means re-simulating the identical design (~13 × 1.5–2 h); RI gene pool nearly exhausted → replace, never add batches. New arena ≈ 4 h pasteur. No new manual curation. `benchmark_new` test (243) reserved for one final spend; `benchmark/` test spent. **`ptbp1_arena` shares 92 genes (130 rows) with `benchmark_new` dev and 298 rows with its consensus genes** → any model trained on ptbp1 rows can never be scored on `benchmark_new` test, and 18.2's transfer test must cluster-exclude. `benchmark/` (Tra2) is MDA-MB-231; the others are HepG2.

### Should the model change? (direct answer)

**Not the architecture, and not first.** Every capacity/architecture variant lands within seed noise; every failure above is a *training-signal* failure (the sim contains no curated-NN-like windows; the model can tell real from sim), not a fitting failure. A bigger Transformer, a GNN, or a 1D-CNN over raw coverage trained on the same sim inherits the same gap — Track D is the proof: a richer coverage reader learned sim-RI's smooth-intron signature and *lost* RI on real data.

What should change, in evidence order:
1. **The training signal** — pretrain on sim, fine-tune on multi-cohort, cohort×class-balanced real data (Stage 20). This is what the original v2 plan specified and has never been done as a fine-tune (only as unweighted single-cohort mixing).
2. **The representation's inputs, not its architecture** — read QC, replicate concordance, coverage shape (Stage 19, needs a re-sim; the parts computable from existing npz are pulled forward into Stage 18.5). The coverage branch is re-tested only *after* sim coverage shape is realistic (Stage 22a).
3. **A different learning regime** — self-supervised pretraining on unlabelled real windows — is the one "model change" whose mechanism attacks the gap directly; expensive, gated on Stage 18 (Stage 22b).

Everything below is 3-seed, McNemar'd, dev-only, pre-registered per stage; `benchmark_new` test is spent once at the end.

---

## Stage 18 — Diagnose before building (local, ~3 days; no pasteur, no main-model training)

### 18.0 Extract the real windows not yet in npz *(mechanical)*
- `ptbp1_arena` (1,622 pos + 800 NN): add `--no-exclude` and `--kind NAME` flags to `sim/build_real_pos_extract_csv.py` / `sim/build_real_nn_extract_csv.py` (they already verify row alignment against `candidates/*.csv`), run with `--kind real_pos_ptbp1` / `real_nn_ptbp1` → `sim/experiment/real_{pos,nn}_ptbp1/`; `python -m dsignal.extract --min-anchor 5 --max-nh 10`; `dsignal.verify` on the positive CSVs (NN already 1600/1600).
- `benchmark/` (140, both halves — domain samples only, never a training pool): generalize `dsignal/real_supervision_ceiling.py cmd_build` with `--benchmark-dir --split --half both --kind real_tra2`; gene id = catalogue best-boundary match (`audit_noise.padding_report` logic) else cluster.
- Re-run the same `build` for `benchmark_new` dev with ENSG ids from `benchmark_build/consensus.tsv` keyed on `(curation.csv_source, padded region)` → new dir `runs/real_dev_npz/` (do not overwrite the committed `runs/real_supervision_ceiling`). **Test half is never extracted.**
- **Jittered-pad variant**: re-extract the whole real pool once more with U[0,100] bp added per edge (seeded), so 18.1 can measure how much of the domain signal is the window convention (real = +0/+100 fixed per class, `benchmark_build/make_candidate_csvs.py:22`; sim = U[0,100], `sim/window.py:65`).
- Pool: ~3,790 real windows, 4 cohorts (`tra2`, `bnew_dev`, `ptbp1`, `aqr`), all `kind` starting with `real_` so `batch["is_real"]` needs no change.

### 18.1 Real-vs-sim discriminator — new `dsignal/domain_probe.py` *(Opus/Sonnet: the interpretation is the deliverable)*
- `build` subcommand: rows via `baseline_gbm.build_dataset` (+ `return_meta` for `kind`, `chrom/start/end`) under `FeatureConfig`/`feature_subset` read from `--ckpt runs/scale_union2_n12480/fold0.pt` (never CLI defaults); one cached table `runs/domain_probe/table.npz`. Sim = `npz_union2` subsampled per class to the real pool's label counts (seeded).
- Groups for `baseline_gbm.grouped_folds`: **50 kb clusters over npz coordinates** (`benchmark_split.cluster`), not gene ids (inconsistent across cohorts, and it handles the ptbp1/bnew overlap).
- `real-vs-sim` subcommand: LightGBM binary (reuse `baseline_gbm` config), 5-fold cluster-grouped OOF → pooled AUC, per-cohort AUC, **leave-one-cohort-out AUC** (sim artefact vs cohort signature), **class-conditional AUC**; per-feature Mann-Whitney AUC; gain importance; **greedy removal curve** (re-rank after each removal, k = 1..30). Variants: `no-geometry` (deployed), `all`, `no-geometry` with `depth_norm="scale"` rows (exact depth invariance; the `DEPTH_ABSOLUTE` mask alone leaks via the pseudocount in `log2_count_ratio`, `features.py:180-194`), and the jittered-pad real pool. Also run with `--sim-npz` pointing at `npz_beta`, `npz_stage3`, `npz_geomwin` — a free realism ranking of every past simulation.
- Outputs: `runs/domain_probe/rvs/{oof.tsv, features.tsv, removal_curve.tsv, summary.json}`.

### 18.2 Real-only NN expressibility *(mechanical + Opus/Sonnet write-up)*
- `real-nn` subcommand: real rows only (ptbp1 + aqr = 3,406), cluster-grouped 5-fold: 5-class (`baseline_gbm.report`/`report_collapsed`) and binary NN-vs-positive (AUC, recall at 5%/10% FPR). Transfer: train on ptbp1+aqr **minus any cluster containing a `bnew_dev` row**, score `bnew_dev`'s 79 NN / 163 positives. LOCO aqr→ptbp1, ptbp1→aqr.
- Paired McNemar between this GBM's OOF on the 242 dev rows and the Transformer ceiling's `runs/real_supervision_ceiling/oof.tsv` — does the Transformer's inductive bias help on real labels at all.

### 18.3 Free re-scores (checkpoints exist) *(mechanical)*
`runs/beta_allaug`, `runs/beta_depthscale`, `runs/stage3` (effect-size realism; only ever scored on the NN-free benchmark) and `runs/ens_union2_twohead` alone, on `benchmark_new` dev with `--out` per-event TSVs → per-class recall from the confusion matrix (note: `score_benchmark` has no `--by-class`; `benchmark_split compare --by-class` does).

### 18.5 Shape/deletion features on the existing npz *(Opus/Sonnet for features.py; mechanical for runs)*
Pulled forward from Stage 19 because they need no re-extraction: `covshape` group = `cov_autocorr_{1,2}` (lag-50 autocorrelation of the coverage profile; `cov_cv_*` already exists), `delcov` group = deletion-coverage summaries from `delcov_*`. New named groups so `--feature-subset` can zero them (three call sites: `baseline_gbm.apply_feature_subset`, `dataset.apply_feature_subset`, `dataset.apply_feature_subset_meta`); `verify_features` involution/condition-swap checks extended. Score them with the 18.1 probe first (are they domain tells?), then 3 seeds vs base on both dev sets. This answers "do hand-built shape features transfer" *before* spending 25 h on a re-sim.

### 18.4 Decision doc `plans/signal-native-v2-stage18-domain-probe.md` + `resume-10.md` *(Opus/Sonnet)*
Rules fixed now:
- 18.1 pooled AUC ≥ 0.9 and removal curve < 0.7 within k ≤ 15 → those features are Stage 19's target list; both 19 arms viable. AUC > 0.85 past k = 30 → gap is diffuse; feature-side arm dropped, Stage 22b moves up behind Stage 20.
- Jittered-pad AUC materially lower than fixed-pad → the sim window sampler (or a deployment-time jitter TTA) is a cheap fix to add to Stage 19/21.
- 18.2 curated binary AUC < 0.70 with 3.4k real labels → current features cannot express curated NN; Stage 19's new groups become the NN lever and Stage 20's NN expectation is "arena-NN and positives only". ≥ 0.80 → real supervision (Stage 20) is the NN lever.

---

## Stage 20 — Real supervision done right (local GPU + 2 short pasteur jobs, ~1 week; starts in parallel with 18)

Stage 14–17 mixed *one* real cohort, unweighted, majority-CSE, into 12.5k sim rows. Missing: cohort diversity (so "real" isn't learnable), class balance within the real fraction, and a fine-tuning regime (no sim in the fine-tune stage ⇒ no real-vs-sim feature to learn).

### 20.1 Arenas 3 and 4 *(mechanical; Opus/Sonnet if the pipeline breaks)*
`benchmark_build/run_arena.sh` + `make_nn_candidates.py --n-total 800`, ~4 h each, `limitram`/four thread-pool vars/`nice -n 19`/`/proc/loadavg`. Selection (pre-registered): ENCODE shRNA KD with its own `possible_controls`, paired-end ≥100 bp, **one K562** (PRPF4 is the recorded pair-2 backup; verify both on the portal before fetching). Multi-run replicates: concatenate FASTQs in a pre-step, don't edit `fetch_align_benchmark_new.sh`. Add `--outSAMattrRGline ID:$srr` to the STAR call so replicate tags exist for Stage 19. Extract positives + NN with the existing builders, gene-disjoint from `ptbp1_arena`/`benchmark`/`benchmark_new` (`build_exclusion_set`, union with `exclude_genes_stage16.tsv`). Only npz/labels return. Training cohorts: AQR + pair 3 + pair 4; eval: PTBP1 confirmatory, `benchmark_new` dev gate, `benchmark/` dev guard.

### 20.2 Fine-tuning path in `dsignal/train.py` *(Opus/Sonnet — training loop)*
- `--select-on none`: fixed-epoch mode (no inner split, no early stop, `--epochs` exactly, `--avg-last-k` tail averaging reused from `train.py:354-365`); `--refit` with it is an argparse error. Held-out fold still scored once.
- `--init-from DIR`: real-pool fold k ← `DIR/fold{k}.pt`. `model_config` comes from the parent (`replace(parent_cfg, domain_adversarial=args.domain_lambda>0)`), `load_state_dict(strict=False)` asserting missing ⊆ `{domain_head.*}` (parent configs predate that field); **scaler from the checkpoint, never refit**; assert `classes`, `feature_subset`, `feature_config` match. Record `parent_checkpoint`; store the fine-tune set's `train_class_counts` and pre-register scoring with an explicit `--train-labels`.
- `--freeze {none,encoder}` (encoder = all but `head.*`, `type_head.*`, `diff_head.*`, `domain_head.*`; optimizer over `requires_grad` params).
- `--warmup-epochs N`: `SequentialLR(LinearLR, CosineAnnealingLR)`; `N=0` is bit-identical to today's `CosineAnnealingLR`.
- Per-row weights: `weight` column (default 1.0) through `read_labels` (7-tuple) → `__getitem__` → `collate` (default 1.0 so `predict._forward`'s item still works) → loss with `reduction="none"` and `(l·w).sum()/(cw[y]·w).sum()` (equals PyTorch's weighted mean when w≡1) in the single-head loss, `two_head_loss`, `domain_loss`. `merge_batches.FIELDS` gains `weight` (empty → 1.0).
- `--balance-real-cells FRAC`: computed in-train on each fold's training entries — every real `(kind, label)` cell gets equal total weight, real rows sum to FRAC × sim total (arm A) or cells equal (arms B/C). `--class-weights {balanced,none}` (default `balanced`) to avoid double balancing.
- `--grl-schedule {const,ramp}`: `lam = 2/(1+e^{-10p})−1` over global steps, passed as `grl_lambda` to `domain_logits` (Stage 17's flagged follow-up).
- Fix `--dump-oof` (`train.py:609`, add `kind` column).
- **Verification**: `--init-from … --select-on none --epochs 0` → `torch.equal` on every tensor + scaler arrays vs parent, and `score_benchmark` per-event TSV identical to the parent's `runs/bnew_dev_*`; no-`weight`-column run `allclose` to a pre-change run (summation order changes); `--warmup-epochs 0` and `--grl-schedule const` reproduce pre-change JSONs; `--freeze encoder` trainable count = head params and encoder tensors `torch.equal` after training.

### 20.3 Pre-register `plans/signal-native-v2-stage20-real-finetune.md`, three arms × 3 seeds
Parent = `runs/scale_union2_n12480` (then each e4 member for the ensemble variant). Real pool = 3 cohorts.
- **A — mixed, balanced, ramped GRL**: sim + real, `--balance-real-cells 0.10 --class-weights none --domain-lambda 1.0 --grl-schedule ramp`, `--select-on inner --refit`.
- **B — head-only fine-tune**: `--init-from --freeze encoder --select-on none`, real-only, cells equal, 20 epochs, lr 1e-4.
- **C — full fine-tune, low LR**: `--init-from --freeze none --select-on none`, real-only, lr 3e-5, `--warmup-epochs 2`, 10 epochs, `--avg-last-k 3`.
Gate = Stage 14's three conditions (median NN recall > 13.9% on `benchmark_new` dev; ≥2/3 seeds; no beyond-noise regression on either dev set: 10/242, 5/70) **plus** leave-one-cohort-out (train on 2 of 3 real cohorts, confirm on PTBP1 the effect is not cohort-specific) **plus** per-class recall — a CSE-only gain with A5/RI losses (Stage 16's shape) is a FAIL regardless of pooled accuracy. Real labels are ~67% pure; report it.

### 20.4 Score, write up, fold *(Opus/Sonnet)*
If an arm passes, apply it to all six e4 members → `runs/ensembles/e6_realft.json`, dev only. Expected value, stated honestly: most likely moves arena-type NN and positives (Stage 16 already moved CSE significantly); curated NN moves only if 18.2 says the features can express it.

---

## Stage 21 — Inference-side levers (local, days; independent)

- **21.1 Selective calling** *(mechanical)*: `predict.py`/`score_benchmark.py` `--min-confidence` (abstain → `NA`); `score_benchmark` prints the accuracy–coverage curve from `p_*`. Measured this session: `benchmark/` dev top-50% by confidence = 94%, top-25% = 100%; `benchmark_new` dev flat below max-p 0.8. No fitting on real data.
- **21.2 v1+v2 combination** *(Opus/Sonnet for the rule; mechanical scoring)*: pre-register **one** rule before scoring — mean of posteriors (v1 re-run via `dsEnsemble.py` to emit softmax, local GPU) and the fallback "v2 unless v2's RI-vs-A margin < τ and v1 says RI". Evaluate on the arena (n=2,422; dev lacks power), confirm on `benchmark_new` dev. Reported alongside v2-only; v2-only remains the preferred final.
- **21.3 Operating point** after Stage 20: re-run `prior_sweep` on dev; keep 0.1 unless a 3-seed-consistent argmax beats it.

---

## Stage 19 — Re-simulate once, carrying the curator's information (pasteur ~25 h unattended + ~1.5 weeks local; gated on 18.4)

Because the union2 BAMs are gone, any extraction change costs a re-sim: do it **once** on the **identical union2 design** (same events/genes/PSI draws/seeds), carrying every change below, so old-npz vs new-npz on the same events is a clean ablation. First check which intermediates survive per batch on pasteur (`design*.tsv`, `count_matrix.tsv`, `transcripts.ordered.fa`, …); `run_batch.sh` skips steps whose outputs exist, and steps 2–4 regenerate bit-identically from the seed if only `design.tsv` survives. **Prove identity** by diffing the new `labels.tsv`/`count_matrix.tsv` against the old.

### 19.1 Sim-side coverage shape *(Opus/Sonnet for R/knob wiring; mechanical batch loop)*
- The knobs that change *coverage shape* in the installed fork are `bias = "rnaf"|"cdnaf"` (positional fragment start; global) and per-sample **`frag_GC_bias`** (101 × n_samples retention-by-GC matrix; thins reads unevenly). Both are validated by `.check_extras` and forwarded to `generate_fragments` — add them to `sim_args` in `sim/run_polyester.R:100-112`. **`gcbias` is silently ignored by `simulate_experiment_countmat`** and, applied manually via `polyester::add_gc_bias(countmat, models, ordered)`, modulates *transcript abundance* (per-replicate PSI), not shape — treat it as a separate "replicate PSI noise" knob gated by `sim.psi_recovery_check`, off by default. First check on pasteur: `data(rnaf); data(cdnaf)` exist in the fork.
- `frag_GC_bias` thins fragments (mean retention < 1) → realised depth drops; pre-register accepting it (sim is 3.5–5.7× too deep anyway) vs rescaling the count matrix.
- Knobs: `run_polyester.R` positional args 12–14 (`bias`, `frag_gc_bias.tsv`, `gcbias.tsv`); `run_batch.sh` env `POLY_BIAS`, `FRAG_GC_BIAS`, `GC_BIAS` (defaults reproduce old batches byte-for-byte); `build_experiment.py` `SeedSequence.spawn(6)` → new 6th stream draws per-sample curve parameters; nothing drawn when off.
- Sim BAMs get replicate identity: `--outSAMattrRGline ID:${name} SM:${name}` in `sim/align_reads.sh:53-61` (`samtools merge` keeps the six `@RG`).
- **Acceptance** on a 2 h pilot (b12 re-simulated; old b12 npz is local): `audit_noise compare --by-label` with `cov_cv` promoted into `METRICS` (`audit_noise.py:429-453`; computed at :397 but never gated) plus a new `cov_autocorr_50`; the 18.1 probe on b12-new vs b12-old (pooled and per-cohort AUC must drop, coverage columns must leave the top of `features.tsv`); `psi_recovery_check`. Full re-sim (13 batches) only if the pilot passes.

### 19.2 Extraction-side *(Opus/Sonnet — CIGAR/coordinate semantics)*
`dsignal/extract.py`: `library_of(read) = RG tag if present else _library_of(query_name)` (moved from `audit_noise`, which imports it back). Additive npz keys, old records still load (`load_record` tolerant): `libs_{1,2}`, `junctions_lib_{1,2}` (long form `start,end,lib,count,nh_weighted_count`), and a `readqc` block in `stats_*_json` (`n_softclip_reads`, `softclip_bases`, `aligned_bases`, `nm_sum`, `n_nh_gt1`, `n_mapq_lt255`, per-library `n_reads_kept`; soft-clip is CIGAR op 4, counted in the same loop). `dsignal.verify` check: per-lib sums equal pooled counts exactly, QC fractions in [0,1].

### 19.3 Feature-side *(Opus/Sonnet)*
`NormalizedRecord` gains `lib_count_{1,2}` `[J,L]`, `n_libs`, `qc_{1,2}`; every rebuild site carries them (`normalize_record` incl. flip order, `mirror`, `rescale_depth`, `dataset._replace`, all five augmentations — `subsample` thins per lib). **Cap at the 2 largest libraries per condition** (`FeatureConfig.max_libs=2`) so ENCODE (2), Tra2 (3), sim (3) look alike — otherwise `n_libs` is a new domain tell. Groups: `replicate` (token: `rep_cv_{1,2}`, `rep_frac_t`, `rep_sign_agree`, `rep_min_present_{1,2}`; event: `max_abs_rep_frac_t`, `mean_rep_cv`), `readqc` (per condition `softclip_read_frac`, `softclip_bp_frac`, `mismatch_rate`, `nh_gt1_frac`, `mapq_lt255_frac`), plus 18.5's `covshape`/`delcov`. Old-format record → zeros + `has_replicate_info=False`. `verify_features` check 7: condition-swap antisymmetry of `rep_frac_t`, scale invariance of `rep_cv`, zeros on old records. **Audit `readqc` with `audit_noise` and the 18.1 probe on the pilot before training on it** — illumina5 vs real SNP/editing mismatch is itself a likely domain tell; keep it a separate subset group so 19.4 can run with and without it.

### 19.4 Train and gate *(mechanical launch; Opus/Sonnet write-up)*
(a) old npz/old features (control), (b) new npz/old features (shape realism alone), (c) new npz/new features, (c′) without `readqc`. 3 seeds each, both dev sets + arena, McNemar. Re-run the probe on (c) — discriminator AUC becomes a standing metric next to `audit_noise`. Fold winners into the ensemble; then re-run Stage 20's winning arm on top.

---

## Stage 22 — Representation changes, conditional

- **22a Coverage branch re-test** (`--coverage-bins 128`) on the bias-corrected sim, Track D's 3-seed protocol. Hypothesis: the RI loss (−0.17/−0.14) disappears.
- **22b Self-supervised pretraining on real windows** — only if 18.1 says the gap is diffuse *and* Stage 20 fails to move curated NN. ~20–50k random expressed-gene windows from the local real BAM pairs (+ pairs 3/4 subsets), pretext = masked-junction-feature reconstruction + condition-swap contrast on `JunctionSetTransformer`, then sim-label fine-tune. Separate plan doc; ~2 weeks.

---

## Final measurement

One spend of `benchmark_new` test (243): best dev-selected model of this line vs v1's six-net ensemble vs `e4_union2`, pre-registered before scoring. **Any model whose training pool contains `ptbp1_arena` rows is ineligible** (gene overlap). Nothing else touches test.

## Order, cost, model tier

| step | what | cost | tier | blocks on |
|---|---|---|---|---|
| 18.0 | real npz extraction + jittered-pad variant | ~half day | mechanical | — |
| 18.1–18.3 | domain probe, real-only NN, free re-scores | ~2 days | Opus/Sonnet (analysis), mechanical (runs) | 18.0 |
| 18.5 | covshape/delcov features on existing npz, 3 seeds | ~1.5 days | Opus/Sonnet | 18.1 |
| 20.1 | arenas 3/4 | 2 × 4 h pasteur, unattended | mechanical | — (start day 1) |
| 20.2 | fine-tune path + identity checks | ~1.5 days | Opus/Sonnet | — |
| 20.3–20.4 | 9 runs (~1 h each) + gate + fold | ~3 days | mechanical + Opus/Sonnet write-up | 20.1, 20.2, 18.4 |
| 21.1 | selective calling | ~half day | mechanical | — |
| 21.2 | v1+v2 rule | ~1 day | Opus/Sonnet rule, mechanical scoring | — |
| 19.1 | polyester bias + RG + pilot batch | ~1 day + 2 h pasteur | Opus/Sonnet | 18.4 |
| 19.1 full | 13-batch re-sim | ~25 h pasteur unattended | mechanical | pilot passes |
| 19.2–19.4 | extraction/features/runs | ~1 week | Opus/Sonnet + mechanical | re-sim |
| 22a | coverage branch on new sim | ~1 day | mechanical | 19 |
| 22b | SSL pretraining | ~2 weeks | Opus/Sonnet | 18.4 + 20 outcome |
| final | test spend | 1 h | — | all |

## Verification (every stage)

`dsignal.verify` after any extraction; `dsignal.verify_features` after any feature change; `audit_noise compare --by-label` + the 18.1 discriminator AUC after any re-simulation; 3 seeds and McNemar on every benchmark comparison (noise ceilings 5/70, 10/242); `benchmark_new` test untouched until the final spend; build → verify → commit per step; pasteur discipline (`limitram`, four thread-pool vars, `nice -n 19`, `/proc/loadavg` before/after); resume doc updated at the end of each stage; never delete a measured number.

## Critical files

- Modify: `dsignal/train.py` (init-from, fixed-epoch mode, weights, warmup, freeze, GRL ramp, `:609`), `dsignal/dataset.py` (`read_labels` :343, `__getitem__` :421, `collate` :476, `_replace` :123, augmentations, `apply_feature_subset*`), `dsignal/baseline_gbm.py` (`build_dataset` :155, `apply_feature_subset` :116, `grouped_folds` :176), `dsignal/features.py` (`NormalizedRecord` :218, new groups, `DEPTH_ABSOLUTE_*` :170), `dsignal/extract.py` (`save_record` :310, `load_record` :328, `extract_condition` :205), `dsignal/audit_noise.py` (`METRICS` :429, `_library_of` :96), `dsignal/predict.py`, `dsignal/score_benchmark.py`, `dsignal/real_supervision_ceiling.py` (`cmd_build` :77), `sim/build_real_{pos,nn}_extract_csv.py`, `sim/merge_batches.py` (`FIELDS` :41), `sim/run_polyester.R` (:100-122), `sim/align_reads.sh` (:53-61), `sim/build_experiment.py` (:227-230), `sim/run_batch.sh` (:95-100), `benchmark_build/run_arena.sh`, `sim/fetch_align_benchmark_new.sh`.
- New: `dsignal/domain_probe.py`; `plans/signal-native-v2-stage18-domain-probe.md`, `-stage20-real-finetune.md`, `-stage19-resim-features.md`, `-resume-10.md`.

## Status

Approved 2026-09-11; saved here per the `CLAUDE.md` workflow. Implementation starts in a separate session, Stage 18 first (18.0 → 18.1/18.2/18.3 → 18.5 → 18.4 decision doc), with 20.1 (arenas 3/4) and 20.2 (fine-tune path) launched in parallel since neither blocks on Stage 18.
