# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`deepSpecas v2` classifies alternative-splicing (AS) events from RNA-seq BAM files with **no image rendering**: junctions and base-resolution coverage are extracted straight from the BAMs and classified by a junction-set Transformer. Five classes, fixed order, defined once in `dsignal/dataset.py`: `("A3", "A5", "CSE", "NN", "RI")`. Training data is simulated end to end (SUPPA event catalogue → PSI-targeted abundances → polyester reads → STAR); the released model is used zero-shot on real data.

It succeeds **deepSpecas v1** (image-based, six ResNet50 nets, <https://github.com/sciccolella/deepSpecas>), which lives in its own repository. Nothing of v1's code is here except a frozen copy of one function (`dsignal/_v1_coords.py`); v1's *results* are committed as TSVs.

This repository was split out of v1's `signal-native-v2` branch with history kept; see `HISTORY.md` (old→new hash map; **hashes cited in `plans/` and logs are old hashes**).

**Start here:** `RESULTS.md` (every claimable number, its source, caveats, retractions), then `plans/README.md` (the decision record; `plans/signal-native-v2-resume-11.md` is the state-of-play document). The programme is closed: every gated intervention of Stages 18–22 failed its gate and the final `benchmark_new` test spend passed (143 vs 110 / 243, p < 0.0001). **No held-out curated half remains**, so any new model-vs-model claim on curated data is exploratory.

## Environment

No requirements.txt/pyproject; dependencies come from conda/mamba.

| where | path | env | notes |
|---|---|---|---|
| anywhere | `envs/inference.yml` | `deepspecas-v2` | CPU torch + pysam; inference, scoring, `dsignal.verify*` |
| local | `/datasets/deepSpecas-v2/envs/DSnet` | `DSnet` | all training (RTX 4080). **Never `pip install` into it.** |
| remote | `ssh pasteur`, `/data/conda/ciccolella/deepSpecas-v2` | `DSpoly2v2`, `DSsim` (`envs/sim_*.yml`) | simulation / polyester / STAR / extraction; no GPU |

The local root filesystem can run out of space (it was at 100%); put new environments and package caches under `/datasets/deepSpecas-v2/`. Only `.npz` records, labels and TSVs come back from pasteur — never BAMs or FASTA. Pasteur's `dsignal/` holds only the modules shipped there; rsync a new one before running it. What is not in git (BAMs, pools, other checkpoints) is catalogued in `DATA.md`; `scripts/fetch_data.sh` fetches/verifies the benchmark BAMs.

### Running jobs on pasteur — read before launching anything parallel

**pasteur is a shared machine.** Other people's jobs run on it, and a careless launch takes it away from them. On 2026-08-17 an unthrottled 32-way render drove the load average to **230 on 64 cores** and made new login shells impossible for ~40 minutes. `crash_pasteur.md` has the exact reproduction; the rules follow from it.

1. **Always wrap remote jobs in `limitram`.** `limitram <LIMIT> <command> …` is `/usr/bin/limitram`, a `systemd-run --user --scope -p MemoryMax=` wrapper:
   ```bash
   THREADS=8 nohup limitram 32G nice -n 19 sim/run_batch.sh b12 12 <prior design.tsv ...> > logs/b12.log 2>&1 &
   ```
2. **Cap the thread pools.** `limitram` bounds memory, not CPU; numpy and torch each size a thread pool to the machine's core count, so N processes silently become N×64 threads. Export all four, every time:
   ```bash
   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
   ```
3. **`nice -n 19`** every background job, keep shard counts modest (≤12). Check `cat /proc/loadavg` before launching and a minute after; if the 1-minute figure goes past ~50, throttle.
4. **Measure throughput early** — sample the output twice a few minutes apart. Both failures showed up as *zero* rows produced while the CPU was busy.
5. **Never `pkill -f <pattern>`** when the pattern also appears in the ssh command line carrying it — it kills your own session. Use `pkill -f "[p]attern"`.
6. Conda is not on PATH in non-interactive ssh shells: `source /opt/miniforge3/etc/profile.d/conda.sh && conda activate <env>`.

## Verification gates (there is no pytest suite, linter or CI)

Scripts are run directly with `python -m`. The gates that stand in for tests:

- `python -m dsignal.verify --csv benchmark/{A3,A5,ES,IR}.csv` — extractor coverage/junction parity against the frozen v1 reference (`dsignal/_v1_coords.py`); needs the benchmark BAMs.
- `python -m dsignal.verify_features --npz-dir <npz pool>` — canonicalization involution, exact A3↔A5 flip, coverage off-by-one; needs a simulated pool.
- `python -m sim.width_leak_check`, `python -m sim.psi_recovery_check` — simulation-design gates.
- `python -m dsignal.audit_noise compare` — **the acceptance test after any re-simulation**: annotation-free, runs the identical code path on real and simulated BAMs.
- `python scripts/check_results.py` — recomputes `RESULTS.md`'s tables from the committed per-event TSVs (read-only).
- The packaged-model gate: `dsignal.score_benchmark --ensemble models/e4_union2/manifest.json --half dev --split benchmark_new/split.tsv --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 --csv benchmark_new/{A3,A5,ES,IR,NN}.csv` must reproduce `runs/bnew_dev_e4_union2.tsv` (bit-exact in DSnet; 0 prediction differences, |Δp| ≤ 1.3e-7 in `inference.yml`). **Dev only.**

## Pipeline / data flow (`dsignal/` + `sim/`)

### `dsignal/` — extraction → features → model → evaluation

1. **`extract.py`** — junction set + base-resolution coverage per `region,bam1,bam2` row, straight from the BAMs. **Never reads a GTF**: strand comes from STAR's `jM` tag (with the `+20` annotated bit masked off via `% 20`) or a reference-dinucleotide fallback. One `.npz` per event.
2. **`features.py`** — the single source of truth for the representation, consumed by *both* the GBM baseline and the Transformer so they are compared on identical information. Per-junction token matrix `[J, 48]` and event-level vector `[32]`. Minus-strand records are canonicalized by mirroring within the window, so "donor" always means the transcription-direction donor and the left-right flip is an exact A3↔A5 map.
3. **`dataset.py`** — torch `Dataset`, five augmentations (`flip`, `condition_swap`, `subsample`, `junction_dropout`, `jitter`, all off by default), `FeatureScaler` (fit on the training fold only, stored in the checkpoint), `collate`.
4. **`model.py`** — `JunctionSetTransformer`: a pre-norm encoder over a *set* of junctions with two learned additive attention biases (bucketized signed log-distance between junction midpoints; a per-head scalar per discrete pairwise relation). The CLS token is projected from the event-level vector and never masked, so a zero-junction event is a valid CLS-only forward pass.
5. **`train.py`** — 5-fold `GroupKFold` **grouped by gene**, explicit no-leak assertion, out-of-fold predictions pooled; one checkpoint per fold.
6. **`predict.py`** — inference. Feature config, scaler, classes and feature subset come from the checkpoint, never CLI defaults. With no `--net`/`--ensemble` it runs the released ensemble at its frozen operating point. `PriorCorrection` handles the train/deploy class-prior mismatch — **applied per fold before the folds are averaged**; correcting the averaged posterior is a different, wrong estimator.
7. **`ensemble.py`** — `EnsembleSpec`/`MultiRunPredictor`: one `Predictor` per member run, extraction once per event. A packaged manifest also carries `defaults` and per-file sha256.
8. **`score_benchmark.py`** — scores a model against a curated benchmark; 5-class and A3+A5→A collapsed accuracy, both confusion matrices, accuracy vs region width, wall clock.
9. **`baseline_gbm.py`** — the LightGBM baseline and the **shared evaluation harness** (`grouped_folds`, `report`, `report_collapsed`, `print_vs_reference`). Route any new model through these or the numbers stop being comparable to the layer-by-layer history.
10. `audit_noise.py` (`measure`/`compare`/`padding`), `prior_sweep.py`, `benchmark_split.py` (`compare` = paired McNemar, `pool`), `v1_reference.py`/`combine_v1v2.py` (read committed v1 prediction TSVs only), `tabulate_runs.py`, `verify*.py`.

### `sim/` — the simulator

```
gtf_gene_spans / gtf_transcript_lengths / gtf_exon_structure   (GTF-derived tables)
  → catalogue.py            SUPPA .ioe files → Event objects (SE→CSE; MX/AF/AL excluded)
  → select_events.py        balanced design: positives + nn_plain / nn_hard / nn_novel
    + window.py             the extraction window around each event span
  → gene_neighbors, novel_junctions, build_premrna_fasta      (noise layers 2-4 inputs)
  → build_experiment.py     transcript × sample count matrix + the noise-layer knob panel
  → order_fasta.py          polyester matches matrix to FASTA by row index, not by name
  → run_polyester.R → align_reads.sh (STAR)      [run_batch.sh drives a whole batch]
  → build_extract_csv.py → dsignal.extract → merge_batches.py → training
```

Easy to get wrong:

- **One gene per event, and gene-disjointness across batches.** `select_events.py --exclude-genes` prevents cross-batch leakage that `GroupKFold` cannot see, because it groups within one labels file.
- **Independent RNG streams** per window draw, negative-span draw and noise layer, so adding a layer does not perturb earlier ones.
- **Window geometry is load-bearing.** `sim/window.py` is on its *third* sampler; its docstring deliberately retains all three as a decision record. Only v3 is live: `window = event span + U[0, 100] bp per edge`, label-independently (worth +16.5 points of zero-shot transfer).
- `workflow/run_all.sh` is a documentation-grade map of the whole pipeline (not run end to end).

## The benchmarks

- `benchmark/{A3,A5,ES,IR}.csv` — 140 manually curated real events (`c1.bam`/`c2.bam`; built from rMATS + SUPPA consensus, PRJNA255099, see `benchmark/README.md`). **No NN events.** Split 70 dev / 70 test (`benchmark/split.tsv`, cluster-disjoint at 50 kb). **Test is spent.**
- `benchmark_new/{A3,A5,ES,IR,NN}.csv` — 485 curated events incl. 159 NN (`split.tsv`, `curation.tsv`). **Test is spent** (2026-10-05).
- Everything tuned is read off **dev**. Use `--half {dev,test,both}` and `--split` on `score_benchmark`/`prior_sweep`; `python -m dsignal.benchmark_split compare A.tsv B.tsv --half dev` for the paired McNemar.
- Compare a **single** v2 model against a **single** v1 net (~76% on `benchmark/`), not against v1's six-net ensemble (87.9%).

## Workflow conventions (how to collaborate on this repo)

### Git
- Commits are fine to create as work is completed, but **never `git push`** — pushing is always the user's call. The remote is `origin` (`git@github.com:sciccolella/deepspecas-v2.git`).
- **Never create or switch branches on your own initiative.** Branching decisions belong to the user. If a new branch seems warranted, propose it and wait. The default branch is `master`.
- Large data never goes in git (`.gitignore` blocks `*.bam *.bai *.npz *.pth`, `sim/experiment/*/`). The only committed binaries are the 30 `models/e4_union2` checkpoints.

### Planning before writing code
Whenever a task requires writing or modifying code:
1. Produce a **step-by-step plan** and discuss it with the user first. Do not start implementing.
2. For each step, note **which model tier should implement it** (mechanical → cheaper/faster; judgment, architecture, tricky debugging → stronger).
3. Wait for explicit approval of the plan before any implementation.
4. Once approved, **save the plan under `plans/`** (e.g. `plans/<short-slug>.md`) and **stop** — a separate session implements it.
5. Alongside the saved plan, give the user a **short prompt** to paste into that implementation session (plan file path + one- or two-sentence goal).

### After implementation
Write a concise report (what was done, deviations from the plan, what was verified), and commit the work (no push).

## Conventions worth knowing before editing

- **Every comparison on a curated benchmark needs a significance test.** Differences of 4–6 events at n=70 are noise; baseline seed spread alone is ~5 events at n=70 and ~10 at n=242. Seed-replicate before believing an A/B; McNemar on paired per-event outcomes throughout.
- **Pre-register the metric and operating point before scoring anything held out.** Held-out halves are spent once; none remains.
- **No new manual curation** — ruled out as a data source (labour-intensive, partly interpretive). Any plan needing labels must get them another way.
- **Never delete a measured number from a docstring or plan doc.** Retracted findings stay, with the retraction next to them (`RESULTS.md` §6 lists the current ones). That is what makes the plan chain readable.
- Build → verify → commit each step before starting the next; verify against real data, smoke-test on a small subset before a full-scale run.
- The five classes are fixed; `score_benchmark.py` reports the A3+A5→A collapse as a derived number so results stay comparable with v1's collapsed 4-class figures.
- Feature-subset changes touch **three** call sites: `baseline_gbm.apply_feature_subset`, `dataset.apply_feature_subset`, and `dataset.apply_feature_subset_meta` (the last is the easy one to forget — it keeps `s_rel`/`e_rel` consistent with the dummy token count under the geometry control). The geometry group lives as private names in `baseline_gbm.py` and is imported by `dataset.py`.
- `N_TOKEN_FEATURES`/`N_EVENT_FEATURES` are baked into `ModelConfig` defaults and every saved checkpoint. Adding a feature does not break loading (config comes from the checkpoint) but will mismatch the stored scaler.
- `Predictor` loads checkpoints with `strict=False`, tolerating only the Stage 22b pretext keys (`mask_token`, `recon_head.*`, `proj_head.*`).
- Honesty caveats still open: `DPSI_BETA`/`PSI_LEVEL_BETA` in `sim/select_events.py` are fitted on all 140 `benchmark/` events (anything with `--psi-dist beta` is domain-calibrated; refit on dev only if reused); the simulated pool shares genes with the `benchmark_new` test set; the NN operating point (`nn_weight 0.1`) is pre-registered, not tuned on test.
