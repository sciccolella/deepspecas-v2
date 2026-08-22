# signal-native-v2 — Stage 8: fresh benchmark + ensemble breadth

*Plan, 2026-08-22. Successor to `plans/signal-native-v2-scale-union2.md` (Stage 7).
Entry context: `plans/signal-native-v2-resume-5.md`.*

---

## Context

Stage 7 left the branch with its best number ever — `runs/ensembles/e4_union2.json`
at **61/70 = 87.1% dev collapsed** — and no way to believe it. The existing
benchmark's test half was spent once in Stage 6 (`plans/signal-native-v2-ensemble-results.md`),
dev has been the tuning surface for five sessions, and e4's lift over the best
single run is 7 discordant events at McNemar p=0.13 when 8–10 are needed at n=70.

So the bottleneck is no longer *finding* a lever. Three levers were on the table:

* **Scale** is measured saturated — the Stage 7 three-point curve rises 51 → 57 →
  56/70 dev while in-sim macro-F1 climbs monotonically. The ceiling is specific to
  zero-shot transfer.
* **View-diversity ensembling** is the one axis still moving, and it is *less*
  exhausted than resume-5 implies (see "What the Stage 4 null actually says").
* **Relaxing one-gene-per-event** buys more scale — the lever already measured
  saturated — at the cost of a real refactor and two new failure modes. Deferred;
  see the last section, which records findings worth keeping.

None of them can produce a claim without a fresh held-out set. This plan therefore
runs two tracks in parallel — they do not contend for resources (**A** is pasteur +
manual curation, **B** is the local RTX 4080) — and converges on a single
pre-registered one-shot spend.

**Intended outcome:** a fresh, fully curated 140-event benchmark from a *different*
BioProject, split dev/test, plus one pre-registered v2 candidate and a v1 reference
scored on it exactly once — the first v1-vs-v2 comparison this branch is entitled
to make since Stage 6.

---

## Pre-registration discipline (read before Stage A5 or B4)

The whole value of this plan is that the new benchmark is clean. Two rules,
non-negotiable, in the tradition of Stages 3/6/7:

1. **The v2 candidate is frozen from Track B on the *old* dev half, and written
   into this plan file, before a single number is read off the new set.**
2. **The new set's test half is scored exactly once.** Its dev half is the future
   tuning surface; test is spent the moment it is read.

Additionally: the fresh set coming from a different BioProject is what finally
retires the `DPSI_BETA`/`PSI_LEVEL_BETA` honesty caveat (fitted on all 140 events
of the *old* benchmark, open since resume-2). Do not re-fit them on the new set.

---

## Track A — the fresh benchmark

Everything here is unscripted in this repo. `git log --diff-filter=A -- benchmark/A3.csv`
is a single bare upload (`28f8a4c7`, 2023-10-30): no rMATS/SUPPA intermediates, no
candidate pool, no generating script. And `benchmark/c1.bam`/`c2.bam` are
`samtools view -L events.bed` subsets (`benchmark/README.md:50-59`) — **no new event
can be sourced from the committed BAMs.** The pipeline is built from README prose.

### A0 — pick the BioProject *(strong model; user decides)*

Shortlist 3–4 candidate public human RNA-seq studies and put the table in front of
the user. Selection criteria, in order:

* Paired-end, ≥3 replicates × 2 conditions (matches the c1/c2 pooling convention
  that every CSV row and `dsignal/extract.py` assume).
* A splicing-relevant perturbation (an SR/hnRNP factor knockdown or similar), so
  the rMATS+SUPPA p<0.05 intersection is populated across all four classes —
  **RI in particular**, which is the thinnest class in the old set (17 events).
* GRCh38-mappable; ideally ≥40 M reads/sample so junction counts are comparable to
  PRJNA255099.
* **Not** PRJNA255099, and not a study sharing its cell line if avoidable.

Deliverable: a short table (accession, design, depth, why) → user picks one.
**Blocking** — nothing downstream starts without it.

### A1 — acquire and align *(cheap model; pasteur)*

Existing assets on pasteur (`/data/conda/ciccolella/deepSpecas-v2/`): `ref/` (5 GB,
GRCh38 + Ensembl 99 GTF) and `star_index/` (28 GB) are already built and are what
`sim/run_batch.sh:37-39` and `sim/align_reads.sh` use. 769 GB free on `/data` (92%
used) — check again before pulling FASTQs, and delete them after alignment.

1. `prefetch`/`fasterq-dump` the six runs into `deepSpecas-v2/fastq/`.
2. Align per run with the **same STAR invocation `sim/align_reads.sh` uses**, so the
   `jM`/`jI` tags `dsignal/extract.py` depends on are present and identically
   parameterised. Do not hand-roll a new STAR command.
3. `samtools merge` per condition → `c1_new.bam` / `c2_new.bam`, index.

Per `CLAUDE.md`: `limitram 32G`, all four thread vars at 1, `nice -n 19`, and check
`/proc/loadavg` before and a minute after launch — pasteur was at load 34/64 with
other users' jobs when this plan was written.

### A2 — rMATS + SUPPA consensus *(strong model; pasteur)*

New code, in a new `benchmark_build/` directory (keep it out of `sim/`, which is the
simulator, and out of `dsignal/`, which must stay annotation-free).

* Install rMATS and SUPPA2 into a **new** conda env on pasteur. Never `pip install`
  into `DSnet`, `DSsim`, or `DSpoly2v2`.
* Run rMATS turbo (`--b1`/`--b2` the per-condition BAM lists) and SUPPA2
  (`generateEvents` is already done — `sim/events/events_*_strict.ioe` exist — then
  `psiPerEvent` + `diffSplice`).
* `benchmark_build/consensus.py`: intersect the two call sets at p<0.05, per class,
  matching events by coordinate with a tolerance (the old README's counts were 62
  A3 / 54 A5 / 41 IR; expect the same order of magnitude). Emit a candidate TSV with
  coordinates, both tools' p-values and dPSI, and the class.
* Sanity gate: the consensus counts must be ≥140 total with RI ≥25, or A0's study
  choice was wrong and the shortlist's next entry should be tried.

### A3 — candidate pool for curation *(cheap model)*

* `benchmark_build/make_candidate_csvs.py`: consensus TSV → `region,bam1,bam2` rows
  using the **same window convention as the old set**. `benchmark/events.bed` is
  interval-identical to the CSV region column (`benchmark/README.md:38-42`), and the
  v2 window sampler was derived from it via `python -m dsignal.audit_noise padding` —
  so measure the old set's padding distribution and reproduce it, rather than
  inventing a new one. Getting this wrong silently changes the geometry the model
  was trained for.
* Render IGV screenshots for every candidate into `benchmark_new/igv_images/<class>/`,
  named `chrom.start.end.png` (the old convention). `~/igv.sh`/`~/run_igv.sh` exist
  and were presumably how the originals were made.

### A4 — manual curation *(user; the schedule driver)*

The user inspects the candidate screenshots and keeps/drops, exactly as the
original 62/54/41 → 28/30/17 cut. Target **~140 events, full set**, so the result is
splittable and future sessions have a dev surface again.

Deliverables: `benchmark_new/{A3,A5,ES,IR}.csv` (headerless `region,bam1,bam2`,
basenames must stay in `CSV_LABEL`, `dsignal/score_benchmark.py:76`), plus
`benchmark_new/events.bed`, plus a `README.md` recording the study, the consensus
counts, and the curation counts.

### A5 — freeze the set *(cheap model)*

1. `samtools view -L benchmark_new/events.bed` the merged BAMs → `benchmark_new/c1.bam`
   /`c2.bam` (~35 MB for 140 events, matching the old pair), index, commit. Repeat
   the old README's caveat: off-window quantities measured on these are lower bounds.
2. **Split, before scoring anything**:
   `python -m dsignal.benchmark_split make --csv benchmark_new/*.csv --out benchmark_new/split.tsv`
   — works as-is (`dsignal/benchmark_split.py:229`), including the 50 kb
   single-linkage clustering and the per-class quota walk. Verify the printed
   no-straddle assertion and per-class dev/test balance.
3. Delete the FASTQs and full BAMs from pasteur. Per `CLAUDE.md`, only CSVs, the
   subset BAMs and TSVs come back.

### A6 — v1 reference on the new events *(cheap model; local GPU)*

`dsignal/v1_reference.py` runs **no model** — it reformats cached v1 prediction CSVs
(`v1_reference.py:42-81`). So v1 must be re-run:

```bash
# six-net ensemble, same flags as run_ensemble_parallel.sh:10-16
python dsEnsemble.py --csv benchmark_new/A3.csv \
  --pyplot2s data/trained_nets/pyplot2s/trained.pth ... --zoom -1 --a --cse
python -m dsignal.v1_reference --glob 'runs/v1_new/pred_*.csv' \
  --out runs/benchmark_events_v1_new.tsv
```

All six 91 MB nets are committed; `dsEnsemble.py:57,163` hardcodes `cuda:0`, so this
is local. Budget minutes, not hours (`report.md:37-49`: ~145 s for all 140 events).
Note v1 emits `"A"`, never A3/A5 — only `correct_collapsed` is comparable
(`v1_reference.py:9-16`).

---

## Track B — ensemble breadth (local, old dev half, runs concurrently)

### What the Stage 4 null actually says

resume-5 and `plans/signal-native-v2-scale-union2.md:174,243` paraphrase Stage 4 as
"breadth is done". **That overstates its source.** Stage 4
(`plans/signal-native-v2-ensemble-results.md:95-101`) cached E3's 16-member
posteriors and swept 3 combination rules × 2 weightings — a bit-identical null about
*rule and weighting*, not about member count or new view families. The separate
Stage 3 null (E3 == E2 bit-for-bit) is specifically about adding **bagged** members
on top of view diversity. Neither forecloses new families, and resume-5:85-88 names
this axis as the one still moving.

Concretely, **architecture has never been varied in any spec** — every member of
every ensemble is 1,403,229 params (`runs/ens_union2_*.log:3`). e4 also dropped
Stage 2's `ens_cse` member and runs all six members at `seed=0`.

All members train on `sim/experiment/npz_union2` + `labels_union2.n12480.tsv`, the
Stage 7 pool, with the e4 baseline config except the varied axis. Member composition
is **pre-registered here** — no greedy selection.

### B1 — architecture families *(cheap model; flags already exist)*

`train.py:419-423` exposes `--d-model/--n-heads/--n-layers/--d-ff/--dropout`, all
untouched to date. Four members, chosen to span capacity and regularisation rather
than to tune:

| run | flags |
|---|---|
| `ens_u2_wide` | `--d-model 288 --d-ff 768` |
| `ens_u2_deep` | `--n-layers 6` |
| `ens_u2_shallow` | `--n-layers 2 --d-model 128` |
| `ens_u2_drop35` | `--dropout 0.35` |

### B2 — seeds × views, and the restored boost member *(cheap model)*

* `ens_u2_seed{1,2,3}` — the e4 baseline config at seeds 1–3. E1 showed seeds alone
  were worthless at 4.8k; seeds crossed with views at 12.5k has never been run.
* `ens_u2_cse` — `--class-weight-boost CSE 1.75`, restoring Stage 2's dropped member.
* `ens_u2_ri` — `--class-weight-boost RI 1.75`. RI is the thinnest benchmark class
  (17/140) and this axis is repeatable and entirely untried beyond CSE.

### B3 — attention-bias ablations *(strong model; small code change)*

The only item here needing code. `dsignal/model.py:212-215` builds `distance_bias`
and `relation_bias` as zero-init `nn.Embedding` tables **shared across layers**, with
no way to disable either.

* Add `use_distance_bias: bool = True` and `use_relation_bias: bool = True` to
  `ModelConfig` (`model.py:66-84`), consumed in `_attention_bias`
  (`model.py:227-242`) by skipping the corresponding term.
* Add matching `--no-distance-bias` / `--no-relation-bias` flags to `train.py`,
  threaded into the `ModelConfig(...)` construction at `train.py:233`.
* **Backwards compatibility is free and must be verified, not assumed:**
  `dsignal/predict.py:178` reconstructs via `ModelConfig(**ck["model_config"])`, so
  new fields with defaults leave every existing checkpoint loading — the same
  mechanism `two_head` used (`model.py:76-79`). Verify by re-scoring a stored run
  (e.g. `runs/geomwin_iv30_refit`) and confirming it reproduces its recorded
  53/70 dev exactly.
* Members: `ens_u2_nodist`, `ens_u2_norel`.

### B4 — compose, evaluate on old dev, freeze *(strong model)*

Pre-registered specs, written before any number is read (`runs/ensembles/`):

| spec | members |
|---|---|
| `e5_arch.json` | e4's 6 + B1's 4 |
| `e5_seedview.json` | e4's 6 + B2's 5 |
| `e5_bias.json` | e4's 6 + B3's 2 |
| `e5_all.json` | e4's 6 + all 11 new members |

Score each on the **old dev half** with the settled operating point, reusing the
cached path so the sweep costs minutes (extraction is once per event and fans out to
all members, `ensemble.py:12-14`):

```bash
python -m dsignal.ensemble score --spec runs/ensembles/e5_all.json --half dev \
  --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
  --cache-out runs/ens5_cache_dev.npz
python -m dsignal.benchmark_split compare \
  runs/benchmark_events_e4_union2_dev.tsv runs/benchmark_events_e5_<x>_dev.tsv --half dev
```

**Selection rule, pre-registered:** the Track C candidate is the spec with the
highest old-dev collapsed accuracy; **ties break toward the larger member count**
(the fuller superset, matching Stage 6's E3-over-E2 rule). If no spec beats e4's
61/70, **e4_union2 is the candidate** — Track B then reports a clean null and Track
C proceeds unchanged. Record the choice in the results doc *before* touching the new
benchmark.

Do **not** re-tune `nn_weight`, the combination rule, or the weighting. The first is
settled (`dsignal/prior_sweep.py` docstring); the latter two are Stage 4's null.

---

## Track C — the one-shot spend

Runs only after A5 and B4 are both complete and the candidate is written down.

1. **New dev first** (sanity, not a decision): score the frozen candidate and the v1
   reference on `benchmark_new` dev. A collapse relative to old dev means something
   is wrong with A1–A3 geometry, not with the model — investigate before touching
   test. `--csv` is free-form `nargs="+"` (`score_benchmark.py:197`) and `--split`
   overrides the split path, so this needs **no code change**; the one cosmetic
   wart is `report()`'s hardcoded "all 140"/v1-123/140 note (`score_benchmark.py:104-126`),
   which should be made to read its counts from the loaded set.
2. **Test, once.** Pre-register the criterion in the same shape Stage 6 used, before
   reading: *parity* = v2 collapsed test accuracy ≥ v1's; *win* = McNemar p<0.05 in
   v2's favour. Both v2 and v1 scored on the same new test rows, keyed `(csv, region)` —
   the keying that a retracted number once got wrong (`CLAUDE.md`, ensemble-results
   Stage 0.1).
3. `python -m dsignal.benchmark_split compare <v2>.tsv <v1>.tsv --half test --split benchmark_new/split.tsv`.
4. Write `plans/signal-native-v2-fresh-benchmark-results.md` and a new `resume-6.md`.
   Per convention: **no measured number is ever deleted**, retractions sit next to
   what they retract, and the honesty ledger carries forward — with the beta-fit
   caveat now retired, and a new one added if the new set's curation touched
   anything the model saw.

---

## Deferred: relaxing one-gene-per-event — findings worth committing

Not implemented by this plan; record these in the results doc so the next session
does not re-derive them.

* **The RI scarcity is largely self-inflicted, and the fix is 3 lines.** `by_label`
  in `sim/select_events.py:191` is insertion-ordered by catalogue order
  `("SE","A3","A5","RI")` (`sim/catalogue.py:113`), so **RI is selected last and
  loses every contested gene**. 92.6% of RI genes (3,642/3,935) also carry a
  CSE/A3/A5 event, and **1,536 RI-carrying genes were burned by non-RI rows** across
  the consumed designs (A3 521, A5 464, CSE 360, nn_hard 191) — roughly 8.5 batches
  of RI runway. Selecting the scarcest class first recovers most of it **without
  touching the invariant**.
* **No gene-budget utility exists.** The "~188 RI genes remaining" figures in
  resume-5 were computed by hand. `sim/catalogue.py:133-143` prints per-label totals
  but knows nothing about consumption. A ~15-line "catalogue genes minus the
  `--exclude-genes` union, per label" reporter is the cheapest missing tool, and
  resume-5:105-107 says every future batch plan must check this budget.
* **The relaxation itself breaks at a known assert.** `sim/build_experiment.py:302`
  (`assert full.index.is_unique`) fires immediately, because the loop is per design
  *row* while `abundance.build_count_matrix` (`sim/abundance.py:129`) emits a
  whole-*gene* matrix plus one `PREMRNA_<gene>` and one `NOVEL_<gene>` row. Fixing it
  is a ~150-line re-key of `build_experiment.py:206-272` to per-gene, plus
  `build_premrna_fasta.py:55-60` and `run_polyester.R:76,89`'s row-order identity.
* **PSI targeting is not composable.** `abundance.targeted_weights` (`:63-90`)
  rescales over a shared gene-wide weight vector; the docstring's ordering workaround
  (`abundance.py:121-127`) works only when exactly one event per gene is labelled.
  **60.4% of within-gene event pairs share ≥1 transcript** (RI 63.5%) — for those,
  whichever is applied last hits target and the other's `empirical_psi` drifts off
  `design.tsv`, which `sim/psi_recovery_check.py` gates on.
* **47.7% of within-gene RI pairs have directly overlapping event spans** before the
  ≤100 bp flank (`sim/window.py:68-103`). Overlapping windows feed the same reads to
  two records with different labels — and GroupKFold puts both in the same fold, so
  it becomes irreducible within-fold noise, not detectable leakage.
* **Nothing downstream needs changing:** `dsignal/train.py:552` →
  `baseline_gbm.grouped_folds` → `GroupKFold` + `assert_no_gene_leak` already handles
  K>1 correctly, and `sim/gene_split.py:5-8` explicitly anticipates it.
  `merge_batches._drop_genes_to` groups whole genes and works unchanged.
* **The decisive objection:** this buys scale, and Stage 7 measured scale as
  saturated past ~8,640 events. Revisit only if a later finding shows the plateau was
  an artifact of the simulation design rather than of transfer.

---

## Verification

**Gates that must pass before each track's output is used:**

* After any `dsignal/` change (B3 only): `python -m dsignal.verify_features` and
  `python -m dsignal.verify`.
* **B3 backwards compatibility:** re-score `runs/geomwin_iv30_refit` on old dev and
  confirm it reproduces **53/70** exactly. A different number means the `ModelConfig`
  change broke checkpoint loading.
* **A1 alignment parity:** run `python -m dsignal.audit_noise measure` on
  `benchmark_new/c1.bam` and compare against the stored old-benchmark measurement.
  The two are different studies so quantities will differ — what must *not* differ is
  the shape of the `jM`/`jI` tag availability and the strand-assignment path, which
  is what `dsignal/extract.py` depends on. A missing `jM` tag here silently degrades
  every downstream number.
* **A3 geometry parity:** `python -m dsignal.audit_noise padding` on the new
  `events.bed` must produce a padding distribution comparable to the old set's. This
  is the check that the new windows match the geometry the model was trained for.
* **A5 split:** `benchmark_split make` prints its own no-straddle assertion; also
  confirm 70/70 and per-class balance.
* **Track C keying:** every per-event TSV must have exactly as many rows as the
  half's size, keyed `(csv, region)` — the check that catches the duplicated-region
  bug that produced one retracted v1 figure.

**End-to-end smoke test, before the real Track C run:** score the frozen candidate on
`benchmark_new` **dev** with `--half dev --split benchmark_new/split.tsv`, confirm
row counts and that no event errors out of extraction, and only then proceed to test.

---

## Build order and model tiers

| stage | where | tier | blocks |
|---|---|---|---|
| A0 study shortlist | local | strong (+ user decision) | all of A |
| B1/B2 members | local GPU | cheap | B4 |
| B3 bias ablation | local | **strong** | B4 |
| A1 acquire/align | pasteur | cheap | A2 |
| A2 rMATS+SUPPA consensus | pasteur | **strong** | A3 |
| A3 candidate CSVs + IGV | pasteur/local | cheap | A4 |
| B4 compose, freeze candidate | local | **strong** | C |
| A4 curation | **user** | — | A5 |
| A5 freeze set + split | local | cheap | C |
| A6 v1 reference | local GPU | cheap | C |
| C1/C2 one-shot spend | local | **strong** | — |

Commit after each stage (`build → verify → commit`, per `CLAUDE.md`). Never `git push`;
never create or switch branches.
