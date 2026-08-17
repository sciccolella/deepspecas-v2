# Documentation cleanup + model-improvement plan (signal-native-v2)

## Context

Branch `signal-native-v2`, HEAD `7cb91fa`, clean tree. Zero-shot transfer to the
140-event curated benchmark sits at **78.6% collapsed** (`runs/benchmark_geomwin.log`)
against v1's reproduced 87.9%. The v3 window fix is the only intervention that has
moved that number; depth invariance and effect-size realism were both measured null.

Two problems:

1. **The docs no longer describe the repo.** `CLAUDE.md` — the file every new session
   reads first — maps only the six root-level v1 files and never mentions `dsignal/`
   (15 modules) or `sim/` (33 modules), i.e. the entire body of work on the checked-out
   branch. It also carries a known factual error. Inside the v2 code, roughly 20
   docstrings still quote the **v1 or v2 window sampler**, pre-scale-up counts, or
   conclusions since retracted — including two that now argue the *opposite* of the
   current design intent.
2. **The next model step is undecided.** `plans/signal-native-v2-resume-2.md` leaves two
   readings open and ranks the v1-representation crossover first. This session's read of
   the error structure supplies a third, more specific mechanism that the resume doc does
   not name, and one free experiment that has never been run.

Both were confirmed by reading the code, not inferred from the plan docs.

Decisions taken with the user: v2 stays out of `README.md` (unpublished; README gets only
its v1 errors fixed); the benchmark is **split 70/70 dev/test**; cheap local experiments
front-load ahead of the crossover and the re-simulation batch.

---

# Part A — documentation

## A0. Save this plan into the repo

Repo convention (`CLAUDE.md`) puts plans under `plans/`. Copy this file to
`plans/signal-native-v2-docs-and-model.md` as step one, then implement from there.

## A1. `CLAUDE.md` — the highest-leverage file

Rewrite, keeping every currently-accurate v1 section. Changes:

- **Fix the factual error at L31.** "sums softmax outputs across nets" is wrong:
  `dsEnsemble.py:143-144` does `allout = sum(outs)` then one softmax — raw logits.
  `report.md:125-128` flagged this and it was never fixed.
- **Add a "Two lines of work" preamble**: v1 = image + ResNet50 ensemble (`master`,
  published); v2 = signal-native junction-set Transformer (`signal-native-v2`, in
  development, no image rendering at all). State which branch is checked out.
- **Add a `dsignal/` + `sim/` module map** at the same altitude as the existing v1 map:
  extractor → features → dataset/augment → model → train → predict/score_benchmark, plus
  the audit and verification tools; and the sim pipeline order
  (`gtf_*` → `catalogue` → `select_events`+`window` → `build_experiment` → polyester/STAR
  → `build_extract_csv` → `dsignal.extract` → `merge_batches`).
- **Point at `plans/signal-native-v2-resume-2.md`** as the entry point for current work.
- **Record the compute split**: training local (`/datasets/deepSpecas-v2/envs/DSnet`,
  never `pip install` into it; `DSv1` for v1), simulation/alignment/extraction on pasteur
  (`ssh pasteur`, `/data/conda/ciccolella/deepSpecas-v2`, envs `DSpoly2v2`/`DSsim`; only
  `.npz`/labels/TSVs come back).
- **Soften L19** ("no test suite"): still true for pytest/CI, but name the actual gates —
  `dsignal/verify.py`, `dsignal/verify_features.py`, `sim/width_leak_check.py`,
  `sim/psi_recovery_check.py`, and `dsignal.audit_noise` as the acceptance test after any
  re-simulation.
- Add `--shard` / `--gpu-render` to the v1 example invocations (both shipped on `master`).

## A2. `README.md` — fix v1, add nothing

- Delete the duplicated, self-contradictory `dsEnsemble.py --help` paste (L17-44 contains
  two disagreeing copies). Replace both blocks (and the `deepSpecas.py`/`plot2s.py` pastes
  at L70-91, L103-123, L128-156) with short flag summaries plus "run `--help` for the full
  list" — pasted help text has now drifted three times.
- Fix `--csv CSV  CSV to predict [region,BAM_path,GTF_path]` (L30, L74). No GTF is
  involved. **Also fix the help strings themselves** in `dsEnsemble.py` and
  `deepSpecas.py` — the README is faithfully reproducing a wrong string.
- Note `--zoom` is now required, and mention `--shard`/`--gpu-render`.
- Replace the `DSpoly2` env recipe (L1-7) or mark it as the v1 env; it has no torch and is
  not the working environment for anything.
- Caveat `runDS.smk` (L158) the way `CLAUDE.md:37` already does.

## A3. `report.md` — mark as v1/`master`, close what shipped

- Header note: this describes `master`'s inference throughput work, not this branch.
- L91 cites `plans/ensemble-inference-runtime.md`, which does not exist → `plans/render-throughput.md`.
- L98-107 proposes `--shard` and `--gpu-render` as future work; both shipped
  (`c675641`, `379cbba`). Mark done and record what they measured.
- L135-136 "No metrics anywhere" is now false — `dsignal/score_benchmark.py` exists.
  Narrow the claim to "no scorer for the *image* ensemble".
- Re-label the timing tables: they are pre-`--shard`/pre-`--gpu-render`.

## A4. `benchmark/README.md` — three additions

- Name `A3.csv`/`A5.csv`/`ES.csv`/`IR.csv` (only the `.txt` lists are mentioned; the CSVs
  are what every tool consumes).
- Document `benchmark/events.bed` and that it is byte-identical to the region CSVs — the
  single fact the whole v3 window fix rests on, currently undocumented here.
- Spell out the consequence of the `view -L` subsetting: off-window background is absent,
  so `antisense_frac`/`background_frac` measured on the real side are **lower bounds**.
- Add the ES consensus count (L12 gives A3/A5/IR only).

## A5. Plan docs

- `plans/signal-native-v2-checkpoint.md:3-9` — banner points at the superseded
  `resume.md` and states the disproved premise ("62.1%", "more data makes transfer
  worse") as fact. Retarget to `resume-2.md` and strike the premise.
- `plans/signal-native-v2-resume-2.md` — fix `HEAD fb30f1d` → `7cb91fa`; add a one-line
  warning that `README.md`/`CLAUDE.md` describe a different project (remove once A1
  lands); add the `dsignal/` CLI index and the two commands that reproduce 78.6%
  (`train --npz-dir sim/experiment/npz_geomwin --labels …geomwin.tsv --augment all
  --feature-subset no-geometry` → `score_benchmark --net runs/geomwin_allaug/fold*.pt`);
  mark which of the 15 `sim/experiment/npz_*` dirs are live vs archaeology.
- Add `plans/README.md`: one line per plan file, chronological, marking superseded ones.

## A6. Stale docstrings in `dsignal/` and `sim/`

Mechanical, but two of these now argue against the current design. Highest leverage first:

| file:lines | fix |
|---|---|
| `dsignal/baseline_gbm.py:19-30, 60-68` | Quotes the **v1** sampler ("flank log-uniform 0.1–10× the event span", "46.5%", widths 1,452–22,734bp) and calls geometry "a simulation-artifact shortcut which would not transfer". Directly contradicted by `sim/window.py:39-49`. Rewrite around v3: geometry is now partly deployment-realistic (13/17 curated IR recovered from geometry alone). |
| `sim/width_leak_check.py:10-15` | Concludes "at chance on accuracy" under the v2 draw; under v3 this check is *expected* to rise (`sim/window.py:46-48`). Invert the conclusion, keep the numbers as history. |
| `sim/gene_neighbors.py:7-8` | `--flank 0` justified by "the window already varies 0.1x-10x the event span" — dead v1 reasoning. Under v3 the window is span+≤100bp, so neighbour detection may be under-scoped. Re-justify or widen the default (flag it; do not silently change behaviour). |
| `dsignal/train.py:220-222`, `dsignal/dataset.py:262` | "geometry is a Phase-2 simulation artifact" — inherited from the dead `baseline_gbm` claim. See B1: the default itself is under test. |
| `dsignal/features.py:180-194, 353-373` | Advocates `depth_norm="scale"` as a free fix; the experiment came back null (`a7c198f`). Keep the mechanism, add the measured result. |
| `dsignal/train.py:232-235` | `--subsample-low` help argues 0.3 is wrong; `--subsample-low 0.15` was tried and *hurt* (73.6%). Record that. |
| `dsignal/score_benchmark.py:20-23` | "Expect the sim-only model to do poorly … Phase 4 is what closes that gap". Replace with the current number and note Phase-4 fine-tuning is not recommended. Also amend L11-13 "nothing in Phase 2 or 3 has been fit to it" — `DPSI_BETA` is fitted on it. |
| `dsignal/predict.py:50-51`, `dsignal/prior_sweep.py:4-6` | "41/140", "540/1260", "43% NN" — pre-scale-up. Current is 13/140 of 4,800 events at 25% NN. |
| `dsignal/audit_noise.py:439, 607-611` | Knob string still names `U(0.3, 0.8)` only (`--psi-dist beta` exists); the padding report closes with "read this as the specification for `sim/window.py`", already implemented. |
| `dsignal/tabulate_runs.py:19-21`, `baseline_gbm.py:51-57` | RF reference 0.488/0.325 is from `npz_scale1260` (v1 windows, 1,260 events). Re-state the provenance in the label, or refresh against `npz_geomwin`. |
| `dsignal/verify_features.py:300`, `dsignal/diagnose_cse_a3.py:191` | Defaults hardcode `npz_scale1260` and `runs/benchmark_events_windowfix.tsv`. Repoint at the geomwin equivalents. |
| `sim/window.py:44-46, 83` | Broken mid-sentence line wrap; a "under v2 …" back-reference inside a live function docstring. The three-iteration module docstring is a deliberate decision record — **keep it**, but make clear at each use site which iteration is live. |
| `sim/select_events.py:284` | `--psi-dist` defaults to `uniform` while its own help argues uniform excludes 81.6% of real events. Leave the default (beta was null on its own — see B4) but say so in the help instead of leaving a contradiction. |
| `sim/psi_recovery_check.py:1,3` | "for the pilot design" — it runs on every batch now. |

**Rule for this pass:** never delete a measured number. Retracted findings get kept with
the retraction next to them — that is what made the resume chain readable.

**Model tier:** A1, A3, A5 and the two "argues the opposite" rows of A6 need judgement →
strong model. A2, A4 and the remaining A6 rows are mechanical → cheaper model, then one
review pass.

---

# Part B — improving the model

## The error structure the plan is built on

30 collapsed errors out of 140 at the current best (`runs/benchmark_events_geomwin.tsv`,
measured this session):

| bucket | n | mean confidence in the wrong call |
|---|---|---|
| NN false positives | 13 (6 A5, 4 A3, 2 ES, 1 IR) | **p_NN = 0.71** — confident |
| CSE ↔ A confusion (8 CSE→A3, 3 CSE→A5, 3 A5→CSE) | 14 | 0.55 — genuinely ambiguous |
| RI mixups | 3 | — |

The two buckets have different signatures and need different fixes. Confident errors are a
**distribution** problem; ambiguous ones are a **label** problem.

### Hypothesis 1 — the NN/positive boundary sits inside the real positive distribution

Not named in the resume doc, and it follows directly from the code:

- `sim/select_events.py:167` gives every `nn_hard` negative a real catalogued AS event
  with `psi1 == psi2` **exactly** — dPSI = 0.
- `sim/select_events.py:136` gives every positive `|dPSI| ~ U(0.3, 0.8)`.
- The curated benchmark's own matched events have median `|dPSI| = 0.180`, and **81.6%
  fall below 0.30** (`resume-2.md:94-96`).

So the model is trained on a boundary at roughly dPSI ≈ 0.15 and four out of five real
positives land on the NN side of it. This predicts NN false positives that are *confident*
(observed: 0.71) and concentrated in A3/A5, whose minor branches carry the fewest reads
(observed: 10 of 13).

It also explains why the `--psi-dist beta` batch was null-to-negative: it moved the
positives down onto negatives still pinned at exactly 0, collapsing the margin without
moving the boundary. sim-CV fell 0.784 → 0.693, which is what an unlearnable boundary
looks like. **The two distributions have to move together.** That is a different
experiment from the one that failed, not a repeat of it.

### Hypothesis 2 — exactly one event per window is differential

`sim/abundance.py:110-114` draws one baseline Dirichlet **shared by both conditions** and
rescales only the labelled event's transcripts. Every other AS event in the gene is
present but perfectly non-differential. Real windows are not like that: curated ES windows
contain a mean of 2.4 catalogued CSE and 0.7 catalogued A5 events, and the ES windows the
model calls A5 contain **2.2** catalogued A5 events against 0.7 for those it calls CSE
(`runs/cse_a3_diagnosis.log`). The earlier `diagnose_cse_a3` run recorded this as a
refutation; on the A5 side the table supports it.

This is the low-confidence bucket, and it is the same *kind* of fix as the window change —
how the problem is framed, not how noisy it is.

## Stage 0 — split the benchmark (do first, blocks everything downstream)

Stratified 70/70 dev/test over the 140 events, gene-disjoint where the CSVs allow, fixed
by seed, **committed as a file** (`benchmark/split.tsv`) so it cannot drift. Every tuning
decision below is read off dev only; test is scored once per milestone and reported
separately. This settles both open honesty caveats: the NN operating point and anything
touching `DPSI_BETA` become dev-set choices rather than test-set fitting.

Cost, stated up front: McNemar on 70 events detects a swing of roughly 8-10 events instead
of 5-6. Several past interventions would have been unmeasurable at that power. The
alternative is that no future number is clean.

*Tier: judgement (stratification + gene disjointness with only 140 events) → strong model.*

## Stage 1 — local, no new simulation

Each step: train → sim-CV → score dev → McNemar vs the current best → commit.

1. **`--feature-subset all` under v3 windows.** Never run: the only `all` run in `runs/`
   is on v2 windows and was never benchmark-scored. The default `no-geometry` exists to
   suppress a v1-window artifact that v3 deliberately re-legitimized — and geometry alone
   recovers 13/17 curated IR events on real data. One training run, one score. Highest
   value-per-hour in the plan.
2. **Fix model selection in `train.py:145-153`.** The best epoch is currently chosen on the
   held-out fold, which is the fold being reported. This inflates sim-CV and makes it
   useless as the steering signal it is being used as. Carve an inner validation split out
   of each training fold. Expect reported sim-CV to *drop*; that is the point. Re-baseline
   everything against the new protocol before comparing anything else.
3. **Two-head model.** Split the head into (a) structural type over {A3, A5, CSE, RI} and
   (b) a binary is-differential output; predict NN when (b) says not differential.
   Both labels already exist in the design (`psi1`/`psi2` in `design.tsv`). This separates
   "what structure is here" — which the simulation gets right and which is depth-robust —
   from "did it change", which is where the whole domain shift lives, and lets the NN
   operating point be chosen on the deployment argument (inputs are pre-filtered
   rMATS/SUPPA candidates) rather than fitted. Directly addresses hypothesis 1 without
   waiting on pasteur.
4. **Condition-swap TTA.** `augment_condition_swap` is already label-invariant and
   implemented; averaging the two orientations at inference is a few lines and free.
5. **Set the NN operating point on principle, on dev.** Use `dsignal/prior_sweep.py`
   (one extraction pass, N settings evaluated analytically). Write the deployment argument
   down *before* looking at the numbers.

*Tier: 1, 4, 5 mechanical → cheaper model. 2 and 3 are architecture/protocol → strong model.*

## Stage 2 — the crossover, queued on pasteur behind Stage 1

Render the 4,800 v3 windows with `plot2s.py` on pasteur, train ResNet50 locally in `DSv1`,
score the same benchmark. No re-simulation. This is a **diagnostic**, not an improvement:
it separates "v2's simulation is wrong" from "signal-native is a fragile representation".
It stays at priority 2 rather than 1 because Stage 1 targets 27 of the 30 known errors
with a specific mechanism, and the render is a long CPU job that can sit in the queue
while Stage 1 runs locally.

Consider `master`'s `--shard` for the render — it exists now and `report.md` is the profile.

## Stage 3 — one re-simulation batch, axes chosen by the error structure

Not the audit's over-injection shortlist; the resume doc's multi-axis batch, re-aimed:

1. **Move positives and negatives together** (hypothesis 1). Positives draw `|dPSI|` from
   `DPSI_BETA`; `nn_hard` draws a *small non-zero* null `|dPSI|` representing technical and
   biological variation rather than exactly 0. The null magnitude should come from
   replicate-level variation in the simulation itself, **not** from the benchmark — that
   keeps the axis honest and is measurable from existing `psi_recovery_*.tsv`.
2. **Per-class effect size.** The pooled beta draw helped A3/A5 and hurt RI; a retained
   intron at |dPSI| ≈ 0.1 carries almost no signal. One distribution does not fit four
   classes.
3. **Differential co-occurrence** (hypothesis 2). Perturb a random subset of the gene's
   *other* catalogued events across conditions too, with their own dPSI draws, keeping the
   label as the curated one. Small change in `sim/abundance.py`; makes the training
   distribution structurally match what real curated windows look like.

Re-run `dsignal.audit_noise` afterwards — it is the acceptance test, cheap and
annotation-free (`resume-2.md:132`).

*Tier: strong model throughout — this is simulation design.*

## Explicitly not in this plan

More scale-up (memory: stop scaling the simulation); more noise layers; Phase-4
fine-tuning; the audit's remaining over-injection knobs as single-axis experiments — that
premise is 0 for 2.

---

# Verification

**Part A.** No behaviour changes except the two v1 `--help` strings, so verification is
consistency, not tests:

- `python -m dsignal.train --help`, `... predict --help`, `... score_benchmark --help`,
  `... audit_noise {measure,compare,padding} --help`, `python dsEnsemble.py --help`,
  `python deepSpecas.py predict --help` — every flag and default quoted in a doc must
  match its output.
- `grep -rn "0\.1x-10x\|log-uniform\|46\.3%\|46\.5%\|62\.1\|41/140\|1260\|scale1260" dsignal/ sim/ *.md`
  returns only intentional history, each with its retraction adjacent.
- Every file path named in a doc exists (`plans/ensemble-inference-runtime.md` currently
  does not).
- Smoke-run the two repro commands added to `resume-2.md` and confirm 78.6%.

**Part B.** Every step:

1. `python -m dsignal.train --npz-dir … --labels … --json-out runs/<name>.json --out-dir runs/<name>` → sim-CV macro-F1;
2. `python -m dsignal.score_benchmark --net runs/<name>/fold*.pt --out runs/benchmark_events_<name>.tsv` → **dev half only**;
3. McNemar on the paired per-event outcomes vs the current best (`runs/benchmark_events_geomwin.tsv`), reported with the p-value — 4-6 events is noise at n=140 and 8-10 at n=70;
4. `python -m dsignal.verify_features` and `python -m dsignal.verify` still pass after any
   feature change; `dsignal.audit_noise compare` after any re-simulation;
5. commit before starting the next step.

Test half is scored once at the end of Stage 1, once at the end of Stage 3, and reported
as the headline. Anything read off dev is labelled as such.

# Critical files

`CLAUDE.md`, `README.md`, `report.md`, `benchmark/README.md`,
`plans/signal-native-v2-{checkpoint,resume-2}.md`, new `plans/README.md`;
`dsignal/{baseline_gbm,train,dataset,features,predict,prior_sweep,score_benchmark,audit_noise,tabulate_runs,verify_features,diagnose_cse_a3}.py`;
`sim/{window,width_leak_check,gene_neighbors,select_events,abundance,psi_recovery_check}.py`;
new `benchmark/split.tsv`.

Reuse rather than rewrite: `dsignal.baseline_gbm.{grouped_folds, report,
report_collapsed, print_vs_reference}` (the shared eval harness — route any new model
through it), `dsignal.predict.{Predictor, PriorCorrection, make_target_prior}` (prior
correction is applied per fold *before* averaging and that is non-negotiable),
`dsignal.prior_sweep` (one extraction pass, N calibration settings),
`dsignal.dataset.{apply_feature_subset, apply_feature_subset_meta}` — note the geometry
group lives as private names in `baseline_gbm.py` and is imported by `dataset.py:248`, so
a feature-subset change touches three call sites and `apply_feature_subset_meta` is the
easy one to forget.
