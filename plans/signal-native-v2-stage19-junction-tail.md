# signal-native-v2 — Stage 19: re-simulate the junction tail (plan + pre-registration)

Approved 2026-09-23; implementation happens in a separate session. **Everything under "Pre-registered gates" is fixed at commit and must not be edited after any pasteur spend.** Results are appended below; retractions go next to what they retract.

## Context

Stage 20.3 (real fine-tuning) and Stage 22b (SSL pretraining) both failed their gates, so real-signal training interventions are closed twice. Stage 19 is the last item on the accuracy plan's evidence-order list.

Stage 18.1 found the real/sim gap is diffuse (discriminator AUC 0.999, still above 0.98 after 30 feature families are removed). Its largest single tell is the **sparse junction tail**, not coverage shape. Stage 19 attacks that axis from the simulation side.

### Measured this session, read-only

Source: a scratch script over committed npz, n ≤ 2000 per set. "Tail" means junctions with ≤ 3 reads summed over both conditions.

| set | junctions/window, median | tail jx/window, median | tail frac | one-cond frac | jx/kb | tail/kb | depth median |
|---|---:|---:|---:|---:|---:|---:|---:|
| aqr | 18 | 11 | 0.69 | 0.65 | 4.83 | 3.12 | 16.5 |
| prpf4 | 9 | 5 | 0.53 | 0.46 | 2.67 | 1.16 | 7.2 |
| u2af2 | 15 | 9 | 0.63 | 0.57 | 4.12 | 2.23 | 20.5 |
| ptbp1 | 16 | 9 | 0.63 | 0.55 | 4.83 | 2.53 | 22.9 |
| bnew_dev | 15 | 9 | 0.63 | 0.57 | 4.37 | 2.29 | 20.5 |
| tra2 | 13 | 7 | 0.58 | 0.52 | 4.00 | 2.07 | 15.1 |
| **union2 (sim)** | **6** | **0** | **0.10** | **0.09** | 2.89 | **0.00** | 86.2 |

Structure of the real tail junctions (aqr / u2af2 / ptbp1 / bnew_dev; union2 in brackets):

| property | real | union2 |
|---|---|---|
| lie fully inside the window | 13–25% | [24%] |
| median intron length | 15–120 kb; q75 170–240 kb | [16 kb; q75 100 kb] |
| canonical motif | 58–70% | [77%] |
| share a donor or acceptor with an expressed junction | 22–36%; the rest have both ends novel | — |
| anchor < 10 bp | 10–12% | — |

- **Tail count barely scales with depth.** Tail junctions per window are roughly constant (5–9), while tail per 1k reads falls about 40× from the lowest to the highest depth tercile. The tail therefore behaves as a per-window count, not a per-read error rate.
- **Most real tail junctions span the window.** They are long-range, often non-canonical splices with one anchor in the window.

### Why the simulator has no tail

- Junctions beyond the event come from three places: other annotated isoforms, **one** novel intron per gene (a copy of the whole transcript at 0.5–3%, placed anywhere in the gene), and neighbour genes (`--flank 0`).
- All three use the **same expected abundance in both conditions** (`sim/abundance.py:198-220`).
- Nothing is condition-specific, and nothing is long-range or spurious.

### Two corrections to the accuracy plan's premises

1. **The pasteur cost is about 8–10 h, not 25 h.** union2 batches measured about 37 min each on 2026-08-21 (polyester ~20 min, STAR ~13 min). The "~25 h" figure is an unmeasured pre-run estimate of 13 × 2 h. This plan budgets explicitly against a 25 h hard cap.
2. **STAR flags are identical on sim and real.** They cannot explain the gap; the tail has to be generated.

### User decisions (this session)

- **Gate power.** The gate is seed-pooled: a 3-seed mean-posterior McNemar, plus sign consistency in every seed.
- **Tail model.** An empirical tail layer, fit only on gene-disjoint training cohorts (aqr/prpf4/u2af2).
- **Scope.** Tail layer + `@RG` + an indel-capable read post-processor.
  - No polyester GC or positional bias.
  - Replicate and read-QC features are **not** part of this stage. The re-sim only makes them possible, and new BAMs are kept for that.

## Design

### Identity principle: tail and indels are additive to the existing realisation

- The union2 design runs unchanged: 13 batches, same `design.tsv`, count matrix and polyester seed. Base reads therefore regenerate bit-identically.
- Tail reads come from a **second** polyester call:
  - its own FASTA and count matrix;
  - seed `batch_seed + 100000`;
  - read headers renamed `tail<k>` so names never collide with `read<k>`.
- Tail reads are concatenated onto each sample's FASTA pair before STAR.
- Old npz vs new npz on the same 12,480 events is then a clean ablation. Its only differences are the tail, the indels, and `@RG` (which the extractor ignores).

### Tail generator (`sim/build_tail_transcripts.py`, new)

**Fit step.** `sim/fit_tail_model.py` reads aqr/prpf4/u2af2 pos+nn npz only. It never reads ptbp1, which shares genes with bnew dev, and it never reads bnew_dev or tra2. It writes `sim/tail_model.json` with:
- per-kb tail rate: an empirical quantile table of tail junctions per kb per window;
- P(inside window) — the alternative is spanning, with one end outside;
- log-intron-length histograms for inside and spanning junctions, separately;
- P(canonical motif);
- P(shares an expressed or annotated site);
- the 1/2/3-read split.

This keeps ptbp1-free provenance, so models trained on the new pool stay eligible for the `benchmark_new` test spend.

**Generation.** For each window (design and neighbour rows alike, label-blind), in its own RNG stream `SeedSequence([seed, 19])`:
1. Draw N = round(rate_kb × window_kb × `TAIL_RATE_SCALE`).
2. For each tail junction, pick the in-window anchor end.
   - With P(shared), use an annotated exon boundary inside the window, from `gtf_exon_structure`.
   - Otherwise, use a uniform position.
3. Pick the partner end: in-window, or at the sampled intron length in a random direction.
4. With P(canonical), snap both ends to the nearest GT…AG (or CT…AC) within ±30 bp in the reference FASTA.
5. Build the transcript as `ref[d-250:d] + ref[a:a+250]`: a 500 bp sequence with the junction at its centre.
6. Draw total fragments T from the fitted 1/2/3 split, and assign each fragment to a uniform random one of the 6 samples.
   - One-condition-only presence then follows naturally, and is checked rather than forced.

The only calibration knob is `TAIL_RATE_SCALE` (default 1.0; an env var in `run_batch.sh`). It absorbs the loss of short-anchor or non-canonical reads that STAR soft-clips instead of splicing.

### Indels (`sim/inject_indels.py`, new)

- A seeded per-sample post-processor over the polyester FASTA pairs, covering base and tail reads.
- Deletions and insertions are injected at per-base rate `INDEL_RATE`, with lengths Geometric(p = 0.6) capped at 5.
- `INDEL_RATE` is fit once from the pooled `Σdelcov/Σcov` of the same three training cohorts. The insertion rate is set equal to it, which is documented as an assumption.
- PCR duplicate copies receive independent indels. This is a known minor inaccuracy, recorded.
- Defaults to off, so the old behaviour is byte-identical.

### `@RG`

- `sim/align_reads.sh:53-61`: add `--outSAMattrRGline ID:${name} SM:${name}`, matching `sim/fetch_align_benchmark_new.sh:73`.
- The existing `samtools merge` keeps the per-record `RG:Z` tags.
- **BAMs are kept on pasteur this time**: roughly 2–3 GB per batch, about 35 GB total, against ~870 GB free. Reads FASTA is deleted after alignment.

### `b1`–`b5` (geomwin) caveat

- These batches' reads came from the old scale-up pipeline, and their `design.tsv` holds v2 windows. `design.geomwin.tsv` is the one that was extracted.
- `run_batch.sh` gains `DESIGN_EXTRACT=design.geomwin.tsv` for them.
- The pilot's identity check (P0) runs on b1 as well as b12 for exactly this reason.

## Steps

| step | what | tier | cost |
|---|---|---|---|
| 19.0 | Pasteur inventory, read-only: which of `design*.tsv`, `count_matrix.tsv`, `transcripts.ordered.fa`, `novel.fa`, `premrna.fa` survive per batch (b1–5, b8–15); reference FASTA path used by the STAR index; disk; `/proc/loadavg`. Record in the plan doc. | mechanical (Haiku/Sonnet) | minutes |
| 19.1 | `sim/tail_stats.py`: promote the scratch analysis to a committed CLI (per-window tail stats + structure table, per class); `sim/fit_tail_model.py` → `sim/tail_model.json`; fit `INDEL_RATE` | Opus (distribution choices, provenance) | ~half day |
| 19.2 | `sim/build_tail_transcripts.py` (FASTA + 6-column count matrix, own RNG stream, canonical snap, strand handling); `sim/inject_indels.py` | Opus (coordinate/strand semantics) | ~1 day |
| 19.3 | Wiring. `run_batch.sh` gets env `TAIL=0/1`, `TAIL_RATE_SCALE`, `INDEL_RATE=0`, `DESIGN_EXTRACT`, plus the second polyester call, header rename and concat step. `align_reads.sh` gets the `@RG` line. Defaults must reproduce old batches exactly. | Sonnet | ~half day |
| 19.4 | Measurement tooling. `audit_noise.METRICS` gains `n_junctions`, `n_tail_junctions` (≤ 3 reads), `one_cond_frac`, `tail_per_kb` (today's metrics only see ≥ 5 reads, so they are blind to the tail). `dsignal.benchmark_split` gains a `pool` subcommand: mean `p_*` over N scoring TSVs → pooled TSV (same columns), so seed-pooled McNemar runs through the existing `compare`. | Sonnet | ~half day |
| 19.5 | **Pilot + Gate A**, on pasteur. Rsync the new modules first. P0 = b12 and b1 with tail off, indels off, `@RG` on. P1 = b12 with tail on and indels on. At most one recalibration (P1′), changing only `TAIL_RATE_SCALE`. Score Gate A locally. | mechanical launch; Opus write-up | ≤ ~3 h pasteur |
| 19.6 | Full re-sim of the 12 remaining batches with P1's knobs (the accepted P1 b12 is reused). Run under the babysit watcher, one batch at a time, 8 threads, `limitram 48G`, four thread-pool vars, `nice -n 19`, loadavg before and a minute after. | mechanical | ~8–9 h pasteur |
| 19.7 | Merge (`merge_batches`), verify, identity diffs; train arm **T** (3 seeds) and, conditionally, arm **T+D** (3 seeds) | mechanical | ~3–6 h local GPU |
| 19.8 | Score all evals; Gate B; domain probe on the full pool; decision doc; `resume-11.md`; update CLAUDE.md headline only if PASS | Opus | ~1 day |

### Pasteur budget

**Hard cap: 25 h cumulative wall clock for Stage 19.**

| item | expected |
|---|---:|
| P0 (b12 + b1) | 1.3 h |
| P1 | 0.8 h |
| P1′ (if needed) | 0.8 h |
| 12 full batches | ~9 h (≈ 45 min each; the tail adds ≤ 5% fragments and one short polyester call) |
| **expected total** | **≈ 12 h** |

Abort rules:
- **Any batch over 90 min:** stop and investigate before continuing.
- **Cumulative time reaches 20 h** with batches remaining: stop and report. Do not finish on the reserve.
- Only npz, labels, design and tail tables come back from pasteur, as always.

## Pre-registered gates (fixed at commit, before any pasteur spend)

### Gate A — realism, on pilot b12 (960 events). All of A0–A4 must hold, or the full re-sim does not run.

**A0 — identity.**
- P0's b12 and b1 npz must equal the local old npz array-for-array: `cov_*`, `delcov_*`, `junctions_*`, and the `stats_*` counts.
- The regenerated `labels.tsv` must be byte-identical.
- Failure is a pipeline bug to fix, not a Gate A FAIL.

**A1 — tail realism.** On P1 b12, measured with `sim/tail_stats.py`, each statistic must fall inside the **[min, max] across the three fit cohorts** (aqr/prpf4/u2af2):

| statistic | required range |
|---|---|
| tail/kb (median) | 1.16–3.12 |
| tail fraction | 0.53–0.69 |
| one-condition fraction of all junctions | 0.46–0.65 |
| tail junctions fully inside the window | 0.13–0.25 |
| canonical fraction of the tail | 0.58–0.70 |

Not gated, because they are depth- or width-confounded and depth is deliberately not changed:
- junctions/window;
- mean per-junction count.

**A2 — the probe moves.** Build `dsignal.domain_probe` twice, with the real pool vs b12-old and vs b12-new, class-matched. For each of the four 18.1 tail tells, |MW-AUC − 0.5| must at least **halve**:

| tell | MW-AUC on b12-old |
|---|---:|
| `mean__log1p_count_2` | 0.026 |
| `log1p_n_junctions` | 0.853 |
| `mean__log1p_min_anchor` | 0.974 |
| `wmean__present_2` | 0.062 |

Pooled AUC is reported but not gated. Stage 18 predicts it stays above 0.9, because the gap is diffuse.

**A3 — label blindness.** A 5-fold LightGBM on tail-only features (per-kb tail count, one-cond fraction, inside fraction, canonical fraction), predicting the class on P1 b12, must reach **balanced accuracy ≤ 0.30** (chance is 0.20). The tail must not become a label shortcut.

**A4 — the standing gates pass on P1 b12:**
- `dsignal.verify`;
- `sim.psi_recovery_check`, event PSI unaffected;
- `sim.width_leak_check`;
- `dsignal.verify_features`.

**If Gate A fails:**
- **A1 or A2:** one recalibration (P1′) is allowed, changing `TAIL_RATE_SCALE` only.
- **A second failure:** Stage 19 **stops before the full re-sim**. The verdict is "the junction tail is not simulable with this generator", at ≈ 3 h pasteur cost.
- **A3 failing:** a design flaw. Stop and re-plan; no recalibration.

### Gate B — accuracy, after the full re-sim

**Arm T** is the only arm that decides the verdict.
- Pool: new `npz_union2_tail` (the same 12,480 events).
- Recipe: identical to `union2_base_s{0,1,2}` — `--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30`, seeds 0/1/2.
- Scoring: `--tta-condition-swap --target-prior nn-weight --nn-weight 0.1`, with `--train-labels` set to the new labels file.
- Base: the already-committed `runs/{bnew_dev,benchmark_events,arena}_union2_base_s{0,1,2}*.tsv`. These are not rerun.
- **"Pooled"** means: the mean `p_*` over the 3 seeds' scoring TSVs, then argmax. Arm and base are pooled identically, giving two 3-member ensembles.

Arm T **PASSES** iff all hold:

| rule | requirement |
|---|---|
| B1 — primary | On `benchmark_new` dev (242), pooled arm collapsed accuracy > pooled base, by paired McNemar with p < 0.05. |
| B2 — consistency | On `benchmark_new` dev, arm collapsed > seed-matched base collapsed in **all 3 seeds**, as a point estimate with no significance requirement. |
| B3 — guard | On `benchmark/` dev (70), pooled arm − pooled base ≥ −5, **and** no seed falls more than 5 below its seed-matched base. |
| B4 — per-class | **All five classes, NN included by name**. On `benchmark_new` dev pooled, no class's recall drops with per-class McNemar p < 0.05. Also, no class drops by ≥ 3 events in all 3 seeds (Stage 20's count rule). |
| B5 — confirmatory guard | On `ptbp1_arena` (2,422), pooled arm collapsed does not regress with McNemar p < 0.05. |

- **Validity preconditions.** A failure here voids the run rather than failing it:
  - labels, windows and `ix` identical to `labels_union2.tsv`;
  - `dsignal.verify` clean on the full pool;
  - `psi_recovery_check` unchanged.
- **Reported for every run, not gated:**
  - per-seed McNemar;
  - in-sim macro-F1;
  - curated-NN recall;
  - the 18.1 probe on the full new pool (AUC plus the removal curve) — the standing metric.

**Arm T+D** (exploratory; does not decide the verdict).
- Same as T, plus `--event-groups delcov`.
- Trained **only if** the full-pool probe puts delcov's MW-AUC in [0.35, 0.65]; it was 0.84 on the old sim.
- Scored against **T** under the same B1–B5. Its result decides only whether delcov may enter future recipes.

### What each outcome decides

- **T PASSES:**
  - Retrain the `e4_union2` members on the new pool and score them as an ensemble on dev.
  - Replicate and read-QC features (the kept `@RG` BAMs) get their own plan.
  - The `benchmark_new` test spend is planned next, as the programme's final measurement. The new pool is ptbp1-free, so it is eligible.
- **Gate A passes, T FAILS:** "tail realism does not transfer."
  - The accuracy plan's evidence-order list is exhausted.
  - The remaining work is the final test-spend plan with the best dev-selected existing model.
  - 22a stays unrun: no coverage-bias sim exists.
  - The kept BAMs remain the substrate for any future replicate-feature plan.
- **Gate A fails twice:** the same as the previous outcome, with the "not simulable" verdict and no full-re-sim cost.

### Deliberately not fixed here

- Exact `tail_model.json` values. They are fit mechanically in 19.1, by a committed script from committed npz.
- Anything about replicate or read-QC features.
- Ensemble composition after a PASS.

## Verification

- 19.1: the fit script is deterministic, and its output is committed.
- 19.2: `build_tail_transcripts` round-trip checks:
  - every emitted junction's coordinates are re-derivable from the FASTA;
  - canonical snaps really are GT/AG in the reference;
  - no tail transcript overlaps an event's own junction coordinates exactly.
- 19.3: defaults reproduce old outputs (proven by A0 on the pilot).
- 19.4: `benchmark_split pool` on a single TSV reproduces that TSV's `pred`/`correct` exactly.
  - Pooling the three base seeds is sanity-checked against a manual mean.
- Throughout:
  - build → verify → commit per step;
  - pasteur discipline (`limitram`, four thread-pool vars, `nice -n 19`, loadavg; never `pkill -f` without `[p]`);
  - `benchmark_new` test untouched;
  - never delete a measured number.

## After approval (this session)

1. Copy this plan verbatim to `plans/signal-native-v2-stage19-junction-tail.md`.
2. Add it to `plans/README.md`, and add a one-line pointer in `resume-10.md`'s "Next" section.
3. Commit it. That commit is the pre-registration timestamp.
4. Stop. No implementation in this session.
5. Hand the user the kickoff prompt for the implementation session.

## Critical files

- **New:**
  - `sim/tail_stats.py`, `sim/fit_tail_model.py`, `sim/tail_model.json`
  - `sim/build_tail_transcripts.py`, `sim/inject_indels.py`
  - `plans/signal-native-v2-stage19-junction-tail.md` (this doc), `plans/signal-native-v2-resume-11.md`
- **Modify:**
  - `sim/run_batch.sh` (knobs; second polyester call around the Rscript at :192-194; FASTA concat; `DESIGN_EXTRACT` at :148-156,208-212)
  - `sim/align_reads.sh:53-61` (`@RG`)
  - `dsignal/audit_noise.py:429-453` (`METRICS`)
  - `dsignal/benchmark_split.py` (`pool` subcommand next to `cmd_compare` :223)
- **Reuse, unchanged:**
  - `sim/run_polyester.R`: the second call uses the same script and args.
  - `sim/merge_batches.py`, `dsignal/domain_probe.py` (`build --sim-npz`, `real-vs-sim`), `dsignal/train.py`, `dsignal/score_benchmark.py`
  - `sim/gtf_exon_structure` tables (annotated boundaries)

---

# Results

## 19.0 — pasteur inventory (2026-09-23, read-only)

**The per-batch inputs the plan assumed are gone.** `$ROOT/sim/batches/` (where
every `run_batch.sh` / `run_b1{2..5}.sh` wrote) no longer exists on pasteur, so
no batch has a surviving `count_matrix.tsv`, `transcripts.ordered.fa`,
`novel.fa`, `premrna.fa`, `neighbors.tsv`, reads or BAMs. The mirror dirs
`sim/experiment/batches{,_geomwin,_bag2,_union2}/b*` on pasteur hold only an
empty `npz/` (and `logs/` for b12–15). No plan doc records the deletion.

What survives, and is enough to regenerate every batch deterministically:

| item | where |
|---|---|
| `design.tsv` for all 13 batches | local `sim/experiment/batches/b1–5` (v2 windows, the one reads were simulated from), `batches_bag2/b8–11`, `batches_union2/b12–15` |
| `design.geomwin.tsv` for b1–5 | local `sim/experiment/batches_geomwin/b1–5/design.tsv`; differs from `batches/b*/design.tsv` **only** in `win_start`/`win_end` (cols 9–10), psi columns included — checked |
| seeds | b1–5 = 1–5 (`sim/regen_designs.sh`), b8–11 = 8–11, b12–15 = 12–15; all default `run_batch.sh` knobs |
| b12–15 step logs (transcript count, total fragments) | local `batches_union2/b1*/logs/` — a cheap identity check on the regenerated count matrix before polyester (b12: 13,721 transcripts, 14,243,582 fragments) |
| reference | `ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa` (+ `.fai`), Ensembl 99 GTF; the STAR index (`star_index/`, STAR 2.7.4a, `--sjdbOverhang 100`) was generated from exactly these |
| envs | `DSpoly2v2` (STAR-avx2, samtools), `DSsim` (Rscript, polyester **1.9.7** = the ASimulatoR fork), `DSrender` |
| `sim/events`, `gene_spans.tsv`, `transcript_lengths.tsv` | present |

Consequence: every batch re-runs `run_batch.sh` steps 2–4 (neighbours, exon
structure, novel junctions, count matrix, FASTA; minutes each) in front of
polyester, with its local `design.tsv` uploaded so step 1 is skipped. This does
not change the plan's cost materially, but it makes A0 a test of the *whole*
deterministic chain, not just polyester/STAR. None of the sim files touched
since b1–5 were simulated (2026-08-15/16) changes their count-matrix path at
default knobs (`38114e2bd` is `select_events` only; Stage 11's knobs default
to the old behaviour); A0 on b1 is what proves it.

Machine state: `/data` 999 G free of 9.1 T (90% used); 64 cores, 251 G RAM;
load average 28.8 at inventory time (someone else's jobs). No reference FASTA or
GTF exists locally (and FASTA never comes back from pasteur), so local tests of
the tail generator use a public Ensembl-99 chr21/22 download.

### Two things the inventory's data pass found (read-only, local npz)

1. **Most real tail junctions have *both* ends outside the window**, not one.
   `bam.fetch` returns a read whose N-op merely spans the window, so a window
   sitting inside a long spurious intron sees that junction. Pos+nn, tail = ≤ 3
   reads, intron boundaries against the window's 1-based (start, end]:

   | cohort | inside | one end out | both out | canonical (inside / one / both) | median intron (inside / one / both) |
   |---|---:|---:|---:|---|---|
   | aqr | 0.234 | 0.178 | 0.589 | 0.88 / 0.74 / 0.54 | 1.6 k / 10.7 k / 147 k |
   | prpf4 | 0.335 | 0.211 | 0.454 | 0.93 / 0.88 / 0.74 | 1.8 k / 8.1 k / 68 k |
   | u2af2 | 0.241 | 0.215 | 0.544 | 0.85 / 0.81 / 0.67 | 1.7 k / 10.6 k / 151 k |

   The generator therefore models three location classes, not two (see 19.2).
2. **Gate A1's "inside" and "canonical" ranges were not computed on the set the
   gate names.** The first three A1 statistics (tail/kb 1.16–3.12, tail
   fraction 0.53–0.69, one-condition fraction 0.46–0.65) reproduce exactly from
   aqr/prpf4/u2af2 **pos+nn** (median per-window tail/kb; pooled tail fraction;
   per-window-mean one-condition fraction). The last two (inside 0.13–0.25,
   canonical 0.58–0.70) reproduce only when the cohort glob also sweeps in the
   `real_ssl_*` random windows and ptbp1/bnew_dev (aqr+ssl 0.117/0.572,
   u2af2+ssl 0.140/0.682, ptbp1 0.238/0.674, bnew_dev 0.207/0.703). On the fit
   cohorts' pos+nn — what the gate text says, and what the tail model is fit on —
   the ranges would be inside **0.234–0.335** and canonical **0.655–0.835**.
   The gate is **not edited**. This is raised with the user before any pilot
   spend, because a faithful fit (pooled inside ≈ 0.25, canonical ≈ 0.72
   before STAR's non-canonical losses) sits on or past the frozen upper bounds.
