# signal-native-v2 — START HERE (2026-08-17)

Resume document. Branch `signal-native-v2`, clean tree, nothing pushed (per repo
convention — pushing is the user's call).

`CLAUDE.md` now covers both lines of work and maps `dsignal/` + `sim/`; `README.md`
is deliberately v1-only (v2 is unpublished). `plans/README.md` indexes this
directory.

Supersedes `plans/signal-native-v2-resume.md` (2026-08-16), which is now
history: its two headline findings were artifacts of a bug this session fixed.
`plans/signal-native-v2-audit-geometry-results.md` is the detailed report for
that session; this file is the short version plus direction.
`plans/signal-native-v2-checkpoint.md` remains the build-mechanism reference.

---

## The headline number, as of the end of Stage 1 (2026-08-17)

**58/70 = 82.9% collapsed on the held-out test half**, scored once, with every
choice in the configuration made on dev: `runs/geomwin_iv30_refit` (5 folds,
honest epoch selection, refit on the full fold) + condition-swap TTA + the
pre-registered `nn_weight` 0.1. Dev, the half that was looked at, gives 53/70 =
75.7% — the 7-point spread between halves is what n=70 noise looks like and is
the reason both are reported. Command:

```bash
python -m dsignal.score_benchmark --net runs/geomwin_iv30_refit/fold*.pt \
  --half test --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \
  --train-labels sim/experiment/labels_geomwin.tsv
```

**The old 78.6% is not comparable and should not be quoted.** Its epoch was
chosen by peeking at the fold it then reported (Stage 1.2), so it was selected
optimistically; the honest re-baseline of that same model is 49/70 on dev.

## The one thing to know

**Zero-shot transfer went 62.1% → 78.6% collapsed, and the cause was the
extraction window, not the noise model.** Same simulated reads, same
alignments, same 4,800 events and genes, same model and hyperparameters — only
the window definition changed. No re-simulation. *(The 78.6% here is the
optimistically-selected number; the finding — that the window, not the noise
model, is what moved transfer — stands, and the ~16-point size of it is far
larger than the ~2-4 points the selection protocol is worth.)*

| | v1 | v2 before | v2 now |
|---|---|---|---|
| collapsed accuracy | **87.9%** | 62.1% | **78.6%** (79.3% prior-corrected) |
| A3 / A5 / ES / IR | 75 / 90 / 91 / 94 | 46 / 50 / 74 / 65 | 82 / 70 / 80 / 82 |
| NN false positives (of 140) | 2 | 32 | 13 |
| sim-CV macro-F1 | — | 0.800 | 0.867 |

Significant: 27 events gained against 4 lost, **McNemar p = 3.4e-05**. Gap to
v1 is now 8.6 points, was 25.8.

v1's 87.9% was reproduced locally on 2026-08-16 (`runs/v1/pred_*.csv`). v1 is
also simulation-only, with a cruder simulator, and cannot separate A3 from A5
(`--a --cse`), so the collapsed number is the fair comparison.

## Two findings from the old resume doc that did not survive

Both were measured under the broken windows. Do not carry them forward.

1. *"More simulated data makes transfer worse."* Under v3 windows, 4,800 events
   beat 960 (78.6% vs 72.9%, p = 0.169 — not significant, but the sign flipped).
2. *"sim-CV is anticorrelated with transfer."* They moved together for the
   first time this session (0.800 → 0.867 alongside 62.1% → 78.6%).

## Why the window was wrong

`benchmark/events.bed` is byte-identical to the region CSVs, and matching all
140 curated windows against the SUPPA catalogue the simulation draws from
(`python -m dsignal.audit_noise padding`) shows the curated window **is** the
event span plus a per-type constant: ES +100 bp/edge, IR exactly 0, A3/A5 −1
(an off-by-one convention). Median event span / window width = **1.000**;
120/125 matched windows agree on both edges within 200 bp.

The old sampler matched the benchmark's *marginal width distribution* while
destroying that relationship — the simulated event filled a label-dependent
4–98% of a window ~2× too wide. `sim/window.py` v3 draws
`win = event span + U[0,100] bp per edge`, label-independently.

**Accepted cost, reported not suppressed.** This reintroduces a width→label
coupling: sim-CV geometry-only rose 0.360 → 0.486, LightGBM 0.319 → 0.446,
width-only tree 0.279 → 0.349. Justified because the benchmark has the same
coupling — the geometry-only control scored *on the benchmark* gets 31.4%
5-class / 35.0% collapsed (above a ~20% random floor, below the 46.4%
always-CSE baseline), and recovers **13/17 curated IR events from geometry
alone**. Not a strong general signal; a real RI one. The shipped model trains
`--feature-subset no-geometry`, which zeroes those columns anyway.

## What was tried and failed

Two well-motivated interventions, both null. Volume-matched controls, McNemar
on the 140 paired outcomes.

| change | set | sim-CV | benchmark | vs control |
|---|---|---|---|---|
| *(control)* v3 windows | 4,800 | 0.867 | 78.6% | — |
| `--feature-subset all` | 4,800 | 0.873 | 74.3% (dev) | p = 0.549 |
| `--select-on inner` (protocol fix) | 4,800 | 0.854 | 68.6% (dev) | p = 0.065 |
| `--select-on inner --refit` | 4,800 | 0.853 | 70.0% (dev) | p = 0.109 |
| `--inner-val-frac 0.30` | 4,800 | 0.862 | 72.9% (dev) | p = 0.727 |
| condition-swap TTA | — | — | 72.9% (dev) | p = 0.625 |
| pre-registered `nn_weight` 0.1 | — | — | 72.9% (dev) | p = 0.500 |
| **all three stacked** | 4,800 | 0.862 | **75.7%** (dev) | p = 0.344 |
| `--two-head` | 4,800 | 0.858 | 67.1% (dev) | p = 0.727 |
| `--two-head` stacked | 4,800 | 0.858 | 75.7% (dev) | p = 1.000 vs stack |
| *(control)* b1 under the new protocol | 960 | 0.756 | 68.6% (dev) | — |
| Stage 3 batch (paired + nulls + co-occurrence) | 960 | 0.642 | 60.0% (dev) | p = 0.308 |
| `--depth-norm scale` | 4,800 | 0.868 | 77.9% | **p = 1.000** |
| `--subsample-low 0.15` | 4,800 | 0.862 | 73.6% | — |
| `--depth-norm drop` | 4,800 | 0.834 | 72.9% | — |
| *(control)* uniform \|dPSI\| | 960 | 0.784 | 72.9% | — |
| `--psi-dist beta` | 960 | 0.693 | 70.0% | p = 0.572 |
| beta + depth scale | 960 | 0.689 | 72.1% | p = 1.000 |

**Model selection was reading the fold it reported** (Stage 1.2). Until
2026-08-17 `train.py` chose the best epoch — and early-stopped — on held-out
fold macro-F1, then reported that same maximum. sim-CV is the steering signal
for every experiment here, so it had to be an estimate, not a best-of-N. The
default is now `--select-on inner`: a gene-disjoint 15% of each *training* fold
picks the epoch, the held-out fold is scored once. `--select-on held-out`
reproduces any older number.

sim-CV 0.867 → **0.854**, so the old protocol inflated by ~0.013 (and was
noisier fold to fold: spread 0.041 → 0.016). **Every sim-CV number on this
branch predating today reads about 0.013 high.**

The surprise: transfer fell as well, dev 55/70 → 48/70 collapsed (78.6% →
68.6%, p = 0.065; 5-class p = 0.022). That is a real change in the model, not
accounting — and it confounds two things the switch does at once, honest
selection *and* 15% less training data per fold. `--refit` (added the same day)
keeps the honest choice and retrains on the full fold for the chosen number of
epochs, which separates them.

**The refit control says the data is not it.** 49/70 vs 48/70 (p = 1.000),
sim-CV 0.853 vs 0.854. The whole transfer cost sits in the epoch choice: picking
on ~960 gene-disjoint held-out events tracked benchmark transfer better than
picking on ~576 inner-val ones. How much of that is a genuinely better selection
signal and how much is the optimism of taking a maximum, this experiment cannot
say — but **78.6% was selected optimistically on both axes, and the honest
re-baseline is ~70% on dev.** Every Stage-1 comparison from here goes against
`runs/geomwin_innerval_refit` (49/70 dev), not against 78.6%.

Worth trying before accepting the loss: a larger `--inner-val-frac` (the
selection signal is noisy at 576 events), or averaging the last k epochs instead
of picking one.

**The two-head model is a null, and hypothesis 1 is not supported** (Stage 1.3).
Splitting the head into a 4-way structural type and a binary is-differential
output was the plan's specific mechanism for the confident NN false positives:
the simulation trains positives at |dPSI| ~ U(0.3, 0.8) against negatives pinned
at exactly 0, 81.6% of real events sit below 0.30, so the boundary should be the
thing that transfers badly. It isn't the head that fixes it. sim-CV 0.853 →
0.858; dev uncorrected 49/70 → **47/70** with NN false positives going 7 → 13;
dev with TTA and the pre-registered prior 53/70 → **53/70**, four events
differing each way.

What the factorization *does* change is responsiveness: uncorrected it calls NN
about twice as often, and the prior then recovers 8 events (p = 0.016 against
its own uncorrected run) where the single head recovers 4. "Did this change" is
now one scalar a deployment argument can move — worth having, not an accuracy
win. The evidence points at the training distribution (Stage 3.1: move
positives and negatives together) rather than at another architecture.

`nn_hard` negatives supervise the type head too — their event_id carries the
SUPPA type, so a negative can teach "this is an A5 site" and "nothing changed"
at once, which the 5-way head cannot express. That part worked as designed; it
just did not buy transfer.

**Window geometry as a feature** (Stage 1.1, dev half). The `no-geometry`
default was set to suppress a v1-window artifact that v3 deliberately
re-legitimized, and `all` had never been benchmark-scored under v3. It buys
sim-CV (0.867 → 0.873) and loses transfer (55/70 → 52/70 collapsed, 7 lost / 4
gained, p = 0.549). The loss is **all RI**: 6/8 → 3/8, two RI events going to
A5. Geometry alone recovers 13/17 curated IR, so geometry carries real RI
signal — but it does not add: given both, the model leans on the *simulated*
width→label coupling, which is only partly the real one. Default unchanged, now
on evidence.

**Depth invariance.** Absolute depth is not a property of an event (the user's
point: the same event occurs at 5x and at 300x), and 16 features carried it.
`--depth-norm scale` maps every record onto a reference spliced depth and is
*exactly* invariant — max feature change 0.0 under 0.2x/5x rescaling. It
changes nothing on the benchmark, including on the shallowest depth quartile
where the mechanism predicted a gain (+2 of 35). Kept, non-default. The
argument for adopting it is design, not evidence: the benchmark is one dataset
at one depth and structurally cannot test what invariance protects against.
`drop` costs both sim-CV and transfer, so the model does use magnitude
productively; `scale` preserves that while removing the nuisance.

**Effect-size realism.** The measurement stands and is solid: **81.6% of
curated events have \|dPSI\| below 0.30**, the simulation's floor (real median
0.180 vs simulated recovered 0.415, n=125, identical SUPPA-junction logic both
sides). Fixing it traded per class rather than gaining — A3 57.1→64.3%, A5
collapsed 66.7→76.7%, ES 76.9→69.2%, IR 82.4→64.7%. A retained intron at
\|dPSI\|≈0.1 carries almost no signal, so RI degrades. If revisited, the draw
should be per class, not pooled.

## Stages 2 and 3 both landed, and together they settle the live question

**Stage 2, the representation crossover.** v1's images + ResNet50 trained on
*v2's* simulated data, scored on the same benchmark, against the *same*
representation trained on v1's own data (`data/trained_nets/squishstack`):

| trained on | dev | test | all 140 | NN false pos |
|---|---|---|---|---|
| v2's simulation | 75.7% | 72.9% | **104/140 = 74.3%** | 22 |
| v1's own simulation | 80.0% | 71.4% | **106/140 = 75.7%** | 10 |

Two events apart. **v2's simulation trains v1's representation as well as v1's
own data does**, so the simulation is not what limits transfer.

The second finding matters more: a *single* v1 representation scores 75.7%, not
87.9%. **87.9% is the six-net ensemble.** The fair comparator for a single v2
model was never 87.9% — it is ~76%, which v2 already matches (dev 75.7%, test
82.9% at the Stage 1 best). Most of the "8.6-point gap" this branch has chased
is ensembling, not representation and not simulation realism.

*Caveats:* the crossover net saw 4,522 of 4,800 events (windows >50 kb capped,
a few crash `plot2s`; CSE trimmed most), 10 epochs, one seed, no tuning, while
the control came with the authors' recipe. The control's own dev/test spread is
80.0% vs 71.4% — n=70 noise again.

**Stage 3, the multi-axis re-simulation.** All three axes at once — per-class
positives floored at twice the measured null, `nn_hard` nulls at the measured
replicate-noise scale, and 6.09 co-occurring events perturbed per window —
against a volume-matched 960-event control under the same protocol:

    sim-CV 0.756 -> 0.642     benchmark dev 48/70 -> 42/70 (68.6% -> 60.0%),
                              15 lost / 9 gained, McNemar p = 0.308

It made things worse. Three axes moved together, so this cannot apportion blame,
but that was the design: the multi-axis batch was meant to be the honest test of
"realism only pays when several axes move together". It does not. **The realism
premise is now 0 for 3** — depth invariance, effect-size realism, and this.

Read with the crossover, the two results agree: the simulation is not the
bottleneck, and making it harder without making it more informative costs sim-CV
and transfer together. Reading 1 from the section below now has the evidence, and
reading 2 has been tested and failed.

## The live question

The audit's premise was that transfer is limited by simulation realism along
the axes it can measure. On its two largest measured discrepancies that premise
is **0 for 2**, while the one change that worked was about how the window is
*drawn* — framing, not noise. Two readings; this evidence does not separate
them:

1. the remaining 8.6 points are not a simulation-realism problem, and the
   representation deserves the attention it has been denied; or
2. realism only pays when several axes move together, and single-axis fixes are
   individually invisible.

Reading 2 is what the plan assumed. Reading 1 now has the supporting evidence.

## Next steps, in order

1. **The representation crossover** — v1's images + ResNet50 trained on v2's
   simulated data, scored on the same benchmark. This is the experiment that
   separates the two readings. It was deferred twice on the grounds that the
   audit had not finished asking; it has asked twice and got no. Cost: render
   the 4,800 v3 windows with `plot2s.py` on pasteur, train ResNet50 locally in
   the `DSv1` env.
2. **NN false positives** — 41% of remaining errors (13 of 32), the largest
   single bucket. Decide the deployment prior *on principle* and fix it before
   scoring. See the caveat below.
3. **A multi-axis calibration batch**, the honest test of reading 2: the rest of
   the audit shortlist at once (depth down ~5× and made a distribution, layer 4
   up ~2×, layer 2 down except RI rows, layers 3 and 5 down ~2.5×) plus the beta
   effect-size draw. Single-axis nulls cannot rule this out.
4. **Per-class effect-size distributions** if (3) runs.
5. Re-run `dsignal.audit_noise` after any re-simulation — it is the acceptance
   test now, cheap and annotation-free.

Not recommended: more scale-up; more noise layers; Phase-4 fine-tuning.

## The benchmark is now split (Stage 0, 2026-08-17)

`benchmark/split.tsv` — 70 dev / 70 test, committed so it cannot drift.
Stratified by truth class (dev A3=14 A5=15 CSE=33 RI=8, test A3=14 A5=15
CSE=32 RI=9) and cluster-disjoint at 50 kb single linkage, which is a
same-locus proxy for gene identity since the CSVs carry regions only and no
usable annotation is in the repo. The sorted inter-event gaps jump 5.6 kb →
67 kb, so the threshold is insensitive: 132 singletons + 4 pairs at anything in
that decade. `python -m dsignal.benchmark_split make` regenerates it (seed 0).

Both caveats below become dev-set choices under it. **Everything tuned is read
off dev; test is scored once per milestone.** Cost, stated rather than hidden:
McNemar at n=70 needs an 8–10 event swing where n=140 needed 5–6.

Usage:

```bash
python -m dsignal.score_benchmark --net runs/<name>/fold*.pt --half dev \
  --out runs/benchmark_events_<name>.tsv
python -m dsignal.benchmark_split compare \
  runs/benchmark_events_geomwin.tsv runs/benchmark_events_<name>.tsv --half dev
```

The current best splits evenly by luck: 55/70 collapsed on each half, both
78.6%, identical per-event predictions to the cached 140-event run.

## Two open honesty caveats — settle before publishing

**(1) is settled as of Stage 1.5.** The deployment argument is written down in
`dsignal/prior_sweep.py`'s docstring *before* any number: deepSpecas classifies
candidate regions from an upstream differential-splicing caller, not the genome,
so the 25% training NN fraction — a design variable, not an estimate — badly
overstates the deploy prior; a consensus-of-two-callers filter leaves order 10%
non-events. The pre-registered operating point is therefore `nn-weight` 0.1.
On dev it gives 51/70 = 72.9% collapsed against 49/70 uncorrected, 7 false NN
calls → 1, gaining 2 and losing 0 (p = 0.500). The oracle `nn_weight=0` reaches
74.3%, so the committed point takes two thirds of what is available without
being the argmax cell, and the curve never costs an event.

1. **The prior strength is being chosen on the test set.** *(settled — see
   above; kept for the record of what the problem was.)* The sweep's 85.7%
   collapsed at `nn_weight=0.05` (and 87.1% at the `nn_weight=0` oracle, vs
   v1's 87.9%) is selected by reading benchmark accuracy. There is a legitimate
   version of the argument — deployment inputs are pre-filtered rMATS/SUPPA
   candidates, so a low NN prior is defensible a priori — but it must be argued
   from the deployment setting and fixed *before* scoring. v1 needs no such
   correction: it emitted 2 false NN calls.
2. **The effect-size marginals are fitted on the benchmark.** `DPSI_BETA` and
   `PSI_LEVEL_BETA` in `sim/select_events.py` come from the benchmark's own 125
   matched events, so any result using `--psi-dist beta` is domain-calibrated
   along that axis, not zero-shot. What leaks is two 2-parameter marginals, not
   labels or structures. The clean fix is a held-out split of the benchmark, or
   a dPSI distribution from an independent dataset. Neither exists yet.

## The audit's table, for reference

1,500 simulated windows vs the 140 curated ones
(`runs/noise_audit_compare.log`). Everything is over-injected except layer 4.
Depth and effect size have now been tested and are null; the rest are untested.

| quantity | sim/real | knob |
|---|---|---|
| `intron_cov_frac` | 7.43× over | layer 2 `--premrna-frac 0.01–0.10` |
| `se_minor_count` | 6.75× over | effect size — **tested, null** |
| `antisense_frac` | 6.36× over † | layer 3 `--neighbor-depth 100–3000` |
| `a3_minor_count` | 5.28× over | effect size — **tested, null** |
| `cov_mean` | 5.20× over | `--gene-mean-depth 2000` — **tested, null** |
| `top_junction_count` | 4.46× over | depth — **tested, null** |
| `dup_rate` | 2.77× over ‡ | layer 5 `PCR_RATE=0.1` |
| `background_frac` | 2.43× over † | layer 3 |
| `novel_junction_rate` | 0.52× **under** | layer 4 `--novel-frac 0.005–0.03` |

† real side is a lower bound — `benchmark/c1.bam` is a `view -L events.bed`
subset, so off-window background is absent; direction only. ‡ rises with depth
on its own; read next to `cov_mean`.

Layer 2 inverts per class: 7.4× over pooled, but **4.2× under for RI** (real IR
intron/exon coverage 0.558, simulated RI 0.133).

`a3_delta`/`se_delta` are **not** in this table any more. They appeared to show
the simulation under-differential at 0.55×; that metric picks the most
*balanced* junction pair in a window, and simulated windows carry ~1.6× more
supported junctions (median 9 vs 5.5) from noise layers 3–4, so it selected
background structure rather than the event. It measured junction multiplicity.
The event-level measurement says the opposite (2.3× more differential).

## Reproducing the headline number

Local, in the `DSnet` env. Note `--feature-subset no-geometry` is the default but is
stated explicitly here because it is load-bearing, and `--augment all` is *not* default:

```bash
python -m dsignal.train \
  --npz-dir sim/experiment/npz_geomwin --labels sim/experiment/labels_geomwin.tsv \
  --augment all --feature-subset no-geometry \
  --out-dir runs/geomwin_allaug --json-out runs/geomwin_allaug.json

python -m dsignal.score_benchmark \
  --net runs/geomwin_allaug/fold*.pt --out runs/benchmark_events_geomwin.tsv
```

→ sim-CV macro-F1 0.867, benchmark 110/140 = 78.6% collapsed.

## The `dsignal/` CLI surface

Eleven runnable entry points, all `python -m dsignal.<name>`:

| module | what it does |
|---|---|
| `extract` | BAMs + region CSV → `.npz` records |
| `train` | 5-fold GroupKFold-by-gene training, one checkpoint per fold |
| `predict` | `region,bam1,bam2` CSV → predictions |
| `score_benchmark` | the 140 curated events; `--target-prior`, `--tau` |
| `prior_sweep` | one extraction pass, N prior settings evaluated analytically |
| `baseline_gbm` | LightGBM baseline **and** the shared eval harness |
| `audit_noise` | `measure` / `compare` / `padding` — the acceptance test |
| `verify` | extractor parity against `plot2s.get_coords_from_bam` |
| `verify_features` | canonicalization, flip, coverage off-by-one |
| `diagnose_cse_a3` | annotation-free structural motif scoring |
| `tabulate_runs` | side-by-side table of `train --json-out` files |

## Which `sim/experiment/` sets are live

Fourteen `npz_*` directories exist; most are archaeology from the layer-by-layer build.

- **Live:** `npz_geomwin` (4,800 events, v3 windows — the current training set),
  `npz_geomwin_b1` (960-event volume-matched control), `npz_beta` (960 events, beta
  effect size).
- **Archaeology, keep for reprovenance, do not train on:** `npz`, `npz_layer{2,3,4,4fix,5}`,
  `npz_nn_novel`, `npz_window`, `npz_windowfix`, `npz_scale1260`, `npz_scaleup`.
  Everything before `npz_geomwin` uses a superseded window sampler, so its numbers are
  not comparable to anything current.

## Environments and paths

Unchanged from the previous resume doc. Briefly:

- **Local** (RTX 4080, all training): `/datasets/deepSpecas-v2/envs/DSnet` —
  **never `pip install` into it**; `/datasets/deepSpecas-v2/envs/DSv1` for
  running v1 (has torchvision + cv2), which is what the crossover needs.
- **pasteur** (simulation/alignment/extraction, no GPU): `ssh pasteur`,
  `/data/conda/ciccolella/deepSpecas-v2`, envs `DSpoly2v2` / `DSsim`.
  `/data` at 83%. Only `.npz` + labels + TSVs come back, never BAMs/FASTA.
  **pasteur's `dsignal/` holds only the modules shipped there** — rsync any new
  one before running it (this cost a failed launch this session).

## Key artifacts added this session

| path | what |
|---|---|
| `dsignal/audit_noise.py` | audit tool: `measure` / `compare` / `padding` |
| `runs/noise_audit_{real,sim_b1..b5}.tsv`, `..._compare.log` | the audit |
| `runs/window_padding.log` | curated window vs SUPPA event span |
| `sim/window.py` | v3 sampler; docstring carries all three iterations |
| `sim/experiment/npz_geomwin/`, `labels_geomwin{,.nn15}.tsv` | 4,800 v3 records |
| `sim/experiment/npz_geomwin_b1/` | 960-event volume-matched control set |
| `sim/experiment/npz_beta/`, `labels_beta.tsv` | 960 events, beta effect size |
| `runs/geomwin_*`, `runs/depth*`, `runs/beta_*`, `runs/subsample015*` | all runs |
| `runs/benchmark_events_geometry_ctl.tsv` | geometry-only control on benchmark |
| pasteur `sim/{regen_designs.sh,reextract_geomwin.sh}` | re-run scripts |
| pasteur `sim/batches/b6beta/` | the beta batch |

Every benchmark number this branch has produced is in
`runs/benchmark_*.log`; grep `aggregate collapsed`.

## Traps hit this session — do not repeat

1. **A `touch DONE` sentinel placed after a loop that can fail** fired on a run
   where every iteration errored, and a wait loop trusted it. Write sentinels
   only on success.
2. **A bash array built and never expanded.** `EXCL+=(--exclude-genes "$p")` was
   assembled and the command ran without `"${EXCL[@]}"`, silently dropping the
   gene-disjointness between batches. The row-identity gate caught it — keep
   gates like that.
3. **`_library_of` split read names on `.`** to scope duplicate detection:
   correct for the benchmark's three merged SRA runs
   (`SRR1513329.6348221`), wrong for simulated `read2270765`, which made every
   read its own library and reported layer 5 as inert. Fixed; the first audit's
   "sim dup_rate ≈ 0" conclusion was void.
4. **`pgrep -f <pattern>` in a wrapper whose own command line contains the
   pattern** matches itself and waits forever. A leftover instance from an
   earlier session was still hung at PID 224814.
5. **Seeds were not recorded anywhere** and had to be recovered by brute force
   (they are 1–5 for b1–b5). `sim/regen_designs.sh` now records them.

## Conventions

- Never `git push`; never create or switch branches unprompted.
- Code changes go through a plan file under `plans/` + a separate
  implementation session, unless the user says to implement in-session.
- Build → verify → commit each step before starting the next.
- Training local, simulation/alignment/extraction on pasteur.
- Every comparison on 140 events needs a significance test — differences of
  4–6 events are noise here, and several of this session's runs differ by
  exactly that much.
