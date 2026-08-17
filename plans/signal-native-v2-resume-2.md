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

## The one thing to know

**Zero-shot transfer went 62.1% → 78.6% collapsed, and the cause was the
extraction window, not the noise model.** Same simulated reads, same
alignments, same 4,800 events and genes, same model and hyperparameters — only
the window definition changed. No re-simulation.

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
| `--depth-norm scale` | 4,800 | 0.868 | 77.9% | **p = 1.000** |
| `--subsample-low 0.15` | 4,800 | 0.862 | 73.6% | — |
| `--depth-norm drop` | 4,800 | 0.834 | 72.9% | — |
| *(control)* uniform \|dPSI\| | 960 | 0.784 | 72.9% | — |
| `--psi-dist beta` | 960 | 0.693 | 70.0% | p = 0.572 |
| beta + depth scale | 960 | 0.689 | 72.1% | p = 1.000 |

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

## Two open honesty caveats — settle before publishing

1. **The prior strength is being chosen on the test set.** The sweep's 85.7%
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
