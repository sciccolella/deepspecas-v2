# `plans/` — index

Plan documents and their implementation reports. Per `CLAUDE.md`, a plan is agreed and
saved here first, and a separate session implements it.

**Start here:** [`signal-native-v2-resume-9.md`](signal-native-v2-resume-9.md) — the
current state of play for the active line of work.

## signal-native-v2 (the active line)

| file | what it is | status |
|---|---|---|
| [`signal-native-v2-resume-9.md`](signal-native-v2-resume-9.md) | **START HERE.** State, direction, open caveats, conventions (2026-09-07) | current |
| [`signal-native-v2-stage12-checkpoint.md`](signal-native-v2-stage12-checkpoint.md) | Stage 12 chronological build log (2026-09-02..07): what was built, how it was verified, what broke and why -- the labeller and its gate, the coverage branch, the PTBP1 arena and its three build failures, every measured number, and the open questions | **build log -- read for mechanism** |
| [`signal-native-v2-real-supervision-results.md`](signal-native-v2-real-supervision-results.md) | Stage 12 results (2026-09-03): the annotation-derived-label gate fails (0.607 vs 0.80 required) and Track A stops -- the catalogue holds the right label 97.5% of the time but ~2 other events per window are equally "present"; Stage 11's `intron_cov_frac` question answered (C1); the coverage branch is a reallocation, not a win (RI pays for A3/A5/CSE), and NN does not move; baseline seed spread is 5 events at n=70 | **implemented; Track B's B2/B3 outstanding** |
| [`signal-native-v2-real-supervision.md`](signal-native-v2-real-supervision.md) | The Stage 12 plan (2026-09-02): real supervision without curation (A), a renewable model-blind arena (B), close-outs (C), a representation change gated on A (D) | A/C/D implemented, B through B1 -- see results doc |
| [`signal-native-v2-resume-8.md`](signal-native-v2-resume-8.md) | Previous resume doc (2026-09-02); accurate through Stage 11 | superseded as the entry point |
| [`signal-native-v2-trash-sim-results.md`](signal-native-v2-trash-sim-results.md) | Stage 11 results (2026-09-02): real-supervision ceiling (Stage 1) doesn't clear zero-shot NN recall; TRASH-simulation pilot (Stage 2) fails its noise gate but localizes why (RI already calibrated, CSE/NN don't respond to the premrna knob) -- Stage 3 (full batches) never ran | **implemented through Stage 2; Stage 3+ not started** |
| [`signal-native-v2-trash-sim.md`](signal-native-v2-trash-sim.md) | The Stage 11 plan (2026-09-02): simulate the real TRASH/NN noise regime, retrain, treat borderline-TRASH as NN | implemented through Stage 2 — see results doc |
| [`signal-native-v2-resume-7.md`](signal-native-v2-resume-7.md) | Previous resume doc (2026-09-02); accurate through Stage 10's `benchmark_new` dev scoring and holds detail resume-8 compresses | superseded as the entry point |
| [`signal-native-v2-bnew-dev-results.md`](signal-native-v2-bnew-dev-results.md) | Stage 10 results (2026-09-02): `benchmark_new` dev scored properly for the first time — v2 beats v1 descriptively (58.7% vs 41.3%/48.8%), NN recall is the dominant weakness (9-14%), `nn_weight=0.1` sensitivity-tested (argmax 0.25-0.5, not significant, kept as-is), TRASH-vs-clean gap confirmed with real scores | **implemented** |
| [`signal-native-v2-resume-6.md`](signal-native-v2-resume-6.md) | Previous resume doc (2026-09-02); accurate through the `benchmark_new` freeze/split and TRASH gate | superseded as the entry point |
| [`signal-native-v2-trash-results.md`](signal-native-v2-trash-results.md) | Full stage-by-stage results of the Stage 9 plan (2026-09-02): `benchmark_new` frozen/split with real NN truth for the first time; TRASH-iness fails its ~0.70 recoverability gate (0.612), so Stages 3-4 (spuriousness features, conditioned NN prior) did not run; Stage 2 found sim falls 2-3x short of `real-TRASH` on several noise metrics | **implemented through Stage 2; Stages 3-4 gated off** |
| [`signal-native-v2-trash-signal.md`](signal-native-v2-trash-signal.md) | The Stage 9 plan (2026-09-02): ingest the manual curation of the 573-event consensus pool, freeze/split `benchmark_new`, and measure whether TRASH-iness is signal-recoverable before acting on it | implemented — see results doc |
| [`signal-native-v2-resume-5.md`](signal-native-v2-resume-5.md) | Previous resume doc (2026-08-21); accurate through the union2 scale/ensemble plan and holds detail resume-6 compresses | superseded as the entry point |
| [`signal-native-v2-fresh-benchmark.md`](signal-native-v2-fresh-benchmark.md) | The Stage 8 plan (2026-08-22): a fresh curated 140-event benchmark from a *different* BioProject (Track A) in parallel with untried ensemble-breadth families — architecture, seeds×views, class-weight boosts, attention-bias ablations (Track B) — converging on one pre-registered v1-vs-v2 spend. Also records why relaxing one-gene-per-event was deferred. Track A3/A4 (consensus + curation) done; Track A's freeze/split is superseded by the Stage 9 plan's Stage 0; Track B and Track C's test spend remain **not started** | **partially implemented** |
| [`signal-native-v2-scale-union2-results.md`](signal-native-v2-scale-union2-results.md) | Full stage-by-stage results of the Stage 7 scale plan (2026-08-21): single-model scale on `npz_union2` plateaus at 56–57/70 dev past ~8,640 events, but view diversity re-ensembled on the largest pool reaches 61/70 (87.1%) dev, dev-tuned only, no test exposure | **implemented** |
| [`signal-native-v2-scale-union2.md`](signal-native-v2-scale-union2.md) | The Stage 7 scale plan (2026-08-20): four new gene-disjoint batches, a three-point scale curve, a conditional ensemble, dev-only throughout | implemented — see results doc |
| [`signal-native-v2-resume-4.md`](signal-native-v2-resume-4.md) | Previous resume doc (2026-08-19); accurate through the ensemble plan and holds detail resume-5 compresses | superseded as the entry point |
| [`signal-native-v2-ensemble-results.md`](signal-native-v2-ensemble-results.md) | Full stage-by-stage results of the ensemble plan (2026-08-19): E3 reaches 77.1% dev / 85.7% test, closing part but not all of the gap to v1 (60/70 dev / 63/70 test); the mechanism that actually worked was more data (`npz_union`), not ensembling | **implemented** |
| [`signal-native-v2-ensemble.md`](signal-native-v2-ensemble.md) | The v2 ensemble plan (2026-08-19). Also **corrects the comparison target**: the v1 six-net ensemble is 60/70 dev, 62/69 test, 122/139 — itself superseded by the results doc's cleaner keying (60/70 dev, **63/70 test**, 123/140) | implemented — see results doc |
| [`signal-native-v2-resume-3.md`](signal-native-v2-resume-3.md) | Previous resume doc (2026-08-18); accurate through the pre-ensemble state and holds detail resume-4 compresses | superseded as the entry point |
| [`signal-native-v2-resume-2.md`](signal-native-v2-resume-2.md) | Previous resume doc (2026-08-17); accurate through Stage 1 and holds the detail resume-3 compresses | superseded as the entry point |
| [`signal-native-v2-docs-and-model.md`](signal-native-v2-docs-and-model.md) | Documentation cleanup + the model-improvement plan (Stages 0–3) | **fully implemented** (Parts A and B) |
| [`signal-native-v2.md`](signal-native-v2.md) | The original top-level plan: Phases 1–5. Every "Phase N step M" reference in a docstring points here | reference |
| [`signal-native-v2-checkpoint.md`](signal-native-v2-checkpoint.md) | Chronological build log of Phases 1–3 and every noise layer. Accurate on *how*; its framing assumes a disproved premise | reference for mechanism only |
| [`signal-native-v2-audit-geometry.md`](signal-native-v2-audit-geometry.md) | Plan: noise-calibration audit + window-geometry realism | implemented |
| [`signal-native-v2-audit-geometry-results.md`](signal-native-v2-audit-geometry-results.md) | Its results — the window fix (62.1% → 78.6%) plus the depth and effect-size nulls | current, the detailed record |
| [`signal-native-v2-resume.md`](signal-native-v2-resume.md) | Previous resume doc (2026-08-16) | **superseded** — two headline findings retracted |

### Phase-2 build history (all implemented; kept for provenance)

| file | what it added |
|---|---|
| [`signal-native-v2-noise-layer2.md`](signal-native-v2-noise-layer2.md) | pre-mRNA background |
| [`signal-native-v2-noise-layer3.md`](signal-native-v2-noise-layer3.md) | neighbouring / antisense genes |
| [`signal-native-v2-noise-layer4.md`](signal-native-v2-noise-layer4.md) | novel unlabelled junctions |
| [`signal-native-v2-noise-layer5.md`](signal-native-v2-noise-layer5.md) | PCR duplicates + adapter contamination |
| [`signal-native-v2-window-randomization.md`](signal-native-v2-window-randomization.md) | window sampler **v1** (flank as a multiple of the event span) |
| [`signal-native-v2-window-geometry-fix.md`](signal-native-v2-window-geometry-fix.md) | window sampler **v2** (width drawn from the pooled benchmark) |
| [`signal-native-v2-phase3-model.md`](signal-native-v2-phase3-model.md) | the junction-set Transformer |
| [`signal-native-v2-prior-correction.md`](signal-native-v2-prior-correction.md) | inference-time class-prior correction |
| [`signal-native-v2-scaleup-7500.md`](signal-native-v2-scaleup-7500.md) | scale-up to 4,800 events; class prior as a design variable |

Window sampler **v3** — the one that is live — was specified from measurement rather than
from a plan file; its derivation is in the audit-geometry results doc and its full
three-iteration record is the `sim/window.py` module docstring.

## v1 / `master`

| file | what it is | status |
|---|---|---|
| [`render-throughput.md`](render-throughput.md) | Plan for `--shard` and `--gpu-render` in `dsEnsemble.py` | implemented (`c675641`, `379cbba`); results in `../report.md` |

`../report.md` is the v1 inference-throughput report, including the Round 2 results.
