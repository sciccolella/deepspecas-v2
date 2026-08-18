# `plans/` — index

Plan documents and their implementation reports. Per `CLAUDE.md`, a plan is agreed and
saved here first, and a separate session implements it.

**Start here:** [`signal-native-v2-resume-3.md`](signal-native-v2-resume-3.md) — the
current state of play for the active line of work.

## signal-native-v2 (the active line)

| file | what it is | status |
|---|---|---|
| [`signal-native-v2-resume-3.md`](signal-native-v2-resume-3.md) | **START HERE.** State, direction, open caveats, conventions (2026-08-18) | current |
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
