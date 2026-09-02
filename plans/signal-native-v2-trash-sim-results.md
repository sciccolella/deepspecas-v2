# signal-native-v2 — Stage 11 results: real-supervision ceiling + TRASH-sim pilot

*Results, 2026-09-02. Plan: `plans/signal-native-v2-trash-sim.md`. Predecessors:
`plans/signal-native-v2-bnew-dev-results.md` (Stage 10),
`plans/signal-native-v2-trash-results.md` (Stage 9).*

## Summary

Stage 1's diagnostic ceiling does not clear zero-shot NN recall, and Stage 2's
TRASH-simulation pilot does not clear its pre-registered noise-calibration
gate even after two rounds of methodology fixes — the pilot's own real
finding (RI already well-calibrated, CSE/NN genuinely short) is more useful
than the number that would have come from blindly re-tuning knobs. **Stage 3
(full re-simulation batches, ~8-10h pasteur) did not run.** Stages 0-2 are
implemented, verified, and committed; the TRASH layer itself
(`build_experiment.py`, `--trash-rate`/`--trash-label-policy`) is real,
tested infrastructure that a future session can pick up with better-targeted
knobs, once Stage 2's finding is understood well enough to act on it.

Everything below is dev-only; both benchmarks' test halves remain unspent.

---

## Stage 0 — dual scoring conventions

`dsignal/score_benchmark.py` gained `apply_trash_as_nn(rows, curation_path)`
and `--trash-as-nn <curation.tsv>`; `dsignal/benchmark_split.py compare`
gained the same via `--trash-as-nn`. Convention B: any `curation.tsv` event
with `trash == "1"` has truth forced to NN regardless of its curated label
(a TRASH event carrying a positive label is borderline enough that NN is the
defensible call — user, 2026-09-02); `trash == "NA"` rows are dropped
(n 242 → 235 on dev).

Verified against the plan's pre-registered baseline table — reproduced
exactly:

| model | A: 5-class | B: TRASH-as-NN | NN recall (B) |
|---|---:|---:|---:|
| `e4_union2` | 58.7% (57.9% n=235) | 51.5% | 9.2% |
| `e3_bagged` | 49.6% | 43.0% | 6.1% |
| `geomwin` single | 48.3% | 41.3% | 5.1% |
| v1 ensemble | 41.3% | 38.3% | 5.1% |

## Stage 1 — real-supervision ceiling (diagnostic only, never a deliverable)

`dsignal/train.py` gained `--dump-oof` (per-event out-of-fold predictions,
opt-in). New `dsignal/real_supervision_ceiling.py` builds a 5-fold
grouped-by-cluster training set straight from `benchmark_new` dev (242
events, extracted locally from the benchmark's own BAMs, no pasteur needed)
and trains/evaluates entirely within it — same base recipe
(`--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30`).

**The ceiling does not clear zero-shot:**

| | NN recall (A) | NN recall (B, TRASH-as-NN) |
|---|---:|---:|
| real-supervision ceiling | 9/79 = **11.4%** | 14/98 = **14.3%** |
| zero-shot `e4_union2` (Stage 10) | 13.9% | 9.2% |

Under convention A the ceiling is actually *below* the best zero-shot model.
Caveat: n=242 (~194/fold train) is thin for a junction-set transformer, so
part of this could be small-sample instability rather than a hard
information-theoretic ceiling — the two readings (intrinsically hard vs. not
enough real data to know) cannot be fully separated at this n. The user
chose to proceed to Stage 2 despite this discouraging signal.

## Stage 2 — TRASH noise layer + one pilot batch

### 2a/2b/2c — implementation

`sim/build_experiment.py`: `SeedSequence(seed).spawn(4)` → `spawn(5)`, the
5th stream draws one Bernoulli "trashy" flag per design row (class-conditional
rate, or `--trash-rate-mode uniform` for the label-independent control arm —
not yet exercised, since Stage 3 never ran). Trashy rows draw
`premrna_frac`/`novel_frac` from an elevated range
(`--trash-premrna-frac-*`, `--trash-novel-frac-*`) instead of the background
one. `--trash-label-policy nn`: a trashy row whose planted label isn't
already NN is emitted with label NN, kind `nn_trash` — `kind` stays outside
`("nn_plain", "nn_novel")` so the underlying biological event is still
simulated with its real psi1/psi2; only the training label changes.
`--out-design` writes the rewritten table (`design_final.tsv`), including
`orig_label`/`orig_kind` for provenance (added in the second gate-fix round,
see below). All new flags default to off/unchanged.

**Deviation from the plan's file assignment**: the trash draw and label
rewrite live in `build_experiment.py`, not `select_events.py` — because
`build_experiment.py` is what already draws `premrna_frac`/`novel_frac` per
row, and is the only place that can gate the elevated ranges on the same
flag that decides the label rewrite, keeping both in one RNG-isolated place.

`sim/run_batch.sh` gained env-var pass-through for every knob mentioned in
the plan (premrna/novel/neighbor-depth/gene-mean-depth fractions, `N_NN`
split into `N_NN_PLAIN`/`HARD`/`NOVEL`, the `TRASH_*` knobs), and now points
`build_extract_csv` at `design_final.tsv` instead of `design.tsv`.

Verified locally (synthetic transcript table over real catalogued events,
no genome needed): count_matrix.tsv byte-for-byte identical with the layer
off; with it on, measured trash rates land close to the requested
class-conditional rates (n=60/class noise), premrna_frac visibly elevated
for flagged rows, `orig_label`/`design_final.tsv` correctly preserve/rewrite
per the policy.

### Incidental fix

`sim/align_reads.sh` was tracked in git as `100644` (non-executable) despite
being invoked directly by `run_batch.sh`; pasteur's copy had been `chmod +x`
out of band at some point, so this only surfaced when this stage's rsync
overwrote it and STAR's step died with "Permission denied". Fixed
(`100644` → `100755`) and committed; unrelated to the TRASH layer itself.

### 2d — pilot batch and the gate

One pilot (`trash_pilot`, seed 101, gene-disjoint from all 15 prior batches
via `--exclude-genes`, 960 events: 720 positive + 240 NN split
plain/hard/novel) ran on pasteur with `TRASH_LABEL_POLICY=nn` and the
measured real class-conditional trash rates (A3 0.03, A5 0, CSE 0.151, RI
0.167, NN 0.545). Standing gates: `psi_recovery_check` mean abs error per
label 0.135-0.192 (within the 0.13-0.20 band, **pass**); 190/960 rows drawn
trashy, close to the target rates given n=180/class; 52 positives relabelled
NN.

**The pre-registered pooled gate fails**, but not for the reason it first
appeared to:

| | sim-clean | real-clean | sim-TRASH | real-TRASH |
|---|---:|---:|---:|---:|
| `intron_cov_frac` (median) | 0.0675 | 0.0356 | 0.0508 | 0.1127 |
| `novel_junction_rate` (mean) | 0.0600 | 0.0973 | 0.1401 | 0.1673 |

Pooled: `novel_junction_rate` passes on the TRASH side (1.19x short, within
1.3x) but not clean (1.62x short); `intron_cov_frac` looks badly
miscalibrated on *both* sides — clean overshoots real by 1.9x, which is
backwards (more sim noise than real "clean" noise) and doesn't square with
Stage 9's own measurement of clean-vs-real agreeing almost exactly
(0.0343 vs 0.0356).

**Root cause, found by stratifying `--by-label` (new `audit_noise.py` flag)
rather than re-tuning blindly:** `intron_cov_frac` is not purely a noise
metric for the RI class — retained-intron *is* high intronic coverage, so
RI's legitimate positive signal dominates whichever pooled group has more of
it. The pilot's design is class-balanced (180/class); `benchmark_new` dev is
naturally CSE-heavy (44%) and RI-light (12%). Per class, the picture flips:

| class | sim-clean | real-clean | ratio | sim-TRASH | real-TRASH | ratio |
|---|---:|---:|---:|---:|---:|---:|
| RI/IR | 0.368 | 0.388 | 1.05x **pass** | 0.244 | 0.279 | 1.14x **pass** |
| CSE/ES | 0.0077 | 0.0267 | 3.5x short | 0.0099 | 0.0658 | 6.6x short |
| NN | 0.0192 | 0.0294 | 1.5x short | 0.0368 | 0.1221 | 3.3x short |
| A3 | 0.0493 | 0.5027 | 10x short (real n=16, likely noisy) | n=1, unusable | - | - |

RI needs no change at all. CSE and NN are genuinely short on both sides, and
critically, **the TRASH-side elevation barely moved CSE's number** (0.0077 →
0.0099 clean → TRASH, despite the verified elevated `premrna_frac` actually
being drawn) — the same knob does not translate proportionally into
`intron_cov_frac` across event structures, plausibly because CSE/NN windows
have shorter/fewer introns that survive the metric's `MIN_INTRON_FOR_COV`
filter. This means the fix is not "raise `--trash-premrna-frac-*` uniformly"
— that was tried once (the pooled read looked like a 1.67-2.2x pooled
shortfall) and the stratified read shows it wouldn't touch the actual
problem (CSE/NN's response to the knob, not the knob's magnitude).

**Getting the stratified read required two follow-up fixes to the gate
machinery itself**, both committed:
1. `sim/build_extract_csv.py` + `dsignal/audit_noise.py`: pass a `trash`
   column through `labels.tsv` into `audit_noise measure`'s per-row
   metadata, so a pilot batch's own trashy/non-trashy subgroups are
   identifiable at all (the plan's worked example only grouped the *real*
   side).
2. `sim/build_experiment.py` `--out-design` + the same pass-through: expose
   `orig_label` alongside the (possibly NN-rewritten) `label` — under
   `--trash-label-policy nn`, every trashy row's label becomes NN by
   construction, so a TRASH-side breakdown by `label` was degenerate (100%
   NN) until `orig_label` (the pre-rewrite planted class) was threaded
   through the same path.

Per the plan's own instruction ("if the gate fails, iterate on the pilot
only — do not proceed to full batches"), and given the fix now needed is
class-specific investigation rather than a knob value, the user chose to
stop here rather than spend another pasteur cycle guessing.

## What did not happen

* **Stage 3 (full batches, 4-5 gene-disjoint runs, ~8-10h pasteur) did not
  run.** The TRASH layer's plumbing is real and reusable, but no
  `npz_trash*` pool exists and no model has been trained on it.
* **Stage 4/5 (evaluation, ensembling)** — moot without Stage 3.
* The label-independent control arm (`--trash-rate-mode uniform`) exists in
  code but has never been exercised.

## Critical files touched

| file | change |
|---|---|
| `dsignal/score_benchmark.py` | `apply_trash_as_nn`, `--trash-as-nn` |
| `dsignal/benchmark_split.py` | `compare --trash-as-nn` |
| `dsignal/train.py` | `--dump-oof` |
| `dsignal/real_supervision_ceiling.py` | new: Stage 1 adapter (build/report) |
| `sim/build_experiment.py` | TRASH layer (`spawn(5)`, `--trash-*`, `--out-design`) |
| `sim/run_batch.sh` | knob pass-through, `design_final.tsv` wiring |
| `sim/build_extract_csv.py` | `trash`/`orig_label` pass-through to `labels.tsv` |
| `dsignal/audit_noise.py` | `trash`/`orig_label` pass-through in `measure`; `--by-label` stratified gate in `compare` |
| `sim/align_reads.sh` | executable-bit fix (incidental) |

## Next session, if this is picked back up

Read this file and `plans/signal-native-v2-trash-sim.md` first. The open
question is mechanistic, not a parameter search: **why does elevating
`premrna_frac` for a trashy row not elevate `intron_cov_frac` proportionally
for CSE/NN the way it does for RI?** Candidates worth checking before
touching any knob again: whether `MIN_INTRON_FOR_COV`/the flank window is
filtering out most of CSE/NN's junctions before the ratio is even computed,
and whether CSE/NN's typically-shorter introns (vs RI's, which by definition
span the whole retained intron) structurally cap how much of an injected
pre-mRNA background row's coverage lands inside the measured interior versus
its flanks.
