# signal-native-v2 — Stage 12 results: the A1 gate fails, and why

*Results, 2026-09-02. Plan: `plans/signal-native-v2-real-supervision.md`.
Predecessors: `plans/signal-native-v2-trash-sim-results.md` (Stage 11),
`plans/signal-native-v2-bnew-dev-results.md` (Stage 10).*

## Summary

**Track A's pre-registered gate (A1) fails on both criteria, and Track A
stops there per its own stop rule.** Annotation-derived labels cannot
substitute for curation on real BAM windows — not because the annotation is
missing the events (it contains the correct label for **97.5%** of non-NN dev
windows) but because a window that contains one curated event typically
contains **~2 other annotated events that are equally "present"** by any
BAM-derived support criterion, and no automatic rule tested here can pick out
the one a curator would name.

| criterion | required | measured (dev) | |
|---|---:|---:|---|
| 1. non-NN agreement | ≥ 0.80 | **0.607** | fail |
| 2. NN recall, clean windows | ≥ 0.50 | **0.314** | fail |

The failure is structural, not a tuning miss: **the ceiling on criterion 1 is
0.804** — the fraction of non-NN dev windows where the correct label appears
among the events that pass the presence filter at all. A perfect
disambiguator would land exactly on the criterion with zero margin. Fifteen
threshold/rule combinations were tried (below); the best non-NN agreement
reached was 0.762, and every gain in NN recall cost non-NN agreement roughly
one-for-one.

Stages C0 and C3 are done and committed. **Stage A2 onward did not run**, so
no pasteur time was spent on this track and no real-supervision training pool
exists. Both benchmarks' test halves remain unspent.

---

## What was built (A0)

| file | what |
|---|---|
| `sim/real_label.py` | the labeller: SUPPA event_id → the two competing forms as intron sets; catalogue interval index; BAM-derived presence filter; the A0 assignment rule |
| `sim/real_label_gate.py` | the A1 gate: runs it over curated windows and reports against curated truth, criteria inline |

Design decisions worth keeping (they are what a future attempt would reuse):

* **Presence means *alternatively spliced in this sample*, not merely
  annotated or expressed** — both competing forms of the event must carry
  evidence. Per SUPPA type: SE inclusion `{(e1end,e2start),(e2end,e3start)}`
  vs exclusion `{(e1end,e3start)}`; A3/A5 the two junctions sharing a
  donor/acceptor; RI the spliced junction vs intronic coverage, since the
  retained form has no junction to count.
* **The annotation→observed coordinate offset is calibrated, not trusted.**
  `calibrate_offset` scores candidate offsets by how many catalogued
  junctions they match. Result on both benchmarks: `(+1,-1)` — the 1-based
  inclusive intron convention — by a factor of >100 over every alternative
  (1529 matches vs 7/3/0). A silent off-by-one here would have been
  indistinguishable from "the annotation doesn't fit the data", which is
  exactly the wrong conclusion to reach by accident.
* **A window with no reads abstains rather than becoming NN.** "No reads" is
  not evidence of "no event", and free NN training data acquired that way
  would be a coverage detector, not a classifier.

## The gate (A1)

Run over `benchmark_new`'s 485 unambiguous curated rows and `benchmark/`'s
140. Defaults declared in the plan before running: `min_reads=5`,
`ri_min_frac=0.10`, `min_containment=0.5`, `min_window_cov=1.0`.

### `benchmark_new`, dev half (n=242, the half criteria are read off)

```
  truth\pred       A3     A5  AMBIG    CSE     NN     RI      n     acc
  A3              11      0      2      0      3      0     16   0.688
  A5               0      9      3      0      0      0     12   0.750
  CSE              2      1     25     69      9      1    107   0.645
  NN               4      4     23     28     15      5     79   0.190
  RI               1      1      8      2      6     10     28   0.357
  non-NN agreement : 99/163 = 0.607     [criterion 1: >= 0.80]  FAIL
  NN recall (clean): 0.314  (n=35)      [criterion 2: >= 0.50]  FAIL
  abstained        : 61/242
```

Test half, reported but not used for the decision: non-NN agreement 0.632,
NN recall clean 0.143. All-485: 0.620 / 0.229. **`benchmark/` (140 events,
the cleaner set, curated before either model existed): 0.614** — so this is
not an artifact of `benchmark_new`'s dirtier candidate pool.

### Why it fails — the diagnostic that matters

Decomposing the dev non-NN rows:

| stage | fraction of non-NN dev windows |
|---|---:|
| correct label present in the catalogue at all (containment 0.0 or 0.5) | **0.975** |
| …and among the events passing the presence filter | **0.804** |
| …and actually chosen by the assignment rule (`label`) | 0.607 |
| …chosen when abstentions are broken by support instead (`resolved`) | 0.712 |

Per class (dev): A3 1.00 → 0.81 present, A5 1.00 → 1.00, CSE 0.98 → 0.82,
RI 0.93 → 0.64. Mean candidate events per window 2.9 (CSE) to 5.5 (A3).

**And the same is true of NN windows**: mean 3.9 candidates, **mean 1.97
passing the presence filter**, with only 19% having zero present events. A
curated-NN window is not a window without annotated alternative splicing —
it is a window where a human judged that nothing convincing was happening.
That distinction is invisible to any support-threshold rule, which is
precisely why criterion 2 fails so badly.

### Everything tried, and the trade-off

Threshold sweep (dev), and two extra rules — uniqueness (accept only windows
where all present events agree on a class) and dominance (require the winner
to beat the best competing-class event by a ratio):

| rule | non-NN agreement | NN recall (clean) |
|---|---:|---:|
| defaults (`min_reads=5`) | 0.607 | 0.314 |
| `min_reads=2` | 0.632 | 0.229 |
| `min_reads=10` | 0.583 | 0.371 |
| `min_reads=20` | 0.497 | 0.457 |
| `resolved` (tie-break instead of abstain) | 0.712 | — |
| + uniqueness, `min_reads=5` | **0.762** | 0.268 |
| + uniqueness, `min_reads=20` | 0.566 | 0.500 |
| + dominance ×2, `min_reads=5` | 0.743 | 0.224 |
| + dominance ×5, `min_reads=5` | 0.737 | 0.242 |

Nothing reaches 0.80, the two criteria trade against each other almost
one-for-one, and the best achievable is bounded above by the 0.804 ceiling.

## Verdict

**Track A stops at A1**, as the plan pre-registered. The specific claim that
died is "annotation + a BAM-derived presence filter recovers what curation
decided". The claim that survives, and is worth keeping, is that **the
annotation is not the bottleneck** — it holds the right label 97.5% of the
time. What curation supplies is *selection among co-present annotated
events*, and that is the part this stage found no automatic substitute for.

What this does **not** establish: that real supervision is worthless. It
establishes that real supervision at ~76% label purity is the best this
particular automatic route offers, against simulation's ~100%. Whether
10^4 events at 0.76 purity beats 1.2×10^4 at ~1.0 is an open empirical
question this stage did not answer, and one that could be answered later
without curation if a reason arises to revisit it.

Note also that this gate is, if anything, *easier* than the production task
in one respect and harder in another: harder because the benchmark windows
are built around a caller's locus rather than around a chosen event; easier
because a production selector could choose its own windows. The measured
~2 co-present events per window is a property of the genome and the
annotation, though, not of the window convention — it does not go away by
choosing windows differently.

## Consequences for the rest of Stage 12

* **A2–A5 do not run.** No real-supervision pool, no third arm, and
  `benchmark_new`'s test half stays unspent — A5's spend was conditional on a
  Track A winner existing.
* **Track D loses its gate.** D's branch (pretraining-and-more-data vs an
  encoder change) was to be selected by A3's three-arm comparison. With A3
  gone, that fork has to be decided some other way — a decision for the user,
  not one to make silently.
* **Track B is unaffected** and is now the main remaining line.
* **Track C1** (Stage 11's `intron_cov_frac` mechanism question) is
  unaffected and still worth doing locally.

## C1 — Stage 11's mechanistic question, answered locally

The question carried over from `plans/signal-native-v2-trash-sim-results.md`:
*why does elevating `premrna_frac` not elevate `intron_cov_frac`
proportionally for CSE/NN the way it does for RI?* Stage 11's two candidate
causes were `MIN_INTRON_FOR_COV`/flank filtering and CSE/NN's intron-length
structure. Answered here from data already in hand (`npz_union2`, n≈2000
sampled, vs `benchmark_new`'s 485 real windows), no pasteur time.

**1. `MIN_INTRON_FOR_COV` is not the cause.** Short introns account for
0–4% of dropped junctions in every class, sim and real alike. That candidate
is eliminated.

**2. The window-edge rule is the dominant filter, and it explains A3/A5 —
not CSE/NN.** `_intron_cov_frac` skips any junction whose intron comes within
`flank`=50bp of a window edge, which drops 42–92% of supported junctions:

| class | sim kept% | real kept% | sim events with **no** measurable junction | real |
|---|---:|---:|---:|---:|
| A3 | 0.17 | 0.08 | 0.48 | **0.79** |
| A5 | 0.21 | 0.06 | 0.46 | **0.91** |
| CSE | 0.35 | 0.57 | 0.13 | 0.03 |
| NN | 0.33 | 0.42 | 0.52 | 0.25 |
| RI | 0.20 | 0.32 | 0.21 | 0.26 |

Only 9–24% of real A3/A5 windows yield a measurable value at all. **Stage
11's "A3 10x short (real n=16, likely noisy)" is now explained rather than
guessed**: the metric is close to undefined for A3/A5 on real data, and its
A3/A5 rows should not be read as calibration evidence in either direction.
Relaxing the rule to allow truncated flanks moves A3 21.7x→9.6x and A5
7.8x→3.3x, but leaves CSE (4.05→3.90) and NN (4.68→4.60) essentially
unchanged — so the edge rule is *not* the CSE/NN story.

**3. Within a domain, the metric is largely set by window geometry.** Across
all five classes, `intron_cov_frac × (intronic/exonic territory of the
window)` ≈ 1 (RI 1.12, NN 0.64, CSE 1.06, A3 0.93, A5 1.04), with median
intronic fraction ranging 0.75 (RI) to 0.99 (CSE). This is the mechanism
behind Stage 11's stratification fix: the *pooled* gate compares whichever
group happens to carry more RI-like geometry, which is why the pooled read
was backwards. Use `--by-label` always, as `resume-8` already instructs.

**4. But the CSE/NN sim-vs-real gap is real, not a geometry confound.**
Matching sim and real within bins of intronic fraction does **not** close it:

| class | unmatched ratio | within-bin ratios (real/sim) |
|---|---:|---|
| CSE | 3.83 | 9.42 / 3.82 / 3.16 / 2.85 |
| NN | 4.73 | 7.24 / 7.95 / 4.10 / 6.05 |
| RI | 1.38 | 1.18 / 1.78 |

So Stage 11's central finding stands: RI is well-calibrated, CSE and NN are
genuinely 3–6x short of real intronic coverage at matched geometry.

**5. Why the knob is sublinear — a hypothesis, not a measured fact.** The
pre-mRNA row is added as a fraction of the gene's total *reads*
(`sim/abundance.py:_add_low_abundance_row`), and those reads spread over the
pre-mRNA's length while spliced reads spread over the mRNA's. Per-base
intronic coverage is therefore ≈ `premrna_frac × L_mRNA / L_preMRNA`: the
knob's effect on the metric is scaled by a **per-gene length ratio**. RI's
retained intron is a genuine annotated isoform whose coverage is set by PSI
rather than by this background, which is why it responds and CSE/NN do not.
Consistent with Stage 11's own numbers — a 3.5x increase in median
`premrna_frac` (U[0.01,0.10] → U[0.08,0.16]) produced only 1.29x in CSE's
`intron_cov_frac` (0.0077 → 0.0099). Within-class rank correlation between
`log icf` and `log` intronic fraction is negative in all five classes but
weak for CSE (rho −0.10, p=0.14), so this is a plausible model, not an
established one; confirming it needs the per-gene length ratio, which is not
in the locally committed tables.

### Verdict on the TRASH layer

**Revivable, with a specific change — not by re-tuning the knob.** A uniform
elevation of `--trash-premrna-frac-*` cannot fix a shortfall whose size
varies per gene by the length ratio above. If the layer is picked up again,
the per-row pre-mRNA fraction should be **solved for the target
`intron_cov_frac`** using the gene's own exonic/intronic length ratio, rather
than drawn from a global range and hoped to land. Until then the layer stays
shelved, and the gate it is measured against must be read `--by-label`, must
exclude A3/A5 (undefined on real data), and should report within-geometry
matched ratios rather than pooled ones.

## Track D — the coverage branch: a reallocation, not an improvement

D's fork was chosen by the user (2026-09-03) after A3 was lost with Track A:
add a binned log-coverage branch to the junction-set transformer, on the
hypothesis that the hand-derived `[J,48]`/`[32]` features do not carry the
intronic-continuity evidence **NN** needs.

### What was built

`features.build_coverage_profile` → `[2, bins]` depth-normalized log coverage,
one row per condition. Bin edges are *fractional* and each bin is the exact
integral of the piecewise-constant coverage over it, so mirroring a record
reverses the profile bit-for-bit; integer edges break that at odd window
lengths and the flip augmentation would quietly stop being an involution.
`model.CoverageEncoder` is a dilated 1D conv stack (1/2/4/8) with mean+max
pooling, added **into the CLS embedding** rather than appended as a token —
CLS is never masked and is what the heads read. `ModelConfig.coverage_bins`
defaults to 0, so every pre-Stage-12 checkpoint loads and behaves identically
(verified: `runs/geomwin_iv30_refit` still scores 53/70 on `benchmark/` dev).
The profile is built from the *augmented* `NormalizedRecord`, so flip,
condition_swap, jitter, subsample and condition-swap TTA all apply to it for
free. `verify_features` gained **check 6** (mirror and condition-swap
symmetries, exact equality, not a tolerance).

### The result: it depends entirely on which benchmark you score

Three paired seeds (0/1/2), base recipe on `npz_geomwin`, `--coverage-bins 128`
against the identical recipe without it:

| | `benchmark/` dev (n=70) | `benchmark_new` dev (n=242) | in-sim macro-F1 |
|---|---|---|---|
| base, per seed | 53 / 56 / 58 → **mean 55.7 (79.5%)** | 117 / 127 / 122 → **mean 121.7** | 0.8604 |
| +coverage, per seed | 62 / 63 / 60 → **mean 61.7 (88.1%)** | 126 / 121 / 121 → **mean 122.7** | 0.8671 |
| paired McNemar | p = 0.023 / 0.119 / 0.754 | p = 0.12 / 0.41 / 1.00 | — |

**On `benchmark/` dev it looks like a large win (+6 events, and it wins all
three pairings). On `benchmark_new` dev it is a null (+1 event out of 242).**
Only one of six paired comparisons clears significance.

### Why, and it is not ambiguous

The branch makes the *same* trade on both benchmarks — mean over three seeds:

| class | `benchmark/` dev Δ | `benchmark_new` dev Δ |
|---|---:|---:|
| A3 | +0.167 | +0.000 |
| A5 | +0.111 | +0.167 |
| CSE | +0.101 | +0.031 |
| NN | — (set has no NN) | **−0.004** |
| RI | **−0.167** | **−0.143** |

It buys A3/A5/CSE accuracy and pays for it in **RI**, consistently, on both
sets. What differs is the class mix: `benchmark/` dev is 89% A3/A5/CSE and
only 8/70 = 11% RI, so the trade nets +6; `benchmark_new` dev is 33% NN
(untouched by the branch) and RI-light-but-CSE-heavy, so the same trade nets
+1. **The branch is a reallocation across classes, and how good it looks is a
property of the evaluation set, not of the model.**

**Track D's hypothesis is not supported.** NN recall — the reason the branch
was built — moves 0.114 → 0.110 on `benchmark_new` dev (three-seed mean),
i.e. not at all. Whatever the coverage profile adds, it is not the evidence
NN needs.

A plausible mechanism for the RI loss, consistent with C1 and worth testing
before anyone revives this: a *simulated* retained intron is a separate
annotated isoform and therefore has near-uniform intronic coverage by
construction, while a real one is ragged. A branch that reads coverage shape
directly can learn sim-RI's signature and mis-transfer, where the
junction-anchored summaries it replaces were blind to that difference. C1
found RI to be the one class whose `intron_cov_frac` *level* already matches
real data (ratio 1.05-1.44) — level, not shape.

### The methodological finding, which outlives the branch

**Baseline seed spread alone is 5 events at n=70 (53/56/58) and 10 at n=242
(117/127/122).** That is the same magnitude as most of the "improvements"
this branch has compared across its whole history. A single-seed A/B on
either dev half cannot distinguish a real effect from seed noise, and the
first seed here would have supported a +9-event claim (p=0.023) that three
seeds cut to +6 on one benchmark and +1 on the other. Any future comparison
on this branch should be seed-replicated before it is believed.

**Verdict: not adopted.** The branch stays in the code, defaulted off
(`coverage_bins=0`), verified, and free — but it is not an improvement to
deploy, and the 88.1% dev figure must not be quoted as a headline. It is what
a class-mix-favourable benchmark says about a model that got worse at RI.

## C2 — measurements recorded so they are not re-derived

From `benchmark_new/curation.tsv`, 485 unambiguous rows, candidate class =
`csv_source` under `{A3:A3, A5:A5, ES:CSE, IR:RI}`:

| quantity | value |
|---|---:|
| rMATS × SUPPA2 consensus class vs curated truth, conditional on the event being real | 304/326 = **93.3%** |
| fraction of consensus candidates that are real events | 326/485 = **67.2%** |
| candidate == truth, all rows | 304/485 = 62.7% |
| … clean windows (`trash == 0`, n=345) | 76.2% |
| … TRASH windows (`trash == 1`, n=126) | **28.6%** |

Caller statistics do **not** separate real from NN: purity is flat at
0.66–0.73 across every threshold on `rmats_pvalue`, `|suppa_dpsi|`,
`|rmats_incleveldiff|` and their conjunction (`|dpsi|≥0.3 & |dinc|≥0.2 &
p≤0.01` → n=96, purity 0.708). Median `rmats_pvalue` is *lower* for the NN
group. Consequence: an uncurated consensus benchmark carries ~33%
positive-label noise no stringency filter removes — usable as a model-blind
**paired** arena (symmetric noise attenuates but does not bias a difference),
not for absolute accuracy.

Crosstab (candidate class × curated truth, n=485):

```
truth  A3  A5  CSE   NN  RI
cand
A3     30   1    1   26   6
A5      0  22    1   11   3
CSE     0   0  209  102   5
RI      3   0    2   20  43
```

## Artifacts

| path | what |
|---|---|
| `sim/real_label.py`, `sim/real_label_gate.py` | the labeller and its gate |
| `runs/a1_gate_bnew.tsv` | per-window gate output, `benchmark_new` (485) |
| `runs/a1_gate_old.tsv` | per-window gate output, `benchmark/` (140) |

## Open honesty caveats

Unchanged from Stage 11 (`nn_weight=0.1`; `DPSI_BETA`/`PSI_LEVEL_BETA` fitted
on `benchmark/`'s 140; `benchmark_new` test unspent, `benchmark/` test spent
at Stage 6), plus:

* The A1 thresholds were declared before the run, but the *sweep* around them
  was inspected afterwards on dev. Nothing was carried forward from it — the
  track stopped — so nothing is fitted to dev here. If this route is ever
  revisited, the sweep table above is dev-fitted and a fresh criterion should
  be pre-registered rather than read off it.
* Stage 10's "descriptive, not pre-registered" caveat on `benchmark_new` was
  **retracted this stage (C0)** on the curator's attestation that the
  `v1_pred`/`v2_pred` columns were visible but not consulted. The caveats
  remain in place in the earlier docs with the retraction beside them.
