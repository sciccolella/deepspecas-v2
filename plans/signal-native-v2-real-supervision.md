# signal-native-v2 — Stage 12 plan: real supervision, a renewable arena, and the representation question

*Plan, 2026-09-02. Approved before implementation; **not yet implemented**.
Predecessors: `plans/signal-native-v2-resume-8.md` (entry point),
`plans/signal-native-v2-trash-sim-results.md` (Stage 11),
`plans/signal-native-v2-bnew-dev-results.md` (Stage 10).*

Constraint set by the user at the outset: **no new manual curation.** It is
extremely labour-intensive and not without failure modes, since parts of the
assessment are interpretive and debatable. Every track below produces its
labels automatically or reuses labels that already exist.

---

## Correction to the committed record, before anything else

Several committed documents state that `benchmark_new`'s curation used
`v1_pred`/`v2_pred` as review aids, and therefore that every v1-vs-v2 number
on that set is "descriptive, not a pre-registered arena":

* `plans/signal-native-v2-bnew-dev-results.md` — Summary point 1, Stage 2's
  "Contamination caveat, repeated", open caveat #5
* `plans/signal-native-v2-resume-7.md` — "The one thing to know" point 1,
  "What is now settled" third bullet, open caveat #5
* `plans/signal-native-v2-resume-8.md` — carried forward by reference

**The curator (the user, 2026-09-02) states that the prediction columns were
visible in the review sheets but did not inform the manual assessment.** The
columns were present; they were not an input. Per the branch's
never-delete-a-finding convention, the caveats stay where they are, with this
attestation recorded next to each — but the conclusion flips: the
`benchmark_new` dev v1-vs-v2 comparison (`e4_union2` 58.7% vs v1 41.3%
5-class, p<0.0001; 58.7% vs 48.8% collapsed, p=0.0032) is a real result, not
merely a descriptive one, and **`benchmark_new`'s test half is a legitimate
pre-registered arena.**

Recording this correction is step **C0** below and should be done first, so
that no later session re-derives the retracted caveat.

### The three benchmarks, and what each one can still be asked

| set | provenance | remaining capacity | can measure NN? |
|---|---|---|---|
| `benchmark/` (140) | **curated before either model existed** (user, 2026-09-02) — unbiased by construction, the strongest provenance on the branch | dev heavily used for tuning; **test spent once** (Stage 6, no parity/win) | **No** — contains zero NN events |
| `benchmark_new/` (485) | curated with prediction columns visible but not consulted — unbiased by curator attestation | dev scored (Stage 10); **test half unspent, one shot** | Yes — 159 NN, 79 on dev |
| Track B arenas | automated, model-blind by construction | renewable, single-use per model generation | Yes, but at an unmeasured noise level |

**`benchmark/` is probably the cleaner set, not merely the older one**
(user, 2026-09-02). Three reasons, and they are about label quality, not
provenance:

* Every one of its 140 events is a curator-**affirmed** positive, with the
  IGV evidence retained (`benchmark/igv_images/`). Candidates that did not
  hold up are simply absent from the file.
* `benchmark_new` retains the entire 573-candidate pool by design, so 159 of
  its events are NN-because-the-candidate-failed, 88 were too ambiguous to
  score at all, and **26% of the scored rows are TRASH** (126/485). It is a
  deliberately harder and dirtier set.
* The accuracy gap corroborates this rather than contradicting it: every
  model measured runs 77–87% on `benchmark/` dev against 58.7% on
  `benchmark_new` dev, and 68.6% even on `benchmark_new`'s *clean* subset
  (Stage 10, Stage 4).

So the two sets trade off in opposite directions — `benchmark/` is cleaner,
`benchmark_new` is more deployment-representative — and `benchmark_new`'s one
structural advantage is that it has an NN class at all, which is the only
reason this stage leans on it. Treat disagreement between the two dev halves
in A3 as **informative about which regime a model is winning in**, not as
noise to be averaged away.

Track B exists so that there is a fourth, fifth and sixth arena after both
are exhausted.

## Measurements taken while writing this plan (record them, don't re-derive)

All computed locally from `benchmark_new/curation.tsv`, 485 unambiguous rows,
candidate class = `csv_source` mapped through `{A3:A3, A5:A5, ES:CSE, IR:RI}`:

| question | answer |
|---|---:|
| automated rMATS×SUPPA2 *class* label vs curated truth, conditional on the event being real | **304/326 = 93.3%** |
| fraction of consensus candidates that are real events at all | **326/485 = 67.2%** (159 became NN under curation) |
| candidate == truth, all 485 | 304/485 = 62.7% |
| same, clean windows (`trash == 0`, n=345) | 76.2% |
| same, TRASH windows (`trash == 1`, n=126) | **28.6%** |

Can caller statistics separate real from NN? **No.** Purity (fraction with
`truth != NN`) is flat at 0.66–0.73 across every threshold tried on
`rmats_pvalue`, `|suppa_dpsi|`, `|rmats_incleveldiff|`, and their conjunction
(`|dpsi|>=0.3 & |dinc|>=0.2 & p<=0.01` → n=96, purity 0.708). Median
`rmats_pvalue` is in fact *lower* for the NN group (0.0016) than the real one
(0.0001 — same direction, both tiny). Full crosstab:

```
truth  A3  A5  CSE   NN  RI
cand
A3     30   1    1   26   6
A5      0  22    1   11   3
CSE     0   0  209  102   5
RI      3   0    2   20  43
```

**Consequence, load-bearing for Track B:** an uncurated consensus benchmark
carries ~33% positive-label noise that no stringency filter removes. That
makes it unusable for *absolute* accuracy claims and perfectly usable as a
**model-blind paired arena**, since symmetric label noise attenuates a
v1-vs-v2 difference without biasing it.

---

## Track A — real supervision from annotation, without curation

**The idea.** The curators were judging *differential usage*. The model's
five classes are *event types*, and `sim/events/*_strict.ioe` already defines
those geometrically, for free, at 10^4 scale. So real labelled training data
does not require curation at all — only a decision about which catalogued
event a window carries and whether it is present in the sample.

**Why it matters.** Stage 11's real-supervision ceiling (11.4%/14.3% NN
recall, n=242) could not distinguish "NN is intrinsically hard in this
feature representation" from "242 events is far too thin for a junction-set
transformer to learn NN". Track A is the only proposed experiment that
separates those two readings, because it moves n from 242 to ~10^4.

### A0 — label semantics and presence filter *(strong model)*

Write the design decision down before writing code:

* **Label**: the class of the uniquely-assigned catalogued event overlapping
  the window (A3/A5/SE→CSE/RI). Reuse `sim/catalogue.py`'s `Event` objects
  and the same exclusions (MX/AF/AL excluded, SE→CSE) so the class definition
  is bit-identical to the simulated side.
* **Uniqueness**: one event per gene and no two catalogued events sharing a
  window, via the machinery already in `sim/select_events.py`. Overlapping
  catalogue entries are the main source of label ambiguity here.
* **Window**: `sim/window.py` v3 unchanged (event span + U[0,100] bp per
  edge, label-independently). Do not invent a fourth sampler.
* **Presence filter**: a BAM-derived support/coverage criterion — the event's
  distinguishing junctions carry ≥ k reads in at least one condition, and the
  window clears a coverage floor. Not model-derived, not dPSI-derived.
* **NN**: expressed windows with no catalogued event overlap, drawn the way
  `select_events.py` draws `nn_plain`/`nn_hard`, and **subject to the same
  expression floor as positives**.

**Known hazard, to be measured rather than assumed:** a support-based
presence filter uses signal the model itself sees, so real-derived NN could
become trivially separable ("no junctions here"). Mitigations, both required:
(i) the shared expression floor above; (ii) a GBM control trained *only* on
the presence-filter quantities, reported alongside — if that control already
separates NN, the real NN class is a shortcut and must not be used as
evidence about NN recall.

### A1 — pre-registered local gate *(strong model)*

Run the A0 labeller over `benchmark_new`'s 485 curated windows and
`benchmark/`'s 140. Both BAM sets are local and window-identical to the
committed CSVs, so this costs **no pasteur time** and can kill or reshape the
track cheaply. Report the confusion vs curated truth, per class, split by
`trash`.

Pass criteria, declared here before the run:

1. **non-NN agreement ≥ 0.80.** Reference ceiling is the consensus
   pipeline's 93.3% conditional-on-real, measured above.
2. **NN recall vs curated NN ≥ 0.50 on clean (`trash == 0`) windows.** For
   scale: zero-shot v2 gets 0.09–0.14, and the consensus pipeline gets 0 by
   construction (it emits no NN candidates).

Outcomes:

* both pass → A2.
* (1) passes, (2) fails → A2 proceeds **for the four positive classes only**;
  NN stays simulated, and Track A makes no claim about NN recall.
* (1) fails → **Track A stops.** Write the negative result up as a stage
  result doc and redirect the compute to B and D. This is a real possible
  outcome and is not a failure of the session.

### A2 — extraction at scale on pasteur *(cheap model for mechanics, strong for conventions)*

First source: `c1_new.bam`/`c2_new.bam` (SRSF3 KD vs control), already
aligned and resident on pasteur.

**Leakage rule, non-negotiable:** those are the same BAMs `benchmark_new` was
built from. Every extracted event must be gene-disjoint from all 573
curation rows — enforced with `select_events.py --exclude-genes`, the same
mechanism the simulated batches use, because `GroupKFold` cannot see across
a training pool and a benchmark file. Verify the disjointness explicitly and
assert it, do not assume it.

Target ~10–15k events from that pair, extended with Track B's aligned pairs
once those exist, for dataset diversity. Only `.npz` + labels come back.
`limitram`, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`, `nice -n 19`, check
`/proc/loadavg` before launching and a minute after, throttle past ~50.

### A3 — three arms on identical folds *(strong model)*

Route everything through `baseline_gbm.grouped_folds` / `report` /
`report_collapsed` so the numbers stay comparable to the branch's history.

| arm | training data |
|---|---|
| sim-only (reference) | `npz_union2` / current `e4_union2` recipe |
| real-only | Track A's pool |
| transfer | sim-pretrain → real-finetune, and sim+real co-train |

Evaluated on: real-pool grouped-by-gene OOF (in-domain), plus
`benchmark_new` **dev** and `benchmark/` **dev** (transfer). McNemar on every
pairwise comparison. Headline metrics are **NN recall** and collapsed
accuracy — not overall accuracy, which Stage 10 showed is dominated by the NN
failure anyway.

### A4 — verdict *(strong model)*

Decide from A3's table which arm becomes the line (the decision deliberately
deferred at planning time): sim-only stands, sim+real co-training, or real
primary with simulation as augmentation. Write the results doc, including
whichever arm lost.

### A5 — the single test spend *(strong model)*

**`benchmark_new`'s test half (243 events) is spent once, on Track A's
winning arm vs the v1 six-net ensemble**, after A4 and not before. One shot,
paired McNemar, pre-registered as: v2's winning arm vs v1, collapsed
accuracy, with 5-class and NN recall reported alongside. Nothing else in this
plan reads the test half.

**Open decision, to be settled at A4 with the numbers in hand, not now:**
whether to *also* re-spend `benchmark/`'s test half (70 events, spent once at
Stage 6) on the same winning arm. The case for it is that `benchmark/` is the
cleaner set (see the table above) and the only one whose curation predates
both models entirely, so a v1-vs-v2 result there has no possible curation-era
model influence of any kind. The case against is that it is a **declared
re-use**, not a fresh spend — the Stage 6 outcome is already known, which
weakens any second reading — and at n=70 it needs 8–10 discordant events for
significance, so it is unlikely to resolve anything `benchmark_new` test does
not. If it is done, it must be reported as a re-spend with the Stage 6 result
quoted next to it, never as a fresh pre-registered arena.

Recommended default: spend `benchmark_new` test alone, and only add the
`benchmark/` re-spend if A3's two dev halves **disagree** about the winning
arm — that is exactly the case where the cleaner set is worth a second, openly
declared reading.

---

## Track B — a renewable, model-blind arena

**Purpose after the correction above.** It is no longer "we need an
uncontaminated arena" — `benchmark_new` test is clean. What remains is that
`benchmark_new` test is *single-use*, and this branch has now exhausted two
benchmark halves across its life. Track B makes arenas renewable, and adds
RBP/cell-type diversity that one SRSF3 knockdown cannot.

### B0 — freeze the accessions *(cheap model)*

Choose and **commit into this plan file** 2–3 ENCODE shRNA-KD/control pairs
before any scoring: RBPs distinct from SRSF3, and a second cell line
(K562 alongside HepG2) if the matched controls exist. Pre-registration means
the accession list is frozen in git before a single number is produced.

### B1 — fetch and align *(cheap model, long-running)*

`sim/fetch_align_benchmark_new.sh` per pair on pasteur. This is the long pole
(fetch + STAR, several hours per pair) and should be launched **first**, in
parallel with A0/A1's local work. Sequential across pairs, `nice`d,
`limitram`ped, thread-capped.

### B2 — automated candidates, no human, no model *(strong model for the NN definition)*

rMATS + SUPPA2 + `benchmark_build/consensus.py` as in Stage 8, unchanged.
Add automatically generated **NN candidates**, which the existing pipeline
does not emit: catalogued event loci where *both* callers report clearly
non-significant, near-zero dPSI, above the same expression floor A0 defines.
That NN definition is itself noisy — it is the automated stand-in for what a
curator decided by eye — and its noise level is unknown, unlike the 33%
figure measured for positives. Say so in the arena's README.

**No model prediction column is written into any file in this track**, at any
stage, so the arena is model-blind by construction rather than by attestation.

### B3 — freeze, then one scoring pass *(strong model)*

Commit the arena CSVs and a README recording: the 67.2%/93.3% purity figures
above as the expected attenuation, the unknown NN noise level, and the
declaration that **absolute accuracies from this arena are not quotable as
accuracy**. Then score v1's six-net ensemble against the current best v2,
paired McNemar, per-dataset and pooled. Only the paired difference is
interpreted.

### B4 — reuse policy *(documentation)*

Each frozen arena is **single-use per model generation**. The next generation
gets fresh accessions. That is the entire point of making it renewable, and
it is also the only thing that keeps it honest.

---

## Track C — close-outs

### C0 — record the curation correction *(cheap model)* — do this first

Add the curator's attestation next to each occurrence of the contamination
caveat listed at the top of this file. Do not delete the caveats; the
convention is that retractions stand next to what they retract.

### C1 — answer Stage 11's mechanistic question *(strong model, small)*

Per `resume-8`'s own instruction: answer it locally from data in hand, not
with another blind pasteur pilot. Why does elevating `premrna_frac` not move
`intron_cov_frac` proportionally for CSE/NN the way it does for RI? Check
(a) how many junctions per class survive `MIN_INTRON_FOR_COV` and the flank
window before the ratio is computed, (b) the per-class intron-length
distribution, (c) where an injected pre-mRNA row's coverage actually lands,
interior vs flank. Then write a verdict: the TRASH layer is revivable with a
specific fix, or it is shelved. Either is an acceptable outcome; leaving it
undecided is not.

### C2 — record this session's measurements *(cheap model)*

The table and crosstab at the top of this file, into the stage results doc,
so no future session re-derives them or re-proposes "just automate the
consensus benchmark" without knowing the purity ceiling.

### C3 — `benchmark_new` README *(cheap model)*

Record that the test half is held for A5 specifically, and why, so it is not
spent by accident on an intermediate result.

---

## Track D — the representation question, gated on A3

Deliberately **not** chosen in advance. A3's outcome selects the branch:

* **real-only ≫ sim-only** → the bottleneck was the sim-to-real domain gap.
  D becomes self-supervised pretraining on unlabeled real windows (masked
  coverage / junction modelling), which is cheap because unlabeled real
  windows are effectively unlimited, plus more real data.
* **real-only ≈ sim-only, both weak on NN** → the bottleneck is the
  representation. D becomes an encoder change: a binned log-coverage 1D-CNN
  branch per condition, fused into the existing junction-set transformer, on
  the hypothesis that the hand-derived `[J,48]` + `[32]` features do not
  carry the intronic-continuity evidence NN needs. Note this touches
  `N_TOKEN_FEATURES`/`N_EVENT_FEATURES` and therefore every stored scaler,
  and the three `apply_feature_subset*` call sites.
* **real-only ≪ sim-only** → the presence filter or the annotation labels are
  wrong; go back to A0 rather than forward to D.

One new arm at a time, through the existing harness, McNemar on every
comparison, and **no test spend in this track** — A5 is the only one.

---

## Sequencing

| when | pasteur | local / GPU |
|---|---|---|
| immediately | B1 fetch+align (long pole) | C0, then A0 + A1 gate |
| after A1 passes | A2 extraction (**after** B1 finishes — do not overlap, see the load rules) | C1, C2 |
| then | B2 candidates | A3 three arms |
| then | — | A4 verdict, B3 arena scoring |
| last | — | **A5: the single test-half spend** |

A0/A1 is roughly half a day of local work and can kill Track A cheaply, which
is why it runs before any A-track pasteur time. B1 is launched first only
because it is fetch/align-bound and independent of everything else.

## Model tiers, summarized

* **Strong**: A0 (label semantics — the crux of the whole track), A1 gate
  interpretation, A3, A4, A5, B2's NN definition, B3, C1, all of D.
* **Cheaper**: A2's rsync/launch/monitor mechanics, B0, B1, C0, C2, C3.

## Conventions (unchanged)

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar. n=70 needs 8–10 discordant
  events; n=140 needs 5–6; `benchmark_new`'s halves (n≈242) need ~10–14.
- Never delete a measured number from a docstring or plan doc — retractions
  and corrections go next to what they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction on pasteur; wrap
  pasteur jobs in `limitram`, cap all four thread-pool env vars, `nice -n 19`,
  check `/proc/loadavg` before and after launching.
- The standing gates still apply after any re-simulation or extractor change:
  `dsignal.verify`, `dsignal.verify_features`, `sim.width_leak_check`,
  `sim.psi_recovery_check`, `dsignal.audit_noise compare --by-label`.
