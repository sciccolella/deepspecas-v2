# signal-native-v2 — START HERE (2026-09-08, Stage 15)

Resume document. Branch `signal-native-v2`, **nothing pushed** (per repo
convention — pushing is the user's call).

Supersedes `plans/signal-native-v2-resume-8.md` as the entry point.
`plans/signal-native-v2-real-supervision.md` is the Stage 12 plan and
`plans/signal-native-v2-real-supervision-results.md` is its full
stage-by-stage record. Stage 13 (B3) is
`plans/signal-native-v2-arena-b3-plan.md` / `plans/signal-native-v2-arena-b3.md`.
Stage 14 (the NN attack) is `plans/signal-native-v2-stage14-nn-attack-plan.md`
/ `plans/signal-native-v2-arena-nn-attack.md` — **read that Results section
before touching NN recall again**, its finding is the most important thing
in this file. Stage 15 (`plans/signal-native-v2-stage15-17-shortcut-attack-plan.md`
/ `plans/signal-native-v2-stage15-shortcut-diagnostic.md`) confirms Stage
14's failure mechanism; Stage 16/17 (same parent plan) attack it and are
in progress — see "What is running right now" below.

---

## The thing to know before the other three: Stage 14 (the NN attack) failed its gate

**Real NN hard negatives move NN recall a lot, but the model pays for it
everywhere else, badly and consistently across 3 seeds — the pre-registered
gate correctly rejects it.** 474 real NN events from a second automated arena
(AQR knockdown, `aqr_arena/`, gene-disjoint from `ptbp1_arena`/`benchmark`/
`benchmark_new`) were merged unweighted onto the 12,480-event sim pool
(`npz_union2_realnn`, NN share 27.7%) and trained at 3 seeds. On `benchmark_new`
dev NN recall jumps 13.9% → median 57.0% (passes), but pooled collapsed
accuracy **drops** 58.7% → median 52.1% (14-20 fewer correct out of 242,
1.4-2× the branch's own measured seed-noise ceiling at that n) with **zero**
`new-only` wins on A3/A5/CSE/RI in any seed — every other class just gets
worse. Same story on `benchmark/` dev (regression guard): 53/70 → median
45/70, also beyond noise, in all 3 seeds. **Gate fails on condition 3, badly
and uniformly.** Not folded into any ensemble; `e4_union2.json` is untouched.

The interesting part is *why*, even though it's unverified: the 474 real rows
are the **only** real BAM data in the entire 12,954-event pool (everything
else, positive or NN, is simulated) and are labelled NN uniformly. NN recall
gains are largest exactly where the eval set is real BAM data (`benchmark_new`
dev +43pp, `ptbp1_arena` confirmatory +64-68pp) and **in-sim held-out macro-F1
doesn't move at all** (0.905-0.911, same as before) — consistent with the
model having learned "looks like real, not simulated, data" → NN, rather than
the intended splicing-vs-no-splicing distinction. Full numbers, per-seed
table and the confirmatory `ptbp1_arena` read (pooled accuracy roughly flat
there — NN gain and other-class losses happen to net out on a 33%-NN set) are
in `plans/signal-native-v2-arena-nn-attack.md`'s Results section. This is the
fourth negative result on the NN-recall line, after Track A's 0.804 cap,
Track D's reallocation, and B3's seed-variance finding — but the first one
that actually *moves* NN recall, which is why the mixing hypothesis above is
worth testing before writing off real-data injection entirely (see the "for
whoever picks this up next" note at the end of that doc).

## The three things to know (Stage 12/13, still current)

**1. `benchmark_new`'s "descriptive, not pre-registered" caveat is retracted.**
The curator (the user) states the `v1_pred`/`v2_pred` columns were visible in
the review sheets but **did not inform the manual assessment**. So the
dev-half v1-vs-v2 result (`e4_union2` 58.7% vs v1 41.3% 5-class, p<0.0001) is
a real result, and **`benchmark_new`'s test half is a legitimate
pre-registered arena**. The caveats stay in the earlier docs with the
retraction beside them, per the never-delete convention.

Related, and stated by the user: **`benchmark/` is probably the *cleaner*
set**, not merely the older one — every one of its 140 events is a
curator-affirmed positive with IGV evidence retained, whereas `benchmark_new`
keeps the whole 573-candidate pool (26% TRASH, 159 NN-because-the-candidate-
failed). They trade off in opposite directions; `benchmark_new`'s one
structural advantage is having an NN class at all. See the three-benchmark
table in the plan.

**2. Track A is dead, and for a reason worth keeping.** Annotation-derived
labels cannot replace curation: the pre-registered gate failed at 0.607 non-NN
agreement (needed 0.80) and 0.314 NN recall (needed 0.50), with the same 0.614
on `benchmark/`. Not a tuning miss — the catalogue holds the correct label for
**97.5%** of non-NN dev windows, but ~2 *other* annotated events per window are
equally "present" by any BAM support criterion, capping a perfect
disambiguator at **0.804**. Curated-NN windows average 1.97 present events: a
curated NN is not a window without annotated splicing, it is one where a human
judged nothing convincing was happening, and no support threshold sees that.

**3. The coverage branch (Track D) is a reallocation, not an improvement.**
Three paired seeds: on `benchmark/` dev it looks like a large win (55.7 →
61.7 of 70, wins all three pairings), on `benchmark_new` dev it is a null
(121.7 → 122.7 of 242). It makes the same trade on both — buying A3/A5/CSE
and paying in **RI** (−0.167, −0.143) — and the verdict flips purely on class
mix. **NN recall, the reason it was built, does not move** (0.114 → 0.110).
Not adopted; the code stays, defaulted off.

## The methodological finding that outlives all three

**Baseline seed spread alone is 5 events at n=70 (53/56/58) and 10 at n=242
(117/127/122)** — the same magnitude as most "improvements" this branch has
ever compared. Seed 0 alone would have supported a +9-event claim at
p=0.023; three seeds cut it to +6 on one benchmark and +1 on the other.
**Seed-replicate any future comparison before believing it.**

## What is running right now (updated 2026-09-08)

Stage 15 is complete (diagnostic, no training; confirmed the shortcut
mechanism, see caveat #5 above). Stage 16/17 are in progress — this session
is working through `plans/signal-native-v2-stage15-17-shortcut-attack-plan.md`.
`ptbp1_work/fastq` (79G) and `aqr_work/fastq`+`align` still hold pair 1's and
pair 2's FASTQ/BAMs on pasteur, if a rebuild is ever needed — neither leaves
pasteur per CLAUDE.md. The 3 Stage 14 checkpoints
(`runs/nn_attack_realnn_seed{0,1,2}/`) are local only (gitignored like every
other `runs/*/` dir); their `.json`/`.log` and all scored `.tsv` outputs are
committed.

## Track B state — B2 and B3 both complete

`ptbp1_arena/` (see its README, which carries every caveat): **2,422 events**,
1,073 CSE / 800 NN / 284 RI / 152 A3 / 113 A5, `dsignal.verify` 4,844/4,844.
The BAM subsets are gitignored (881MB); the CSVs, `consensus.tsv` and the BEDs
that define the arena are committed.

**B3 ran** (Stage 13, `plans/signal-native-v2-arena-b3.md`, pre-registered
before scoring). Headline: v1 six-net ensemble vs v2's `e4_union2` at
`--nn-weight 0.1`, pooled `correct_collapsed` over all 2,422 events —
**v1 881/2422 = 36.4%, v2 1175/2422 = 48.5%, McNemar p<0.0001. v2 wins
pooled and every per-class recall except RI** (v1 89.4% vs v2 75.0%).
**NN is v2's largest per-class deficit** (16.9% recall vs next-lowest A3's
59.2%) at the as-deployed operating point; a prior-free secondary run shows
NN recall at 56.0% with no prior correction, confirming the deficit is
partly an artefact of `nn_weight=0.1` being tuned on a 0%-NN benchmark, not
purely a modelling failure. Per the pre-registered decision rule: **Stage 14
is the NN attack** — build arena pair 2 from a different KD experiment and
use its NN as hard negatives, holding pair 1 (this arena) as eval, checking
any resulting move against curated `benchmark_new` dev before trusting it.

**Stage 14 ran and failed its gate** (`plans/signal-native-v2-arena-nn-attack.md`) —
see the top section of this doc for the headline and the hypothesis for
next steps. `aqr_arena/` (pair 2, AQR knockdown, 2,016 events / 800 NN) is
built, verified and committed the same way `ptbp1_arena` was, but is never
itself scored — its only role was supplying 474 gene-disjoint real NN rows.
Pair 3 is not built; nothing currently plans to.

### Three bugs the first arena build hit, all fixed in the committed scripts

Worth knowing because pair 2 should now run end to end: rMATS could not import
`rmatspipeline` (it lives in the install dir, not site-packages, and
`rmats.py`'s shebang wants a bare `python` the env lacks); SUPPA2 diffSplice
exits 0 leaving `.dpsi.temp.0` and never renames it on this install; and
`make_candidate_csvs.py` did not create its output directory.

### Two things caught in the NN half before anything was scored

1. **`p == 1.0` in a SUPPA2 `.dpsi` means "not measurable", not "unchanged"** —
   `dPSI` is NaN for exactly those rows and no others, and a measurable
   permutation p-value never exceeds ~0.485. The originally pre-registered
   `p > 0.5` therefore selected the empty set. Amended to `p > 0.20` plus an
   explicit non-NaN guard, recorded as a dated amendment in
   `benchmark_build/make_nn_candidates.py`.
2. **The BAM subsets were cut over the positives-only BED**, leaving 11 of the
   first 12 NN windows at zero coverage — every NN would have scored correct
   for a model that had learned nothing but a coverage detector. Recut over all
   2,422 regions. If you ever rebuild them, use
   `candidates/all_events.sorted.bed`.

### Leak checks on the arena (measured, not assumed)

| shortcut | arena | curated `benchmark_new` |
|---|---:|---:|
| width alone (AUC) | 0.493 (p=0.57) | — |
| mean coverage alone (AUC) | 0.417 (p=0.004) | 0.564 (p=0.050) |

No width leak. The coverage skew is small and — the interesting part — the
*same magnitude but opposite sign* as in curated data: curated NN skews to
higher coverage, automated NN slightly lower. A model leaning on depth would be
rewarded on one set and punished on the other.

## Track B provenance

* **B0 done, frozen in the plan**: PTBP1 KD `ENCSR064DXG`
  (SRR3469390/SRR3469391) vs its own ENCODE `possible_controls`
  `ENCSR603TCV` (SRR3469516/SRR3469517), HepG2, all verified against the
  ENCODE portal and NCBI SRA.
* **B1 done**: fetched and aligned on pasteur.
  `sim/fetch_align_benchmark_new.sh` now takes its runs and output prefix
  from the environment; unset, it reproduces the SRSF3 invocation exactly.
* **B2 done**: `benchmark_build/run_arena.sh` scripts what Stage 8 did
  interactively (an arena that cannot be rebuilt is not renewable), and
  `benchmark_build/make_nn_candidates.py` adds the NN half.
* **B3 done** — see above; it spent the arena's single use.
  The NN half's own noise level remains **unmeasured**, unlike the positives'
  67.2% purity: there is no curated set of automatically-derived NN to check
  it against. Same caveat applies unchanged to pair 2's (`aqr_arena`) NN half.
* Pair 2 (AQR, `aqr_arena/`) built for Stage 14, real NN only. Pair 3 not
  frozen — nothing currently plans to build it.

## Numbers measured this stage, so they are not re-derived

From `benchmark_new/curation.tsv` (485 unambiguous rows): the rMATS × SUPPA2
consensus **class** is right for **93.3%** of events *that are real*, but only
**67.2%** of candidates are real, and **no** caller-statistic threshold
separates the two (purity flat at 0.66-0.73 across `rmats_pvalue`,
`|suppa_dpsi|`, `|rmats_incleveldiff|` and their conjunction). Candidate ==
truth is 76.2% on clean windows and **28.6%** on TRASH.

**C1 (Stage 11's open mechanistic question) is answered.**
`MIN_INTRON_FOR_COV` is eliminated (0-4% of drops). The window-edge rule is
the dominant filter but explains **A3/A5**, not CSE/NN — only 9-24% of real
A3/A5 windows yield a measurable `intron_cov_frac` at all, so Stage 11's
"A3 10x short" was never a calibration reading. Within a domain the metric is
largely set by window geometry (`icf × intronic/exonic ≈ 1` in all five
classes), which is the mechanism behind the `--by-label` fix. **But the
CSE/NN gap survives geometry matching** (CSE 2.9-9.4×, NN 4.1-8.0×), so Stage
11's central finding stands. The knob is sublinear because the pre-mRNA row is
a fraction of the gene's *reads* spread over pre-mRNA length, scaling its
effect by a per-gene length ratio — **the TRASH layer is revivable only by
solving per row for the target metric, never by raising the knob uniformly.**

## Open honesty caveats

1. **`nn_weight=0.1`** and **`DPSI_BETA`/`PSI_LEVEL_BETA`** — unchanged from
   resume-7/8.
2. **`benchmark_new`'s test half (243) is unspent and reserved** for one
   specific measurement: Track A's winning arm vs v1 (plan step A5). Track A
   died, so **that spend is not owed to anything right now** — do not spend it
   on an intermediate result. `benchmark/`'s test half stays spent (Stage 6).
3. **A1's thresholds were pre-registered; the sweep around them was inspected
   afterwards on dev.** Nothing was carried forward (the track stopped), so
   nothing is fitted — but if the route is revisited, pre-register a fresh
   criterion rather than reading one off that table.
4. **Track D's 88.1% dev figure must not be quoted as a headline.** It is what
   a class-mix-favourable benchmark says about a model that got worse at RI.
5. **Stage 14's real-vs-simulated-shortcut mechanism is confirmed** (Stage 15,
   `plans/signal-native-v2-stage15-shortcut-diagnostic.md`). NN-prediction
   rate on 1,622 real non-NN-truth `ptbp1_arena` events: baseline 2.3%, all 3
   Stage-14 seeds 48.0-53.3% (McNemar p < 1e-220 each, `b=0` — no event moves
   away from NN), uniform across A3/A5/CSE/RI. The 474 real NN rows being the
   only real BAM data in the pool, labelled NN uniformly, is the mechanism,
   not a plausible story.

## Conventions (unchanged)

- Never `git push`; never create or switch branches unprompted.
- Every benchmark comparison needs McNemar — **and now, seed replication**.
  n=70 needs 8-10 discordant events; n=140 5-6; n=242 roughly 10-14.
- Never delete a measured number from a docstring or plan doc — retractions
  and corrections go next to what they retract.
- Build → verify → commit each step before starting the next.
- Training local (`DSnet`), simulation/alignment/extraction/arena building on
  pasteur; wrap pasteur jobs in `limitram`, cap all four thread-pool env
  vars, `nice -n 19`, check `/proc/loadavg` before and after.
- Gates: `dsignal.verify`, `dsignal.verify_features` (**now six checks** —
  check 6 covers the coverage profile's symmetries), `sim.width_leak_check`,
  `sim.psi_recovery_check`, `dsignal.audit_noise compare --by-label`.
