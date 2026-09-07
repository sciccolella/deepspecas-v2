# Stage 14 — the NN attack: pre-registration

Written and committed per `plans/signal-native-v2-stage14-nn-attack-plan.md`
Step 0, **before** Step 2 pulls any FASTQ. Nothing below this line, up to
the "## Results" heading, may be edited once building/scoring starts.

## Entrants and pool composition

- **Baseline (pre-attack)**: `runs/ensembles/e4_union2.json` (6 members),
  scored at `--target-prior nn-weight --nn-weight 0.1 --tta-condition-swap`.
  Existing scoring artifacts, read off (not re-derived):
  - `benchmark_new` dev (242 events): `runs/bnew_dev_e4_union2.tsv` /
    `.log` — **aggregate collapsed 142/242 = 58.7%**, **NN recall 11/79 =
    13.9%**.
  - `benchmark/` dev (70 events): `runs/benchmark_events_stack_dev.tsv`
    (the single-model `geomwin_iv30_refit` checkpoint, per CLAUDE.md's
    reproduction recipe) — **53/70 collapsed**. This is the regression-guard
    number Stage 13/B3 also used unchanged.
- **Sim labels file to merge against**: `sim/experiment/labels_union2.n12480.tsv`.
  Cross-checked against `runs/ensembles/e4_union2.json`'s per-member
  `"labels"` field: 5 of 6 members (`scale_union2_n12480`, `ens_union2_allfeat`,
  `ens_union2_twohead`, `ens_union2_depthscale`, `ens_union2_augsub`) use
  `labels_union2.n12480.tsv`; only `ens_union2_nn15` uses the
  `.nn15.tsv` NN-upsampled variant. This attack merges against the
  majority/base file, `labels_union2.n12480.tsv` (12,480 events, `npz_union2`),
  consistent with the fixed "as-is, unweighted" mixing decision — the nn15
  variant is itself an upsampling knob and would confound attribution.
- **Pool composition**: pair-2 (AQR, see below) **NN-only** rows, added
  unweighted on top of `npz_union2` / `labels_union2.n12480.tsv`. No pair-2
  positives (their purity is as unmeasured as pair 1's — this is a targeted
  NN fix, not a general real-data upgrade). No upsampling, no
  `--class-weight-boost`.

## Pair 2

**AQR knockdown**, `ENCSR431SXF` vs its ENCODE `possible_controls` entry
`ENCSR225PRV`, both HepG2 — chosen over the PRPF4 backup for maximum
mechanistic distance from PTBP1 (hnRNP, pair 1) and SRSF3 (SR protein,
`benchmark_new`): AQR is a core intron-binding spliceosomal factor.

### Step 1 — frozen accessions (verified against ENCODE portal + NCBI SRA, 2026-09-07)

| condition | ENCODE | GEO | runs |
|---|---|---|---|
| AQR shRNA KD, HepG2 | `ENCSR431SXF` | GSE177214 | rep1 SRR14843100, rep2 SRR14843099 (one run each, paired-end 100bp, HiSeq 2500) |
| non-target shRNA control, HepG2 | `ENCSR225PRV` | GSE176746 | rep1 SRR14840064 (one run); rep2 SRR14840062 + SRR14840063 (two runs) |

`ENCSR225PRV` is confirmed as `ENCSR431SXF`'s sole `possible_controls` entry
(fetched directly from the ENCODE experiment JSON), same relationship
`ptbp1_arena/README.md`'s condition table documents for pair 1. Same cell
line/protocol/assay as pair 1 and `benchmark_new`'s SRSF3 pair; different RBP.

**Correction to the Stage 14 plan draft** (`plans/signal-native-v2-stage14-nn-attack-plan.md`
Step 1): it states the control's rep1 is "SRR14840064+SRR14840061". Verified
against the ENCODE experiment JSON (`biological_replicates` field per file)
and cross-checked on NCBI SRA: **`SRR14840061` is not part of this
experiment at all** — it is "microRNA-seq from ovary" (`SRX11164272`, a
different SUNY Albany RIP-seq submission that happens to share the same
`SRR148400xx` accession block). `ENCSR225PRV` has exactly 3 runs total:
rep1 = `SRR14840064` alone (no second run to concatenate), rep2 =
`SRR14840062` + `SRR14840063` (the two-run replicate the plan anticipated,
just on the wrong rep number). The concatenation pre-step in Step 2 is
therefore needed for **rep2 only**, not rep1. This retraction stays here
per the never-delete convention; the plan file itself is not edited.

## Gene-disjointness rule

Pair-2's training genes (its NN-candidate `gene_id` column, direct) must be
disjoint from:

1. **`ptbp1_arena/`** — both `consensus.tsv` and
   `candidates/nn_candidates.tsv`'s `gene_id` columns (both files carry
   `gene_id` directly; pair 1 stays held-out eval, never touched by this
   attack's training pool).
2. **`benchmark_new/`** and **`benchmark/`** — both still-live curated sets.
   Neither carries a `gene_id` column (only `region`, and `gene_symbol` in
   `benchmark_new/curation.tsv` but not consistently across the raw
   per-class CSVs), so "same gene" is approximated the same way
   `dsignal/benchmark_split.py` already does for its own dev/test split:
   single-linkage clustering of genomic regions at `--link 50000` (50 kb) on
   the same chromosome, via `benchmark_split.cluster()`. Pair-2's NN
   candidate regions are unioned with `benchmark_new`'s and `benchmark`'s
   region sets before clustering; any pair-2 region landing in a cluster
   that also contains a `benchmark_new`/`benchmark` region has its `gene_id`
   added to the exclusion set (all pair-2 candidates sharing that gene_id
   are excluded, not just the one clustered region).

Output artifact: `sim/experiment/exclude_genes_nn_attack.tsv` (one `gene_id`
column, pooled from all three sources above), built in Step 3. Asserted
empty-overlap after filtering — same visible-assertion style
`sim/select_events.py` already uses (print excluded/kept counts, hard
`assert` on the post-filter intersection being empty).

## Evaluation order and go/no-go gate

Fixed now, applied mechanically in Step 8:

1. **`benchmark_new` dev — primary gate.** `dsignal.score_benchmark --half dev
   --split benchmark_new/split.tsv --tta-condition-swap --target-prior
   nn-weight --nn-weight 0.1`, then `dsignal.benchmark_split compare` against
   `runs/bnew_dev_e4_union2.tsv` (baseline above), `--by-class` for the
   NN-recall read.
2. **`benchmark/` dev — regression guard.** Same protocol against
   `runs/benchmark_events_stack_dev.tsv` (53/70 baseline above).
3. **`ptbp1_arena` pair 1 — confirmatory only.** `--half both` against
   `runs/arena_v2_e4union2.tsv`, `--by-class`. Scored once per seed, never
   used to pick a seed or checkpoint. Any NN-recall movement on pair 1
   alone is **not sufficient** — its NN purity is as unmeasured as pair 2's
   own (Trap 2 in the B3 doc applies symmetrically here).

**Go/no-go gate (fixed, mechanical)**:

- **PASS** requires all three:
  1. Median-over-3-seeds `benchmark_new` dev NN recall beats the 13.9%
     (11/79) baseline.
  2. ≥2/3 seeds individually beat the 13.9% baseline on `benchmark_new` dev
     NN recall.
  3. No more-than-seed-noise regression on `benchmark_new` dev pooled
     collapsed accuracy (baseline 58.7%, 142/242) or on `benchmark/` dev
     (baseline 53/70) — "seed noise" read against the branch's own measured
     spread (5/70 at n=70, 10/242 at n=242, per `plans/signal-native-v2-resume-9.md`).
- Anything else is **FAIL** — including a pass on `ptbp1_arena` pair 1 alone
  with a fail/null on `benchmark_new` dev.

## Seeds

3 seeds: train seeds 0, 1, 2 — matching the branch's seed-replicate
convention (`plans/signal-native-v2-resume-9.md`, "seed-replicate any future
comparison before believing it").

## Standing constraints (unchanged, restated for this attack)

- `benchmark_new`'s test half stays reserved, untouched by this attack.
- Never delete a measured number from this doc or any other; retractions go
  next to what they retract.
- Build → verify → commit each step; never `git push`; never create/switch
  branches unprompted.
- pasteur jobs: `limitram`, all four thread-pool env vars capped, `nice -n
  19`, `/proc/loadavg` before and after.

## Results

Steps 0-7 executed exactly as pre-registered above (aqr_arena built and
verified, `dsignal.verify` 4,032/4,032 arena-wide + 948/948 after extraction;
gene-disjointness applied, 326/800 NN candidates dropped, 474 kept, zero
gene overlap asserted; merged unweighted onto `labels_union2.n12480.tsv` ->
`labels_union2_realnn.tsv`, 12,954 events, NN share 27.7%; 3 seeds trained,
in-sim macro-F1 0.905/0.911/0.907). Full commit trail: pre-registration
`da40e8b0`/`33b09a92`, arena build `b4401522`, extract-CSV `07155235`, npz
extraction `0c91db6a`, merge `5ce67f07`, training `<the commit that follows
this results write-up>`.

### benchmark_new dev (primary gate)

| seed | pooled collapsed | vs baseline (142/242=58.7%) | McNemar p | NN recall | vs baseline (11/79=13.9%) |
|---|---:|---:|---:|---:|---:|
| 0 | 128/242 = 52.9% | −14 events | 0.1557 | 46/79 = 58.2% | +35 events |
| 1 | 122/242 = 50.4% | −20 events | 0.0446 | 45/79 = 57.0% | +34 events |
| 2 | 126/242 = 52.1% | −16 events | 0.1011 | 45/79 = 57.0% | +34 events |
| **median** | **52.1%** | **−16 events** | | **57.0%** | **+34 events** |

Every other class regresses hard, in every seed, with **zero** `new-only`
wins (the new model never gets right anything the baseline got wrong on
these classes) — A3 87.5%→~54% (McNemar p≈0.03–0.06), A5 75.0%→~30%
(p≈0.03–0.06), CSE 83.2%→~56% (p<0.0001), RI 67.9%→~29% (p=0.001), all three
seeds. Macro-recall drops from 65.5% to 42.9–46.7%.

### benchmark/ dev (regression guard)

| seed | pooled collapsed | vs baseline (53/70=75.7%) | McNemar p |
|---|---:|---:|---:|
| 0 | 45/70 = 64.3% | −8 events | 0.0768 |
| 1 | 44/70 = 62.9% | −9 events | 0.0490 |
| 2 | 47/70 = 67.1% | −6 events | 0.2101 |
| **median** | **64.3%** | **−8 events** | |

`benchmark/` has zero NN truth, so this whole regression is spurious NN
false positives crowding out A3/A5/CSE/RI predictions.

### ptbp1_arena pair 1 (confirmatory only — informative, not gating)

| seed | pooled collapsed | vs baseline (1175/2422=48.5%) | McNemar p | NN recall | vs baseline (135/800=16.9%) |
|---|---:|---:|---:|---:|---:|
| 0 | 1230/2422 = 50.8% | +55 events | 0.0900 | 652/800 = 81.5% | +64.6pp |
| 1 | 1188/2422 = 49.1% | +13 events | 0.7171 | 675/800 = 84.4% | +67.5pp |
| 2 | 1175/2422 = 48.5% | 0 events | 1.0000 | 641/800 = 80.1% | +63.2pp |

Pooled accuracy is roughly flat here (none of the three differences are
significant) even though NN recall quadruples-plus, because pair 1 is 33%
NN by construction — large enough that the NN gain and the non-NN losses
happen to net out close to zero on *this* set. Per the pre-registration,
this alone is explicitly not sufficient and is not part of the gate; it is
recorded because it clarifies the shape of the effect (see below), not
because it passes anything.

### Gate (applied mechanically, per the fixed criteria above)

1. Median-over-3-seeds NN recall on `benchmark_new` dev beats 13.9% baseline:
   **57.0% — PASS.**
2. ≥2/3 seeds individually beat the 13.9% baseline: **3/3 — PASS.**
3. No more-than-seed-noise regression on `benchmark_new` dev pooled collapsed
   accuracy or on `benchmark/` dev: **FAIL, badly and consistently.** The
   `benchmark_new` dev drop (14–20 events at n=242) is 1.4–2× the branch's
   own measured seed-noise ceiling at that n (10 events,
   `plans/signal-native-v2-resume-9.md`); the `benchmark/` dev drop (6–9
   events at n=70) is 1.2–1.8× the measured ceiling there (5 events). Every
   one of 6 seed×benchmark measurements regresses; none is within noise.

**Overall: FAIL.** Condition 3 is not marginal — it fails by a wide margin,
in the same direction, in all three seeds, on both gating benchmarks.

### What the shape of the result suggests (an unverified hypothesis, not a claim)

The regression is not a modest reallocation like Track D's — it is a
uniform collapse of A3/A5/CSE/RI recall with **zero** compensating
`new-only` wins on any of those classes, paired with a large NN recall gain
that is *largest exactly where the benchmark is real BAM data* (`benchmark_new`
dev +43pp, `ptbp1_arena` +64–68pp) and does not occur at all on sim-internal
held-out folds (in-sim macro-F1 stayed at 0.905–0.911, unchanged from
`e4_union2`'s training regime). The plausible mechanism: 474 real events out
of 12,954 is a small fraction (3.7%), but they are the *only* real BAM
records in the entire pool (every other row, positive or NN, is simulated)
and are labelled NN uniformly. If the transformer can pick up on some
real-vs-simulated BAM signature that is easier to learn than the intended
splicing-vs-no-splicing distinction, the model would learn "looks like real
data" -> NN, which explains every observation here: NN recall jumps hardest
on real curated data, non-NN real events lose recall in exactly the
direction of NN, and nothing moves on the in-sim held-out folds because
those folds contain no real-vs-sim contrast to exploit. This is **not
verified** by this attack — confirming it would need a diagnostic separate
from Stage 14's scope (e.g. scoring real *positive* events, not just real
NN, to see whether real-ness alone predicts the NN prediction independent of
truth) — and is recorded as a hypothesis for whoever picks this up next, not
as a finding.

### Decision (Step 8, per the fixed gate above)

**FAIL.** Per the plan's fail branch: this is not folded into any ensemble,
`e4_union2.json` is untouched, and this is written up as a fourth negative
result on the NN-recall line, alongside Track A's 0.804 annotation cap
(`plans/signal-native-v2-real-supervision-results.md`), Track D's
reallocation-not-improvement, and B3's seed-variance finding
(`plans/signal-native-v2-resume-9.md`). Unlike those three, which found the
NN gap resistant to attack, this one *does* move NN recall substantially —
just at a cost this branch's own gate correctly rejects. `benchmark_new`'s
test half stays reserved, untouched by this attack, per the standing
constraint above.

**For whoever picks this up next**: the "unweighted, as-is" mixing decision
was fixed in Step 0 specifically because it is the cleanest to attribute —
and it is now attributable: the failure is not from too much real data
(474/12,954 is small) but plausibly from *homogeneous* real data (all real
rows share one label). A follow-up attack with real events spanning multiple
labels (would need real positives, not just real NN, undermining the
"targeted NN fix, not a general real-data upgrade" framing this attack was
scoped under) or a domain-adversarial term explicitly penalizing a
real-vs-sim shortcut are two directions worth considering; neither is
attempted here.
