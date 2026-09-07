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

(To be appended after Step 8, per the fixed gate above. This section is
empty at pre-registration time.)
