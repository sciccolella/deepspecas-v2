# Stage 14 — the NN attack

## Context

Stage 13 (`plans/signal-native-v2-arena-b3.md`) pre-registered and ran a
head-to-head scoring of v1's six-net ensemble and v2's `e4_union2` ensemble
on `ptbp1_arena/`, a 2,422-event automated, model-blind benchmark with an NN
(no-splicing-change) class — the one comparison neither `benchmark/` (no NN
at all) nor `benchmark_new` (n=242 dev, NN present but small) can give. v2
won pooled (48.5% vs 36.4% collapsed, McNemar p<0.0001) and beat v1 on every
per-class recall except RI. **NN recall was v2's largest per-class deficit**
(16.9% at the deployed `nn_weight=0.1` operating point) — three prior
attacks on NN (Stage 11, Track A, Track D) had all failed to move it, so
this result both confirms NN as the real gap and licenses the pre-registered
next move: `plans/signal-native-v2-arena-b3.md`'s Step 5 decision rule
commits to "the NN attack" when this exact condition holds.

The attack: build a **second** automated arena (**pair 2**) from a
*different* ENCODE shRNA knockdown experiment than PTBP1 (pair 1) or SRSF3
(`benchmark_new`), extract its NN candidates as **real** hard negatives
(not simulated), add them to the existing simulation training pool, retrain,
and check whether NN recall moves — validated on curated `benchmark_new`
dev (the only set with both curated labels and an NN class), with `ptbp1_arena`
(pair 1) held out as a confirmatory-only eval, never a tuning signal.

This is narrower than the "fine-tuning on real data" idea CLAUDE.md already
discourages (`plans/signal-native-v2-resume-2.md`): that caveat is about
*blanket* domain-adaptation fine-tuning after three prior simulation-realism
fixes failed to move transfer. This attack injects only real, extracted NN
examples — no synthetic-realism modeling — is evidence-triggered by a
measured result (not a generic bet), and is checked against curated data
before being trusted.

**Decisions fixed by the user, going into this plan** (see Step 0 for how
they're recorded):
- **Pair 2 = AQR knockdown**, `ENCSR431SXF` vs control `ENCSR225PRV`
  (HepG2), over the PRPF4 backup — chosen for maximum mechanistic distance
  from PTBP1 (hnRNP) and SRSF3 (SR protein): AQR is a core intron-binding
  spliceosomal factor.
- **Real:sim mixing ratio = as-is, unweighted.** Add pair-2's real NN rows
  on top of the existing ~12,480-event sim pool at whatever count they
  naturally are — no upsampling, no `--class-weight-boost`. Cleanest,
  easiest to attribute if the number moves.
- **Go/no-go gate = the draft as written**: median-over-3-seeds NN recall
  beats baseline on `benchmark_new` dev, AND ≥2/3 seeds individually beat
  baseline, AND no more-than-seed-noise regression on dev pooled accuracy
  or on `benchmark/` dev.

## Step 0 — pre-register (strong model)

Write `plans/signal-native-v2-arena-nn-attack.md`, committed **before**
Step 2 pulls any FASTQ. Contents, fixed now:

- **Entrants and pool composition**: pair-2 accessions (below); NN-only,
  unweighted addition to the sim pool (no pair-2 positives — their purity
  is as unmeasured as pair 1's, and this attack's premise is a targeted NN
  fix, not a general real-data upgrade); exact sim-labels file to merge
  against — **cross-check `runs/ensembles/e4_union2.json`'s per-member
  `"labels"` field first** (mixed `labels_union2.n12480.tsv` /
  `.n12480.nn15.tsv` across members in the existing ensemble) and record
  which one this attack's baseline/merge uses, so the comparison isn't
  silently against the wrong sim pool size.
- **Gene-disjointness rule**: pair-2's training genes must be disjoint from
  `ptbp1_arena/` (both `consensus.tsv` and `nn_candidates.tsv`'s `gene_id`
  columns — pair 1 stays held-out eval), `benchmark_new/`, and `benchmark/`
  (both still-live curated sets). `benchmark_new`/`benchmark` carry regions
  only, so reuse `dsignal/benchmark_split.py`'s single-linkage
  clustering-at-50kb as the same "same gene" proxy it already uses, rather
  than inventing a new threshold. Output artifact:
  `sim/experiment/exclude_genes_nn_attack.tsv` (one `gene_id` column,
  pooled from all three sources), built in Step 3 and asserted empty-overlap
  after filtering, the same visible-assertion style `sim/select_events.py`
  already uses.
- **Evaluation order and go/no-go gate** (both fixed above): `benchmark_new`
  dev is the primary gate; `benchmark/` dev is the regression guard;
  `ptbp1_arena` pair 1 is confirmatory only, scored once per seed, never
  used to pick a seed or checkpoint — write explicitly that any NN-recall
  movement on pair 1 alone is *not* sufficient (its NN purity is as
  unmeasured as pair 1's own Trap 2 in the B3 doc).
- **3 seeds** (train seeds 0/1/2), matching the branch's seed-replicate
  convention.
- Record baseline numbers to compare against (read off, not re-derived
  later): `e4_union2`'s existing `benchmark_new` dev NN recall and pooled
  collapsed accuracy, and `benchmark/` dev's current 53/70.

## Step 1 — freeze pair-2 accessions (strong model)

Verify `ENCSR431SXF` (AQR, HepG2, paired-end 100bp, SRR14843100/SRR14843099)
against the ENCODE portal and NCBI SRA, and confirm `ENCSR225PRV` is its
`possible_controls` entry — same freeze procedure `ptbp1_arena/README.md`'s
condition table documents for pair 1.

Resolve the control's multi-run-per-replicate wrinkle: rep1 =
SRR14840064+SRR14840061, rep2 = SRR14840062+SRR14840063 (`sim/fetch_align_benchmark_new.sh`
expects one SRR per replicate slot). Concatenate each replicate's two runs'
FASTQs before alignment (`cat SRR14840064_1.fastq SRR14840061_1.fastq >
cond2_rep1_1.fastq`, mate 2 similarly) via a short pre-step ahead of the
existing fetch script — do not edit `fetch_align_benchmark_new.sh` itself,
since it's shared with any future pair-3 build that may not have this
wrinkle.

Append the frozen accessions/GEO IDs/run structure to
`plans/signal-native-v2-arena-nn-attack.md`, in `ptbp1_arena/README.md`'s
condition-table format, before any fetch runs.

## Step 2 — build pair 2 on pasteur (cheap model, mechanical)

Reuse pair 1's exact pipeline and pasteur conventions (`limitram`,
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`, `nice -n 19`, `/proc/loadavg` before/after):

1. `sim/fetch_align_benchmark_new.sh` (after the concatenation pre-step) to
   fetch/align into a fresh `aqr_work/`.
2. `benchmark_build/run_arena.sh aqr_work aqr_arena aqr` with
   `C1_REP1/REP2`, `C2_REP1/REP2` env vars set to the AQR/control run IDs.
   No script edits expected — copy pair 1's literal invocation (check the
   B2 commit / pasteur shell history for exact `IOE_DIR`/`tpm_dir` args
   rather than re-deriving argument order).
3. `benchmark_build/make_nn_candidates.py` with pair 1's same thresholds
   (`p > 0.20` + NaN guard, `|dPSI| < 0.05`, TPM ≥ 1.0) to build the NN half.
   Both of pair 1's known gotchas (SUPPA2 `p==1.0` NaN guard, BAM subset
   must be cut over the BED covering positives *and* NN) are already fixed
   in the committed scripts and apply automatically.
4. Write `aqr_arena/README.md` modeled on `ptbp1_arena/README.md`.

Gate before moving on: `dsignal.verify` over `aqr_arena` (expect a clean
pass), and `sim/width_leak_check.py`-style AUC sanity (expect ~0.5, no
class separability from width alone).

## Step 3 — pair-2 NN candidates → extract-CSV + labels.tsv (cheap model to implement against a template; strong model to review the gene-disjointness filter)

New file: `sim/build_real_nn_extract_csv.py`, modeled on
`sim/build_extract_csv.py`, reading `aqr_arena/candidates/nn_candidates.tsv`
(columns confirmed present: `source_class, chr, strand, region_start,
region_end, gene_id, gene_symbol, rmats_pvalue, rmats_incleveldiff,
suppa_event_id, suppa_dpsi, suppa_pvalue`) instead of a sim `design.tsv`.

1. Build/load the exclusion set (`sim/experiment/exclude_genes_nn_attack.tsv`
   from Step 0), drop any row whose `gene_id` is in it, print
   excluded/kept counts, assert zero post-filter overlap.
2. `event_id` = `"<gene_id>;<suffix from suppa_event_id>"` (already close
   to this shape — `suppa_event_id` is `"<gene_id>;A3:..."` etc., a light
   reformat via `sim/gene_split.py`'s convention, not a new ID scheme).
3. `label` = `"NN"` for every row (NN-only, per the fixed pool-composition
   decision); `kind` = `"real_nn_aqr"` for provenance.
4. Extract-CSV rows: re-key filtered `nn_candidates.tsv` rows against
   `aqr_arena/candidates/NN.csv` by `suppa_event_id` to reuse the
   already-padded regions `make_nn_candidates.py` computed, rather than
   recomputing window padding — check this join before assuming it's
   1:1 (`make_nn_candidates.py`'s NN.csv construction order).

Outputs: `sim/experiment/real_nn_aqr/extract.csv`,
`sim/experiment/real_nn_aqr/labels.tsv` — a sibling batch directory to
`sim/experiment/batches_union2/`, shaped for direct use by
`sim/merge_batches.py --batch`.

## Step 4 — extract and verify (pasteur, cheap model, mechanical)

```bash
python -m dsignal.extract --csv sim/experiment/real_nn_aqr/extract.csv \
  --out-dir sim/experiment/real_nn_aqr/npz
python -m dsignal.verify --csv sim/experiment/real_nn_aqr/extract.csv
```

Same thread-pool caps / `nice -n 19` as every pasteur job. Confirm npz count
matches the filtered `labels.tsv` row count; investigate (don't silently
accept) any unexplained drop.

## Step 5 — merge with the sim pool (cheap model, mechanical)

```bash
python sim/merge_batches.py \
  --batch union2=sim/experiment/npz_union2 \
  --batch real_nn_aqr=sim/experiment/real_nn_aqr \
  --out-npz sim/experiment/npz_union2_realnn \
  --out-labels sim/experiment/labels_union2_realnn.tsv
```

Use the exact `union2` labels file recorded in Step 0 (per the
`e4_union2.json` member cross-check) as this batch's labels source. No
`--nn-frac` / upsampling — the fixed decision is as-is, unweighted.

## Step 6 — train, 3 seeds (strong model to launch/monitor; mechanical thereafter)

Same recipe as the current best single-model run:

```bash
for seed in 0 1 2; do
  python -m dsignal.train --npz-dir sim/experiment/npz_union2_realnn \
    --labels sim/experiment/labels_union2_realnn.tsv \
    --augment all --feature-subset no-geometry --refit --inner-val-frac 0.30 \
    --seed $seed --out-dir runs/nn_attack_realnn_seed$seed \
    --json-out runs/nn_attack_realnn_seed$seed.json
done
```

Note in the write-up: `dsignal/train.py` has no fine-tune/warm-start path —
every fold and the `--refit` retrain instantiate a fresh
`JunctionSetTransformer`. This is retraining from scratch on an augmented
pool, not literal fine-tuning of an existing checkpoint; the plan's framing
("narrower than blanket real-data fine-tuning") holds regardless, since what
distinguishes this attack is the *evidence-triggered, NN-only real-data
injection*, not the warm-start mechanics.

## Step 7 — evaluate in the pre-registered order (cheap model to run; strong model to read against the gate)

Per seed, in this fixed order:

1. **`benchmark_new` dev** (primary gate) — `dsignal.score_benchmark` with
   `--half dev --split benchmark_new/split.tsv`, then
   `dsignal.benchmark_split compare` against the pre-attack `e4_union2`
   baseline TSV on `benchmark_new` dev (generate that baseline first if it
   doesn't already exist in `runs/`), `--by-class` for the NN-recall read.
2. **`benchmark/` dev** (regression guard) — same protocol against
   `runs/geomwin_iv30_refit`'s existing 53/70.
3. **`ptbp1_arena` pair 1** (confirmatory only) — `--half both` against
   `runs/arena_v2_e4union2.tsv`, `--by-class`. Scored once per seed, never
   used to pick a seed or checkpoint.

## Step 8 — decision (strong model)

Apply Step 0's fixed gate mechanically. Append a `## Results` section to
`plans/signal-native-v2-arena-nn-attack.md` (never edit the pre-registration
text itself).

- **Pass**: fold the winning pool into a new ensemble member
  (`runs/ensembles/e5_<name>.json` or a direct `e4_union2.json` update),
  re-score the whole ensemble on `benchmark/` dev and `benchmark_new` dev
  before promoting it to any new headline.
- **Fail**: write up as a fourth negative result alongside Track A's 0.804
  cap, Track D's reallocation, and B3's seed-variance finding; consolidate
  per Stage 13's own decision rule. `benchmark_new`'s test half stays
  reserved either way.

## Verification

- Gates at each stage: `dsignal.verify` (pair-2 build, and again after
  extraction), `sim.width_leak_check` on `aqr_arena`, npz-count vs
  filtered-labels-row-count check after extraction.
- Regression guard: `benchmark/` dev must stay at 53/70 ± seed noise.
- Row-count / key-set sanity before any `benchmark_split compare`, matching
  the convention Stage 13 used: paired TSVs must share identical
  `(csv, region)` key sets.
- Build → verify → commit each step; never `git push`; never create/switch
  branches unprompted. pasteur jobs wrapped in `limitram` with capped
  thread pools and `nice -n 19`, checking `/proc/loadavg` before and after.

## Files

**New**: `plans/signal-native-v2-arena-nn-attack.md` (pre-registration +
results), `aqr_arena/` (mirroring `ptbp1_arena/`'s layout),
`sim/build_real_nn_extract_csv.py`, `sim/experiment/exclude_genes_nn_attack.tsv`,
`sim/experiment/real_nn_aqr/{extract.csv,labels.tsv,npz/}`,
`sim/experiment/npz_union2_realnn/`, `sim/experiment/labels_union2_realnn.tsv`,
`runs/nn_attack_realnn_seed{0,1,2}/`, `runs/nn_attack_*.tsv` (eval outputs).

**Possibly new**: `runs/ensembles/e5_<name>.json`, only if Step 8 passes.

**Not modified**: no existing sim/experiment npz or labels files, no
existing ensemble json (until Step 8's pass branch), no model/feature code
in `dsignal/model.py` or `dsignal/features.py` — this attack is a data
change, not an architecture change.
