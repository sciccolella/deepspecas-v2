# Reproduction driver

[`run_all.sh`](run_all.sh) is a **map, not a button**. It records, in order, the
commands behind the released `e4_union2` ensemble:

```
design -> simulate (pasteur) -> extract -> merge -> train 6 members (local GPU) -> score dev
```

Like v1's `runDS.smk`, it documents the pipeline and its gates and has not been run
end to end as a single script: the real work ran in stages over weeks on two
machines. What *was* checked when it was written: it parses (`bash -n`), every module
and file it names exists, and the `dsignal.train` / `dsignal.score_benchmark` flags it
uses are accepted. The simulation half was **not** re-run.

## Stages

| stage | where | what | gate before accepting |
|---|---|---|---|
| `design` + `simulate` | pasteur (no GPU) | `sim/run_batch.sh <tag> <seed> <prior design.tsv ...>`: SUPPA event design → polyester reads → real STAR realignment → `dsignal.extract` → `.npz` | `sim.psi_recovery_check`, `sim.width_leak_check`, and `dsignal.audit_noise compare` (real vs simulated through the identical code path) |
| `merge` | local | `sim/merge_batches.py`: 13 batches → `npz_union2` (12,480 events, 12,480 genes); nested subsamples; the 15%-NN labels for the `nn15` member | 12,480 records, 12,480 distinct genes |
| `train` | local, GPU (`DSnet`) | six members, 5-fold gene-grouped CV each, `--refit --inner-val-frac 0.30` | the no-gene-leak assertion in `dsignal.train` |
| `score` | local | `dsignal.score_benchmark --ensemble ... --half dev` | reproduces `runs/bnew_dev_e4_union2.tsv` |

## Rules that are not optional

- **Dev only.** Both curated test halves are spent (`RESULTS.md` §5.1); do not score
  them again for selection.
- **One gene per event, and gene-disjointness across batches.** Each batch passes
  `--exclude-genes` for every earlier design, otherwise different seeds re-pick the
  same genes and leak in a way per-batch `GroupKFold` cannot see.
- **Pasteur is shared.** `limitram`, capped thread pools, `nice -n 19`, check
  `/proc/loadavg` before and a minute after launching (details and the incident in
  [`../crash_pasteur.md`](../crash_pasteur.md)).
- **Seed-replicate before believing an A/B**: seed spread alone is ~5 events at n = 70.

## What you cannot rebuild from git alone

Per-batch `design.tsv`/`labels.tsv` (needed for the exclusion chain), the `.npz`
pools, the STAR index and the reference FASTA are not in this repository; see
[`../DATA.md`](../DATA.md). The top-level `sim/experiment/labels_*.tsv` are in git and
fix which records each training run used.

## Environments

`envs/sim_polyester.yml` (R, polyester), `envs/sim_align.yml` (STAR, samtools, SUPPA)
for pasteur; training uses a CUDA torch env (the lab one is `DSnet`: python 3.11,
torch 2.5.1, numpy 2.4, scikit-learn 1.9, lightgbm 4.7).
