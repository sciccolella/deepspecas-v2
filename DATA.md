# Data that is not in git

Everything below is referenced by `plans/`, `runs/*.log` or the code but is **not
committed** (see `HISTORY.md` for what was dropped from history). Sizes were
measured 2026-10-05 on the machine that did the work.

"old repo" = `~/code/deepSpecas` (branch `signal-native-v2`, kept as the archive; it
still has all of these working-tree files). "pasteur" = the shared remote
`ssh pasteur`, `/data/conda/ciccolella/deepSpecas-v2` (paths per `CLAUDE.md`,
**not re-checked** when this file was written).

## Needed to run the released model and the verification gates

| what | size | needed for | where it lives now |
|---|---|---|---|
| `benchmark/c1.bam`, `c2.bam` (+`.bai`) | 35 MB | `dsignal.verify`, `score_benchmark` on `benchmark/` | old repo; `scripts/fetch_data.sh` |
| `benchmark_new/c1.bam`, `c2.bam` (+`.bai`) | 145 MB | scoring `benchmark_new` dev | same |
| `benchmark_new_uncurated/c1.bam`, `c2.bam` (+`.bai`) | 165 MB | rebuilding the uncurated candidates | same |
| `models/e4_union2/**/fold*.pt` | 162 MB | released model | **in git** (manifest has sha256) |

`scripts/fetch_data.sh` downloads these from `$DEEPSPECAS_DATA_URL`
(**placeholder until the BAMs are uploaded** to a release/Zenodo record) or copies
them from a checkout with `--from DIR`, and verifies every file against
`scripts/data.sha256`. Fetched files are ignored by `.gitignore`
(`*.bam`, `*.bai`, `*.npz`, `*.pth`) so they cannot be committed by accident.
`scripts/fetch_data.sh --verify` checks what is present.

## Needed only to reproduce training or the arena experiments

| what | size | where it lives now |
|---|---|---|
| `ptbp1_arena/c1.bam`, `c2.bam` (+`.bai`) — Stage 12 arena | 841 MB | old repo; `scripts/fetch_data.sh --set arena` (checksummed). Other arenas (`aqr_`, `prpf4_`, `u2af2_arena`) ship their CSVs only; no BAMs for them were found in the old repo checkout (presumably only on pasteur; not checked) |
| `sim/experiment/<subdir>/` — npz pools (`npz_geomwin`, `npz_union2`, `npz_real3`, ...), simulation `batches*`, `real_*` pools | 534 MB, 40 subdirs | old repo working tree; pasteur is the origin. This includes each batch's own `design.tsv`/`labels.tsv` (they sit under the dropped subdirectories; needed for the gene-exclusion chain). Top-level `sim/experiment/labels_*.tsv` **are** in git and define the training sets, so a pool can be rebuilt from them with `sim/` + pasteur |
| `runs/*/` fold checkpoints other than `e4_union2` (117 run dirs; `.pt` ≈ 3 GB across `runs/`, e.g. `ft_ref` 189 MB, `ens_u2_wide` 60 MB, `ssl_ft_*` 28 MB each) | ~3.5 GB `runs/` total | old repo working tree. Each is reproducible from its committed `.json`/`.log` (seed 0, ~7 min on an RTX 4080); metrics are committed |
| `runs/v1_crossover/` — v1 trained on v2's simulated data (`rs50.pth`, rendered images) | 379 MB | old repo only; v1's *results* are the committed `runs/v1_*.tsv`/`.log` |
| `data/trained_nets/<mode>/trained.pth` — the six v1 pretrained nets | 541 MB | **v1 repository** (`master`); needed only to re-run v1 |
| `/datasets/deepSpecas-v2/{envs,ref,fastq,star_index,sim,work}` | `ref` 115 MB; rest 0 B locally | local `ref/` holds the GRCh38.99 GTF used by `sim/gtf_*`; the STAR index, FASTQ and polyester work dirs live on pasteur |
| `sim/events/` — SUPPA `.ioe` event catalogue | ~40 MB | regenerated on pasteur with SUPPA2 `generateEvents` from the GTF (`.gitignore`) |

## What is in git instead

All `benchmark*/` CSVs, `split.tsv`, `curation.tsv`, `events.bed`; every top-level
`sim/experiment/*.tsv`; per-event result TSVs and logs in `runs/`;
`runs/ensembles/*.json`; `benchmark/igv_images/`. A figure quoted in `plans/`
should be traceable to one of those without any file in the tables above, with the
exception of re-scoring, which needs the BAMs.
