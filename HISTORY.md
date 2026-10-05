This repository was split out of `deepSpecas` (`~/code/deepSpecas`), branch
`signal-native-v2`, at commit `afb6225b5` (its parent plan commit is
`7e902c803`, the final `benchmark_new` test spend). **Hashes cited in
`plans/`, `runs/*.log` and other docs are OLD hashes from that repository**;
resolve them to this repository's hashes through `HISTORY_MAP.tsv` (old → new).
Commits pruned by the filter (they touched only dropped paths) map to
`0000000000000000000000000000000000000000`.

## How the history was produced

`git filter-repo --invert-paths` on a fresh `--no-local` clone, dropping:

- `sim/experiment/*/*` (every subdirectory: npz pools, batches; top-level
  label/design TSVs are kept)
- `*.bam`, `*.bai`, `*.pt`, `*.pth`, `*.npz` everywhere
- v1 code and artifacts: `deepSpecas.py`, `dsEnsemble.py`, `plot2s.py`,
  `plot.py`, `DSutils.py`, `mc_mplp.py`, `jitimg.py`, `runDS.smk`,
  `polyDS.smk`, `esmPredict.py`, `run_esm.sh`, `run_ensemble_parallel.sh`,
  `report.md`, `hm.png`, `data/`, `plans/render-throughput.md`

Kept (deliberately): `benchmark/igv_images/` (5.8 MB of IGV screenshots that
document the curated truth set).

| | before | after |
|---|---|---|
| commits on the branch | 244 | 226 |
| `.git` object store | 1.4 GB | 18 MB (pack 17.5 MiB after `gc --aggressive`) |
| largest blob | 107 MB | < 5 MB |

Verified: the final tree equals the source tree at `afb6225b5` restricted to the
kept paths (`diff -r` empty). Default branch is `master`, as in the source repo.
The large data that was dropped is described in `DATA.md` (added later).
