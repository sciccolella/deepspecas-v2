# v2 → its own repository, and release consolidation

Agreed 2026-10-05. The v2 research programme has ended (resume-11: every gated
intervention failed its gate, the final `benchmark_new` test spend PASSED, and
no held-out curated half remains). This plan moves v2 out of `deepSpecas`
branch `signal-native-v2` into a **new, standalone repository with its history
preserved**, then turns it into something releasable.

## Decisions already taken (by the user, 2026-10-05)

| question | decision |
|---|---|
| history | **Keep the commits.** Rewrite this repo's `signal-native-v2` with `git filter-repo` so only v2 paths stay. Messages, authors and dates survive; data blobs leave history. |
| binaries | **Weights in, BAMs out.** Commit the 30 `e4_union2` fold checkpoints (~162 MB, ~5.4 MB each). BAMs, `.npz` pools and other large data are fetched separately. |
| v1 coupling | **Vendor minimal.** Copy only `plot2s.get_coords_from_bam` into a small frozen module so `dsignal.verify` still runs. Drop the v1 crossover tooling from the final tree; v1's *results* stay as committed TSVs. |
| location | `~/code/deepSpecas-v2`. This checkout stays untouched as the archive. |
| remote | The user creates the empty remote and supplies the origin URL. **Pushing is still the user's call**: add the remote, and push only when the user explicitly says so in that session. |

Proposed, not yet confirmed (confirm at the start of the implementation session):
the new repo's default branch is called `main`.

## What leaves history (filter rules)

Measured on `7e902c803` before writing this plan:

- Branch diff vs `master`: 186k files. `.git` is 1.4 GB, almost all of it
  `sim/experiment/<subdir>/` (npz pools and batches, ~180k files) and BAMs
  (largest blob 107 MB, `benchmark_new_uncurated/c2.bam`).
- Tracked `runs/` is text only, apart from 3 `.pt`: 310 tsv (34 MB), 378 log (1.2 MB), 150 json.
  `runs/*/` is gitignored, so the 30 `e4_union2` checkpoints are **not in
  history**. They come in as a new commit (step 6).
- Top-level `sim/experiment/*` files (labels/design TSVs) total 11 MB. Every
  non-BAM file under the benchmark/arena dirs totals 8.7 MB.
- No renames on the branch outside `sim/experiment/` and `runs/`, so path filters do not cut file histories short.

**Drop from all history:**
- `sim/experiment/*/` (subdirectories only; keep top-level files in `sim/experiment/`)
- `*.bam`, `*.bai`, `*.pt`, `*.pth`, `*.npz` everywhere
- v1: `deepSpecas.py`, `dsEnsemble.py`, `plot2s.py`, `plot.py`, `DSutils.py`,
  `mc_mplp.py`, `jitimg.py`, `runDS.smk`, `run_esm.sh`,
  `run_ensemble_parallel.sh`, `report.md`, `hm.png`, `data/`,
  `plans/render-throughput.md`
- `benchmark/igv_images/` (v1 screenshots). Check its size first; keep it if small and the user wants it.

**Keep:** `dsignal/`, `sim/` (minus the above), `plans/` (the decision
record), `benchmark/`, `benchmark_new/`, `benchmark_new_uncurated/`,
`benchmark_build/`, `*_arena/`, `tettamanti/` (non-binary files), text in `runs/`,
`CLAUDE.md`, `README.md`, `crash_pasteur.md`, `.gitignore`.

Commits left empty by the filter are pruned (default). `master`'s pre-branch v1
commits mostly disappear; those that touched `benchmark/` remain. That's fine.

## Steps

| # | Step | Tier |
|---|---|---|
| 0 | **In this repo, before cloning:** commit the 17 untracked `runs/*.log` so they reach history. Add `sim/experiment/npz_union/` to `.gitignore` (it gets filtered out anyway). Commit this plan and its `plans/README.md` row. | Haiku |
| 1 | **Get `git filter-repo`** without touching `DSnet`: `pipx install git-filter-repo`, or the single-file script into `~/.local/bin`. **Never `pip install` into DSnet.** | Haiku |
| 2 | **Clone and analyse.** `git clone --no-local --single-branch -b signal-native-v2 ~/code/deepSpecas ~/code/deepSpecas-v2`, then `git filter-repo --analyze`. Check the path report against the rules above. List any path >1 MB not covered by the rules and decide on each before filtering. | Opus |
| 3 | **Filter.** One `git filter-repo` run using `--paths-from-file` (plus `--invert-paths` for the drop list), with the rules as written above. Rename the branch to `main` (if confirmed) and remove the `origin` remote that points back at the old repo. **Gates:**<br>(a) the final tree equals the source tree at `7e902c803` restricted to the kept paths (`git diff --stat` between them is empty);<br>(b) no blob in history is >5 MB (`git rev-list --objects --all \| git cat-file --batch-check`);<br>(c) `.git` size after `gc --aggressive` is recorded in the commit message;<br>(d) commit count before→after is recorded.<br>Commit messages get their hashes rewritten automatically. Plan docs and CLAUDE.md that *cite* hashes do not: handle that in step 4. | Opus |
| 4 | **Hash map.** Commit `.git/filter-repo/commit-map` as `HISTORY_MAP.tsv` (old→new; pruned commits map to `0000…`). The first line of `HISTORY.md` points to the source repo, branch and `7e902c803`, and says that hashes in plan docs are *old* hashes, resolved through the map. | Haiku |
| 5 | **v1 decoupling commit.**<br>• Vendor `plot2s.get_coords_from_bam`, with its helpers, as `dsignal/_v1_coords.py`: frozen, with a header saying where it came from (path + old commit). Point `dsignal/verify.py` at it.<br>• `grep` for other v1 imports. `extract.py` only mentions `plot2s` in docstrings, so leave those.<br>• Delete `sim/render*.sh`, `sim/build_v1_trainset.py`, `sim/build_render_csv.py`. Keep `dsignal/combine_v1v2.py` and `dsignal/v1_reference.py` only if they read committed TSVs and nothing else; otherwise delete them.<br>**Gate:** `python -m dsignal.verify` and `python -m dsignal.verify_features` pass, with the BAMs copied in as ignored local files. | Sonnet |
| 6 | **Package the model.**<br>• Copy the 30 checkpoints from `runs/{scale_union2_n12480,ens_union2_{allfeat,twohead,depthscale,augsub,nn15}}/fold*.pt` to `models/e4_union2/<member>/fold*.pt`. Write `models/e4_union2/manifest.json`, pointing at those paths, with sha256 for each file, alongside `runs/ensembles/e4_union2.json`.<br>• Make the settings frozen for the test spend (TTA condition-swap, `--target-prior nn-weight --nn-weight 0.1`) the defaults of `dsignal.predict` when no model is given, so the default call runs the released ensemble at its scored operating point.<br>• Check CPU fallback.<br>**Gate:** `score_benchmark` on `benchmark_new` **dev** with the packaged model gives `pred`/`p_*` identical to `runs/bnew_dev_e4_union2.tsv`. **Never re-score test.** | Sonnet |
| 7 | **Data outside git.** `DATA.md` covers:<br>• every artifact the plan docs reference that is not in the repo (BAMs, `npz_*` pools, arena BAMs, `runs/*/` checkpoints other than e4), with its size and where it lives now (this machine's paths, the old repo, pasteur);<br>• `scripts/fetch_data.sh`, which downloads the benchmark BAMs from a release/Zenodo URL (**placeholder until the user uploads**) and checks them against committed sha256 sums;<br>• `.gitignore` updated so fetched files can't be committed by accident. | Sonnet |
| 8 | **Environment recipes.**<br>• `envs/inference.yml`: minimal (torch, pysam, numpy, pandas, scikit-learn). Build it in a scratch prefix and run step 6's gate in it.<br>• `envs/sim.yml`: from `DSsim`/`DSpoly2v2` on pasteur via `conda env export --from-history`, read-only on pasteur. | Sonnet |
| 9 | **Results summary**, `RESULTS.md`, holding every number that can be claimed and its caveats:<br>• `benchmark/` test: E3 85.7%, single model 82.9%, v1 ensemble 63/70<br>• `benchmark_new` test: 143 vs 110/243, p<0.0001<br>• per-class recall, NN 3/80, and 85.9% vs 67.5% excluding NN<br>• wall clock: 0.106 vs ~17 s/event<br>• caveats: genes shared between the sim pool and the test set; `DPSI_BETA`/`PSI_LEVEL_BETA` fit on all 140 events; `e4_union2` selected on `benchmark/` dev; no held-out half remains; retracted figures stay listed as retracted.<br>Every number must cite its source log/TSV in the repo. | Opus |
| 10 | **README.md for v2** (replaces v1's): what it is; install (`envs/inference.yml`); input format `region,bam1,bam2` (same as v1); default ensemble command; output columns; `--min-confidence`; the known NN weakness, stated plainly; link to `RESULTS.md`, `DATA.md` and `plans/README.md`. Credit v1/deepSpecas and link the original repo. | Sonnet |
| 11 | **Reproducibility driver**, `workflow/README.md` plus a documentation-grade `Snakefile` or `run_all.sh`: design → polyester/STAR (pasteur, with `limitram`/thread caps/`nice` per `crash_pasteur.md`) → extract → merge → train the 6 members → score dev. Like `runDS.smk`, it's written to document the pipeline and doesn't have to run end to end. | Sonnet |
| 12 | **CLAUDE.md for the new repo.** Rewrite for v2 only: no branch talk; v1 is mentioned as the predecessor in another repo. Keep the pasteur rules, verification gates, v2 conventions, git rules (never push without the user; never create or switch branches) and the planning workflow. Entry point: `RESULTS.md` + `plans/README.md`. Seed the new project memory dir (`~/.claude/projects/-home-ciccolella-code-deepSpecas-v2/memory/`) with the durable memories from the old one (pasteur/limitram, seed-replicate/McNemar rules, no manual curation, incremental verify), updated with the new paths. | Haiku |
| 13 | **Remote.** `git remote add origin <URL from user>`. Run `git status`/`git log --oneline \| head` and report the size. **Push only on the user's explicit go**, in that session. | Haiku |
| 14 | **Back in this repo:** add a short note at the top of `CLAUDE.md` and `plans/README.md` saying v2 development continues in the new repo (URL), with `signal-native-v2` @ (final hash) kept as the archive. Commit. No push. | Haiku |

Build → verify → commit each step before the next. Every step from 4 on is its
own commit in the new repo.

## Out of scope here (optional later tracks, discussed 2026-10-05)

- **B. Manuscript**: figures plus methods text.
- **C. Exploratory NN work**: only with a new label source that needs no manual
  curation, e.g. orthogonal long-read truth. Starts with a scoping step:
  does such a dataset exist for samples we have short reads for?
