# U2AF2 arena — pair 4 (Stage 20.1, real fine-tuning cohort)

**Not a curated benchmark and not itself scored as an arena.** Built
2026-09-11/13 by `benchmark_build/build_arena_pair.sh` (the exact pipeline
`ptbp1_arena`/`aqr_arena` used: `sim/fetch_align_benchmark_new.sh` ->
`benchmark_build/run_arena.sh` -> `make_nn_candidates.py --n-total 800 --seed 0`),
no human inspection, no model prediction in any file here. Its role is to
supply a further **real training cohort** for Stage 20's sim-pretrained ->
real fine-tune arms (`plans/signal-native-v2-stage18-22-accuracy-plan.md`).

| condition | ENCODE | runs (single-run replicates, PE100) |
|---|---|---|
| U2AF2 shRNA KD, K562 | `ENCSR904CJQ` | rep1 `SRR3469570`, rep2 `SRR3469571` |
| non-target shRNA control, K562 | `ENCSR344XID` (= `ENCSR904CJQ`'s sole `possible_controls`) | rep1 `SRR3469464`, rep2 `SRR3469465` |

Accessions verified on the ENCODE portal + ENA before fetching
(`plans/signal-native-v2-stage18-domain-probe.md`). U2AF2 is a core 3'-splice-site recognition
factor, mechanistically distinct from SRSF3 (`benchmark_new`), PTBP1 (pair 1)
and AQR (pair 2). STAR wrote `@RG`/`RG:Z:<srr>` for the first time on this
pair (Stage 19 input).

Consensus positives: **2201** (`consensus.tsv`, `candidates/{A3,A5,ES,IR}.csv`);
NN half: **799** (`candidates/nn_candidates.tsv`, the same amended criterion
as pair 1, width-balanced against the positives -- `runs/stage20/u2af2_work.log`).

After gene-disjointness against `ptbp1_arena`, `benchmark/` and
`benchmark_new` (`sim/build_real_{pos,nn}_extract_csv.py`, 50 kb clustering;
positives also unioned with `sim/experiment/exclude_genes_stage16.tsv`):
**1109 positives** (`sim/experiment/real_pos_u2af2`, kind `real_pos_u2af2`) and
**480 NN** (`sim/experiment/real_nn_u2af2`, kind `real_nn_u2af2`), extracted with
`--min-anchor 5 --max-nh 10`; `dsignal.verify` passes on both
(`runs/stage20/extract34.log`). BAMs and FASTQs stay on pasteur (`u2af2_work/`).
