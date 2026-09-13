# PRPF4 arena — pair 3 (Stage 20.1, real fine-tuning cohort)

**Not a curated benchmark and not itself scored as an arena.** Built
2026-09-11/13 by `benchmark_build/build_arena_pair.sh` (the exact pipeline
`ptbp1_arena`/`aqr_arena` used: `sim/fetch_align_benchmark_new.sh` ->
`benchmark_build/run_arena.sh` -> `make_nn_candidates.py --n-total 800 --seed 0`),
no human inspection, no model prediction in any file here. Its role is to
supply a further **real training cohort** for Stage 20's sim-pretrained ->
real fine-tune arms (`plans/signal-native-v2-stage18-22-accuracy-plan.md`).

| condition | ENCODE | runs (single-run replicates, PE100) |
|---|---|---|
| PRPF4 shRNA KD, HepG2 | `ENCSR466NRW` | rep1 `SRR14845398`, rep2 `SRR14845397` |
| non-target shRNA control, HepG2 | `ENCSR093ERJ` (= `ENCSR466NRW`'s sole `possible_controls`) | rep1 `SRR14831566`, rep2 `SRR14831565` |

Accessions verified on the ENCODE portal + ENA before fetching
(`plans/signal-native-v2-stage18-domain-probe.md`). PRPF4 is a tri-snRNP (U4/U6.U5) spliceosomal
factor, mechanistically distinct from SRSF3 (`benchmark_new`), PTBP1 (pair 1)
and AQR (pair 2). STAR wrote `@RG`/`RG:Z:<srr>` for the first time on this
pair (Stage 19 input).

Consensus positives: **664** (`consensus.tsv`, `candidates/{A3,A5,ES,IR}.csv`);
NN half: **800** (`candidates/nn_candidates.tsv`, the same amended criterion
as pair 1, width-balanced against the positives -- `runs/stage20/prpf4_work.log`).

After gene-disjointness against `ptbp1_arena`, `benchmark/` and
`benchmark_new` (`sim/build_real_{pos,nn}_extract_csv.py`, 50 kb clustering;
positives also unioned with `sim/experiment/exclude_genes_stage16.tsv`):
**285 positives** (`sim/experiment/real_pos_prpf4`, kind `real_pos_prpf4`) and
**447 NN** (`sim/experiment/real_nn_prpf4`, kind `real_nn_prpf4`), extracted with
`--min-anchor 5 --max-nh 10`; `dsignal.verify` passes on both
(`runs/stage20/extract34.log`). BAMs and FASTQs stay on pasteur (`prpf4_work/`).
