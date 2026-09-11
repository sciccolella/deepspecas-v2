"""Stage 16 (multi-label real-data attack): turn an arena's real *positive*
candidates (all four non-NN classes) into an `extract.csv`/`labels.tsv`
batch shaped like `sim/build_extract_csv.py`'s output, ready for
`dsignal.extract` and `sim/merge_batches.py --batch`. Sibling of
`sim/build_real_nn_extract_csv.py` (Stage 14's NN-only version); this one
reads `<arena-dir>/consensus.tsv` (the caller-derived positives, columns
`class, chr, strand, region_start, region_end, gene_id, gene_symbol,
rmats_pvalue, rmats_fdr, rmats_incleveldiff, suppa_event_id, suppa_dpsi,
suppa_pvalue`) instead of `nn_candidates.tsv`, and each of
`<arena-dir>/candidates/{A3,A5,ES,IR}.csv` instead of `NN.csv` -- verifying
the row-alignment class by class, since `consensus.tsv` interleaves classes
while each candidates CSV is one class's own subsequence in the same
relative order (`benchmark_build/make_candidate_csvs.py`'s
`df.groupby("class")`, which iterates classes alphabetically and preserves
row order within each group).

`label` is `consensus.tsv`'s own `class` column verbatim -- it is already in
the five-class label space (`CSE`/`RI`/`A3`/`A5`; the CSV *filename* stem
differs, `CSV_NAME` maps class->stem the other direction from
`dsignal.score_benchmark.CSV_LABEL`'s stem->class) -- multi-label, unlike
Stage 14's NN-only batch. `kind` is `"real_pos_aqr"` for provenance, matching
Stage 14's `"real_nn_aqr"` naming pattern.

Gene-disjointness: rebuilt from scratch against these positives' own
regions (not reused verbatim from Stage 14's exclusion set, since positives
occupy different loci within the arena than the NN half and can cluster
differently against curated benchmark regions -- see
`plans/signal-native-v2-stage16-multilabel-real.md`), then unioned with an
existing exclusion set (`--union-with`, Stage 14's
`sim/experiment/exclude_genes_nn_attack.tsv`) so both real batches share one
consistent exclusion set in the merged pool.

Usage::

    python -m sim.build_real_pos_extract_csv \\
        --arena-dir aqr_arena --bam1 aqr_work/align/c1_aqr.bam \\
        --bam2 aqr_work/align/c2_aqr.bam \\
        --other-arena-consensus ptbp1_arena/consensus.tsv \\
        --other-arena-nn ptbp1_arena/candidates/nn_candidates.tsv \\
        --curated-csv benchmark_new/A3.csv benchmark_new/A5.csv benchmark_new/ES.csv \\
            benchmark_new/IR.csv benchmark_new/NN.csv \\
            benchmark/A3.csv benchmark/A5.csv benchmark/ES.csv benchmark/IR.csv \\
        --union-with sim/experiment/exclude_genes_nn_attack.tsv \\
        --out sim/experiment/real_pos_aqr/extract.csv \\
        --labels-out sim/experiment/real_pos_aqr/labels.tsv \\
        --exclude-out sim/experiment/exclude_genes_stage16.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

from benchmark_build.make_candidate_csvs import CSV_NAME, PAD
from dsignal.benchmark_split import cluster
from dsignal.predict import read_prediction_csv
from sim.build_real_nn_extract_csv import _gene_ids, _region_events


def build_exclusion_set(pos_rows, other_consensus, other_nn, curated_csvs, link: int) -> set[str]:
    """Same shape as `sim.build_real_nn_extract_csv.build_exclusion_set`,
    generalized to positives (kept as a separate function -- the docstring
    there is NN-specific and importing it directly would be misleading)."""
    excluded = set()
    for path in (other_consensus, other_nn):
        excluded |= _gene_ids(path)
    n_direct = len(excluded)

    pos_events = [
        (r["gene_id"], r["chr"], int(r["region_start"]), int(r["region_end"]))
        for r in pos_rows
    ]
    curated_events = _region_events(curated_csvs)
    combined = (
        [(gid, gid, chrom, start, end, None) for gid, chrom, start, end in pos_events]
        + curated_events
    )
    clusters = cluster(combined, link)
    n_pos = len(pos_events)
    for cl in clusters:
        has_curated = any(i >= n_pos for i in cl)
        if not has_curated:
            continue
        for i in cl:
            if i < n_pos:
                excluded.add(combined[i][0])
    n_clustered = len(excluded) - n_direct
    print(f"exclusion set: {n_direct} genes from ptbp1_arena (direct), "
          f"+{n_clustered} more from 50kb clustering against curated regions "
          f"= {len(excluded)} total", file=sys.stderr)
    return excluded


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena-dir", required=True, help="e.g. aqr_arena")
    ap.add_argument("--bam1", required=True)
    ap.add_argument("--bam2", required=True)
    ap.add_argument("--other-arena-consensus", default=None, help="e.g. ptbp1_arena/consensus.tsv")
    ap.add_argument("--other-arena-nn", default=None, help="e.g. ptbp1_arena/candidates/nn_candidates.tsv")
    ap.add_argument("--curated-csv", nargs="+", default=None)
    ap.add_argument("--link", type=int, default=50000)
    ap.add_argument("--union-with", default=None,
                     help="an existing gene-exclusion TSV to union into the result "
                     "(e.g. Stage 14's exclude_genes_nn_attack.tsv)")
    ap.add_argument("--no-exclude", action="store_true",
                     help="Stage 18.0: skip gene exclusion and keep every row -- for a "
                     "cohort extracted as a *domain sample*, never a training pool "
                     "(see sim/build_real_nn_extract_csv.py)")
    ap.add_argument("--kind", default="real_pos_aqr",
                     help="labels.tsv `kind` value (default %(default)s); must start with 'real_'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-out", required=True)
    ap.add_argument("--exclude-out", default=None)
    args = ap.parse_args(argv)
    if not args.kind.startswith("real_"):
        ap.error("--kind must start with 'real_'")
    if not args.no_exclude and not (args.other_arena_consensus and args.other_arena_nn
                                    and args.curated_csv and args.exclude_out):
        ap.error("--other-arena-consensus, --other-arena-nn, --curated-csv and --exclude-out "
                 "are required unless --no-exclude")

    consensus_tsv = f"{args.arena_dir}/consensus.tsv"
    with open(consensus_tsv, newline="") as f:
        all_rows = list(csv.DictReader(f, delimiter="\t"))

    by_class = defaultdict(list)
    for r in all_rows:
        by_class[r["class"]].append(r)

    # verify row-alignment against each per-class candidates CSV before trusting it
    aligned_rows = []
    for cls, rows in sorted(by_class.items()):
        cand_csv = f"{args.arena_dir}/candidates/{CSV_NAME[cls]}.csv"
        cand_rows = list(read_prediction_csv(cand_csv))
        assert len(rows) == len(cand_rows), (
            f"{consensus_tsv} class={cls} has {len(rows)} rows but {cand_csv} has "
            f"{len(cand_rows)} -- not 1:1, cannot pair by row index")
        pad = PAD[cls]
        for i, (tsv_row, (region, bam1, bam2)) in enumerate(zip(rows, cand_rows)):
            expect_start = max(1, int(tsv_row["region_start"]) - pad)
            expect_end = int(tsv_row["region_end"]) + pad
            expect_region = f"{tsv_row['chr']}:{expect_start}-{expect_end}"
            assert region == expect_region, (
                f"{cls} row {i}: {consensus_tsv} implies region {expect_region!r} but "
                f"{cand_csv} has {region!r} -- the two files are not row-aligned")
            aligned_rows.append((tsv_row, region))
        print(f"verified {len(rows)} rows 1:1 between {consensus_tsv} (class={cls}) "
              f"and {cand_csv}", file=sys.stderr)

    if args.no_exclude:
        exclude = set()
        print("--no-exclude: keeping every row (domain sample, not a training pool)", file=sys.stderr)
    else:
        exclude = build_exclusion_set(
            all_rows, args.other_arena_consensus, args.other_arena_nn, args.curated_csv, args.link)

        if args.union_with:
            prior = _gene_ids(args.union_with)
            n_before = len(exclude)
            exclude |= prior
            print(f"unioned with {args.union_with}: {len(prior)} genes, "
                  f"{len(exclude) - n_before} new -> {len(exclude)} total", file=sys.stderr)

        with open(args.exclude_out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["gene_id"])
            for gid in sorted(exclude):
                w.writerow([gid])
        print(f"wrote {args.exclude_out}: {len(exclude)} genes", file=sys.stderr)

    kept = [(tsv_row, region) for tsv_row, region in aligned_rows
            if tsv_row["gene_id"] not in exclude]
    n_excluded = len(aligned_rows) - len(kept)
    print(f"{n_excluded} rows excluded, {len(kept)} kept", file=sys.stderr)

    kept_genes = {tsv_row["gene_id"] for tsv_row, _region in kept}
    assert not (kept_genes & exclude), "kept rows still overlap the exclusion set"

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        for _tsv_row, region in kept:
            writer.writerow([region, args.bam1, args.bam2])

    with open(args.labels_out, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ix", "event_id", "label", "kind"])
        for ix, (tsv_row, _region) in enumerate(kept):
            writer.writerow([f"{ix:05d}", tsv_row["suppa_event_id"], tsv_row["class"], args.kind])

    print(f"wrote {len(kept)} rows to {args.out} and {args.labels_out}")


if __name__ == "__main__":
    main()
