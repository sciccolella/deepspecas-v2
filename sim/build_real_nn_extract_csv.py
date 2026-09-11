"""Stage 14 (the NN attack): turn an arena's real NN candidates into an
`extract.csv`/`labels.tsv` batch shaped like `sim/build_extract_csv.py`'s
output, ready for `dsignal.extract` and `sim/merge_batches.py --batch`.

Reads `<arena-dir>/candidates/nn_candidates.tsv` (the caller-derived hard
negatives from `benchmark_build/make_nn_candidates.py`, columns
`source_class, chr, strand, region_start, region_end, gene_id, gene_symbol,
rmats_pvalue, rmats_incleveldiff, suppa_event_id, suppa_dpsi,
suppa_pvalue`), not a sim `design.tsv` -- there is no window sampler here,
the region is the arena's own (already the same shape a sim window would be:
event span + the source class's padding convention, see
`benchmark_build/make_candidate_csvs.PAD`).

Gene-disjointness (`plans/signal-native-v2-arena-nn-attack.md`): drops any
row whose `gene_id` is in the exclusion set built from (a) `ptbp1_arena`'s
own `consensus.tsv` + `candidates/nn_candidates.tsv` gene_id columns
(direct, since both carry gene_id already), and (b) `benchmark_new`'s and
`benchmark`'s regions, approximated via `dsignal.benchmark_split.cluster`'s
single-linkage-at-50kb "same locus" proxy (those CSVs carry regions only, no
gene_id) -- the same proxy that benchmark's own dev/test split already
uses, rather than inventing a new threshold. Writes the pooled exclusion set
to `--exclude-out` (default `sim/experiment/exclude_genes_nn_attack.tsv`) and
asserts the kept rows' gene_ids do not intersect it.

`event_id` is exactly `suppa_event_id` -- already shaped
`"{gene_id};{suffix}"` (e.g. `"ENSG00000160209;A3:21:...`"`), the same
convention `sim/catalogue.py`/`sim/select_events.py` use and
`sim/gene_split.gene_id_of` recovers by splitting on the first `;`. `label`
is `"NN"` for every row (this attack is NN-only, no pair-2 positives -- see
the pre-registration's fixed pool-composition decision); `kind` is
`"real_nn_aqr"` for provenance.

Regions come from `<arena-dir>/candidates/NN.csv` (already padded per the
source class's convention by `make_nn_candidates.py`), not recomputed here.
`nn_candidates.tsv` and `NN.csv` are written from the same DataFrame in the
same iteration order (`make_nn_candidates.py`), so they are 1:1 by row
index -- this is verified, not assumed: each row's `NN.csv` region is
recomputed from `nn_candidates.tsv`'s `chr`/`region_start`/`region_end` and
the class's pad and asserted equal before anything is trusted.

Usage::

    python -m sim.build_real_nn_extract_csv \\
        --arena-dir aqr_arena --bam1 aqr_work/align/c1_aqr.bam \\
        --bam2 aqr_work/align/c2_aqr.bam \\
        --other-arena-consensus ptbp1_arena/consensus.tsv \\
        --other-arena-nn ptbp1_arena/candidates/nn_candidates.tsv \\
        --curated-csv benchmark_new/A3.csv benchmark_new/A5.csv benchmark_new/ES.csv \\
            benchmark_new/IR.csv benchmark_new/NN.csv \\
            benchmark/A3.csv benchmark/A5.csv benchmark/ES.csv benchmark/IR.csv \\
        --out sim/experiment/real_nn_aqr/extract.csv \\
        --labels-out sim/experiment/real_nn_aqr/labels.tsv \\
        --exclude-out sim/experiment/exclude_genes_nn_attack.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys

from benchmark_build.make_candidate_csvs import PAD
from dsignal.benchmark_split import cluster
from dsignal.predict import read_prediction_csv


def _gene_ids(path: str, col: str = "gene_id") -> set[str]:
    with open(path, newline="") as f:
        return {r[col] for r in csv.DictReader(f, delimiter="\t")}


def _region_events(csv_paths):
    """[(stem, region, chrom, start, end, None)], the shape
    `dsignal.benchmark_split.cluster` expects."""
    out = []
    for path in csv_paths:
        for region, _bam1, _bam2 in read_prediction_csv(path):
            chrom, rest = region.split(":")
            start, end = rest.split("-")
            out.append((path, region, chrom, int(start), int(end), None))
    return out


def build_exclusion_set(nn_rows, other_consensus, other_nn, curated_csvs, link: int) -> set[str]:
    excluded = set()

    # (a) ptbp1_arena's own genes, direct
    for path in (other_consensus, other_nn):
        excluded |= _gene_ids(path)
    n_direct = len(excluded)

    # (b) benchmark_new/benchmark regions, clustered at 50kb with pair-2's own
    # NN regions -- any pair-2 region sharing a cluster with a curated region
    # has its whole gene excluded.
    pair2_events = [
        (r["gene_id"], r["chr"], int(r["region_start"]), int(r["region_end"]))
        for r in nn_rows
    ]
    curated_events = _region_events(curated_csvs)
    combined = (
        [(gid, gid, chrom, start, end, None) for gid, chrom, start, end in pair2_events]
        + curated_events
    )
    clusters = cluster(combined, link)
    n_pair2 = len(pair2_events)
    for cl in clusters:
        has_curated = any(i >= n_pair2 for i in cl)
        if not has_curated:
            continue
        for i in cl:
            if i < n_pair2:
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
    ap.add_argument("--link", type=int, default=50000,
                     help="single-linkage gap for the same-locus proxy (default: 50000, "
                     "matching dsignal.benchmark_split's own default)")
    ap.add_argument("--no-exclude", action="store_true",
                     help="Stage 18.0: skip gene exclusion and keep every row -- for a "
                     "cohort extracted as a *domain sample* (ptbp1_arena shares 92 genes "
                     "with benchmark_new dev and can never be a training pool for a model "
                     "scored on benchmark_new test). --other-arena-*/--curated-csv/"
                     "--exclude-out are then unused")
    ap.add_argument("--kind", default="real_nn_aqr",
                     help="labels.tsv `kind` value (default %(default)s); must start with "
                     "'real_' -- dsignal.dataset derives the domain label from that prefix")
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

    nn_tsv = f"{args.arena_dir}/candidates/nn_candidates.tsv"
    nn_csv = f"{args.arena_dir}/candidates/NN.csv"
    with open(nn_tsv, newline="") as f:
        nn_rows = list(csv.DictReader(f, delimiter="\t"))
    nn_csv_rows = list(read_prediction_csv(nn_csv))
    assert len(nn_rows) == len(nn_csv_rows), (
        f"{nn_tsv} has {len(nn_rows)} rows but {nn_csv} has {len(nn_csv_rows)} -- "
        "not 1:1, cannot pair by row index")

    # verify the row-index pairing before trusting it (see module docstring)
    for i, (tsv_row, (region, bam1, bam2)) in enumerate(zip(nn_rows, nn_csv_rows)):
        pad = PAD[tsv_row["source_class"]]
        expect_start = max(1, int(tsv_row["region_start"]) - pad)
        expect_end = int(tsv_row["region_end"]) + pad
        expect_region = f"{tsv_row['chr']}:{expect_start}-{expect_end}"
        assert region == expect_region, (
            f"row {i}: {nn_tsv} implies region {expect_region!r} but {nn_csv} has "
            f"{region!r} -- the two files are not row-aligned")
    print(f"verified {len(nn_rows)} rows 1:1 between {nn_tsv} and {nn_csv}", file=sys.stderr)

    if args.no_exclude:
        exclude = set()
        print("--no-exclude: keeping every row (domain sample, not a training pool)", file=sys.stderr)
    else:
        exclude = build_exclusion_set(
            nn_rows, args.other_arena_consensus, args.other_arena_nn, args.curated_csv, args.link)

        with open(args.exclude_out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["gene_id"])
            for gid in sorted(exclude):
                w.writerow([gid])
        print(f"wrote {args.exclude_out}: {len(exclude)} genes", file=sys.stderr)

    kept = [(tsv_row, region) for tsv_row, (region, _b1, _b2) in zip(nn_rows, nn_csv_rows)
            if tsv_row["gene_id"] not in exclude]
    n_excluded = len(nn_rows) - len(kept)
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
            writer.writerow([f"{ix:05d}", tsv_row["suppa_event_id"], "NN", args.kind])

    print(f"wrote {len(kept)} rows to {args.out} and {args.labels_out}")


if __name__ == "__main__":
    main()
