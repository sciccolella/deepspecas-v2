"""Verification checks for dsignal/extract.py, run against benchmark/*.csv.

Two independent checks, per plans/signal-native-v2.md "Verification":

1. Coverage/skip parity against `plot2s.get_coords_from_bam`, with filters
   disabled so the two code paths see the same reads. `get_coords_from_bam`
   folds D into "coverage" and unconditionally zeroes index 0; this script
   adds delcov back into extractor coverage and ignores index 0 to compare
   the parts of the semantics that actually coincide.
2. Junction-identity parity: the jI/jM tag path vs the CIGAR-walk path (using
   --ref for motif) must agree exactly on the junction set (coordinates,
   count, NH-weighted count, min anchor) and on strand, with filters
   disabled. This is the "no GTF, tags vs from-scratch CIGAR walk" audit.

Usage:
    python -m dsignal.verify --csv benchmark/A3.csv benchmark/A5.csv \
        benchmark/ES.csv benchmark/IR.csv [--ref /path/to/genome.fa]

Without --ref, check 2 is skipped for reads with no jI/jM tags (none
expected here, since benchmark/*.bam are STAR output) and motif/strand
comparison in check 2 is skipped (coordinates are still compared).
"""

import argparse
import csv
import sys

import numpy as np
import pysam

import plot2s
from dsignal.extract import ReadFilters, extract_condition, parse_region


def _regions_from_csvs(csv_paths):
    for path in csv_paths:
        with open(path) as f:
            for row in csv.reader(f):
                if not row:
                    continue
                yield parse_region(row[0]), row[1], row[2]


def check_coverage_parity(regions, verbose: bool) -> int:
    """Check 1: extractor cov+delcov vs get_coords_from_bam's covs, skip vs
    skips, with all read-level filters off (min_anchor=0, max_nh=None,
    min_mapq=0) so both paths process the identical read set."""
    filters = ReadFilters(min_anchor=0, max_nh=None, min_mapq=0)
    n_checked = 0
    n_failed = 0
    bam_cache = {}

    for region, bam1_path, bam2_path in regions:
        for bam_path in (bam1_path, bam2_path):
            if bam_path not in bam_cache:
                bam_cache[bam_path] = pysam.AlignmentFile(bam_path, "rb")
            bam = bam_cache[bam_path]

            ref_covs, ref_skips = plot2s.get_coords_from_bam(bam, region, mincov=0)
            cond = extract_condition(bam, region, ref=None, filters=filters, use_tags=True)

            ours_cov = cond.cov.astype(np.int64) + cond.delcov.astype(np.int64)
            n_checked += 1

            # get_coords_from_bam unconditionally zeroes index 0; skip it.
            cov_ok = np.array_equal(ours_cov[1:], ref_covs[1:].astype(np.int64))
            ref_skips64 = ref_skips.astype(np.int64)
            our_skips = _condition_skips(bam, region, filters)
            skip_ok = np.array_equal(our_skips[1:], ref_skips64[1:])

            if not (cov_ok and skip_ok):
                n_failed += 1
                print(
                    f"COVERAGE MISMATCH region={region} bam={bam_path} "
                    f"cov_ok={cov_ok} skip_ok={skip_ok}",
                    file=sys.stderr,
                )
            elif verbose:
                print(f"  ok  cov/skip  region={region} bam={bam_path}", file=sys.stderr)

    for bam in bam_cache.values():
        bam.close()

    print(f"[coverage parity] {n_checked - n_failed}/{n_checked} regions match")
    return n_failed


def _condition_skips(bam, region, filters):
    """Recompute just the skip array via extract_condition (kept separate
    from cov since ConditionRecord doesn't expose skip directly)."""
    chrom, start, end = region
    n = end - start
    skip_diff = np.zeros(n + 1, dtype=np.int64)
    for read in bam.fetch(chrom, start, end):
        cigartuples = read.cigartuples
        if cigartuples is None or read.is_unmapped:
            continue
        pos = read.reference_start
        for op, oplen in cigartuples:
            if op not in (0, 2, 3, 7, 8):  # M, D, N, =, X consume the reference
                continue
            op_start, op_end = pos, pos + oplen
            if op == 3:  # N
                s, e = max(op_start - start, 0), min(op_end - start, n)
                if e > s:
                    skip_diff[s] += 1
                    skip_diff[e] -= 1
            pos = op_end
    return np.cumsum(skip_diff[:n])


def check_junction_parity(regions, ref_path: str, verbose: bool) -> int:
    """Check 2: jI/jM-tag-derived junctions vs CIGAR-walk-derived junctions
    (using --ref for motif), filters off. Coordinates/counts/anchors must
    match exactly; strand must match when ref is given."""
    filters = ReadFilters(min_anchor=0, max_nh=None, min_mapq=0)
    ref = pysam.FastaFile(ref_path) if ref_path else None
    n_checked = 0
    n_failed = 0
    bam_cache = {}

    for region, bam1_path, bam2_path in regions:
        for bam_path in (bam1_path, bam2_path):
            if bam_path not in bam_cache:
                bam_cache[bam_path] = pysam.AlignmentFile(bam_path, "rb")
            bam = bam_cache[bam_path]

            tag_cond = extract_condition(bam, region, ref=None, filters=filters, use_tags=True)
            cigar_cond = extract_condition(bam, region, ref=ref, filters=filters, use_tags=False)
            n_checked += 1

            tag_keys = {(r["start"], r["end"]): r for r in tag_cond.junctions}
            cigar_keys = {(r["start"], r["end"]): r for r in cigar_cond.junctions}

            coord_ok = set(tag_keys) == set(cigar_keys)
            count_ok = coord_ok and all(
                tag_keys[k]["count"] == cigar_keys[k]["count"]
                and tag_keys[k]["min_anchor"] == cigar_keys[k]["min_anchor"]
                for k in tag_keys
            )
            strand_ok = True
            if ref is not None and coord_ok:
                strand_ok = all(
                    tag_keys[k]["strand"] == cigar_keys[k]["strand"] for k in tag_keys
                )

            if not (coord_ok and count_ok and strand_ok):
                n_failed += 1
                print(
                    f"JUNCTION MISMATCH region={region} bam={bam_path} "
                    f"coord_ok={coord_ok} count_ok={count_ok} strand_ok={strand_ok}",
                    file=sys.stderr,
                )
                if verbose:
                    print("  tag-only:", set(tag_keys) - set(cigar_keys), file=sys.stderr)
                    print("  cigar-only:", set(cigar_keys) - set(tag_keys), file=sys.stderr)
            elif verbose:
                print(f"  ok  junctions  region={region} bam={bam_path}", file=sys.stderr)

    for bam in bam_cache.values():
        bam.close()
    if ref is not None:
        ref.close()

    print(f"[junction parity] {n_checked - n_failed}/{n_checked} regions match")
    return n_failed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="reference FASTA for motif/strand cross-check")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    regions = list(_regions_from_csvs(args.csv))
    print(f"loaded {len(regions)} events from {len(args.csv)} CSV(s)", file=sys.stderr)

    n_fail_cov = check_coverage_parity(regions, args.verbose)
    n_fail_junc = check_junction_parity(regions, args.ref, args.verbose)

    if not args.ref:
        print(
            "NOTE: --ref not given, strand cross-check (jM vs reference dinucleotides) skipped",
            file=sys.stderr,
        )

    if n_fail_cov or n_fail_junc:
        sys.exit(1)


if __name__ == "__main__":
    main()
