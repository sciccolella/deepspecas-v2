#!/usr/bin/env python
"""Stage 9 / Stage 0.2: re-partition ``benchmark_new/curation.tsv``'s 485
unambiguous rows into ``region,bam1,bam2`` CSVs by curated ``truth``, using the
old naming convention (``CSV_LABEL``: ``ES``==CSE, ``IR``==RI) so
``dsignal.score_benchmark.CSV_LABEL`` keeps working once it also maps
``NN``->``NN``.

Regions are **not re-padded**: each row's region is the exact window from the
matching ``benchmark_new_uncurated/<csv_source>.csv`` row it was curated from,
padded per the original *candidate* class (Stage 8 Track A3's
``make_candidate_csvs.py``: CSE +100bp/edge, RI/A3/A5 +0). Curation only moves
rows between truth labels; it does not touch window geometry -- that is what
Stage 0's geometry-parity check (``audit_noise padding``) verifies.

The 87 ambiguous rows and the 1 blank stay in ``curation.tsv`` only.

Usage::

    python benchmark_build/make_curated_csvs.py \\
        benchmark_new/curation.tsv benchmark_new_uncurated benchmark_new
"""
from __future__ import annotations

import csv
import sys

TRUTH_CSV = {"A3": "A3", "A5": "A5", "CSE": "ES", "RI": "IR", "NN": "NN"}


def main() -> None:
    curation_tsv, uncurated_dir, out_dir = sys.argv[1:4]

    with open(curation_tsv, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if not int(r["ambiguous"])]

    bam1, bam2 = f"{out_dir}/c1.bam", f"{out_dir}/c2.bam"

    by_csv = {name: [] for name in TRUTH_CSV.values()}
    bed_rows = []
    for r in rows:
        out_name = TRUTH_CSV[r["truth"]]
        region = r["region"]
        by_csv[out_name].append(f"{region},{bam1},{bam2}")
        chrom, rest = region.split(":")
        start, end = rest.split("-")
        bed_rows.append((chrom, int(start) - 1, int(end)))  # BED is 0-based half-open

    for name, lines in by_csv.items():
        out_path = f"{out_dir}/{name}.csv"
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {out_path}: {len(lines)} events")

    with open(f"{out_dir}/events.bed", "w") as f:
        for chrom, start, end in bed_rows:
            f.write(f"{chrom}\t{start}\t{end}\n")
    print(f"wrote {out_dir}/events.bed: {len(bed_rows)} regions")

    assert sum(len(v) for v in by_csv.values()) == 485


if __name__ == "__main__":
    main()
