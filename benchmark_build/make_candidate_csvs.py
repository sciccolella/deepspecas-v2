#!/usr/bin/env python
"""Stage 8 / Track A3: consensus.tsv -> candidate region,bam1,bam2 CSVs +
events.bed, reproducing the old benchmark's window/event-span padding
convention (python -m dsignal.audit_noise padding): ES/CSE +100bp each edge,
IR (RI) +0, A3/A5 +0 (the measured -1 is an off-by-one coordinate-convention
artifact, not real padding).

Manual curation (plan Stage 8 Track A4) is deferred -- this writes the FULL
uncurated consensus pool, not a ~140-event curated subset. Output goes to
benchmark_new_uncurated/, not benchmark_new/, so it can't be mistaken for
the frozen, split, curated set the plan's Track C one-shot test spend
requires.

Usage: make_candidate_csvs.py <consensus_tsv> <out_dir> <bam1> <bam2>
"""
import sys

import pandas as pd

PAD = {"CSE": 100, "RI": 0, "A3": 0, "A5": 0}
CSV_NAME = {"CSE": "ES", "RI": "IR", "A3": "A3", "A5": "A5"}  # CSV_LABEL convention: ES==CSE, IR==RI


def main():
    consensus_tsv, out_dir, bam1, bam2 = sys.argv[1:5]
    df = pd.read_csv(consensus_tsv, sep="\t")

    bed_rows = []
    for cls, sub in df.groupby("class"):
        pad = PAD[cls]
        rows = []
        for _, r in sub.iterrows():
            start = max(1, int(r["region_start"]) - pad)
            end = int(r["region_end"]) + pad
            region = f"{r['chr']}:{start}-{end}"
            rows.append(f"{region},{bam1},{bam2}")
            bed_rows.append((r["chr"], start - 1, end))  # BED is 0-based half-open
        out_csv = f"{out_dir}/{CSV_NAME[cls]}.csv"
        with open(out_csv, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"wrote {out_csv}: {len(rows)} events")

    with open(f"{out_dir}/events.bed", "w") as f:
        for chrom, start, end in bed_rows:
            f.write(f"{chrom}\t{start}\t{end}\n")
    print(f"wrote {out_dir}/events.bed: {len(bed_rows)} regions")


if __name__ == "__main__":
    main()
