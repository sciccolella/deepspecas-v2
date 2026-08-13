"""Build a `region,bam1,bam2` CSV from a pilot design.tsv for
`dsignal.extract`'s CLI. All design rows share the same pooled c1.bam/c2.bam
(unlike the real benchmark's per-event BAM pairs).

Usage:
    python -m sim.build_extract_csv --design sim/design.tsv \
        --bam1 sim/experiment/aligned/c1.bam \
        --bam2 sim/experiment/aligned/c2.bam \
        --out sim/experiment/extract.csv
"""

from __future__ import annotations

import argparse
import csv


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True)
    ap.add_argument("--bam1", required=True)
    ap.add_argument("--bam2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--labels-out",
        required=True,
        help="ix,event_id,label,kind TSV, row order/index matches --out and "
        "the dsignal.extract CLI's {ix:05d}.*.npz output naming",
    )
    args = ap.parse_args(argv)

    with open(args.design) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            region = f"{row['chrom']}:{row['region_start']}-{row['region_end']}"
            writer.writerow([region, args.bam1, args.bam2])

    with open(args.labels_out, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ix", "event_id", "label", "kind"])
        for ix, row in enumerate(rows):
            writer.writerow([f"{ix:05d}", row["event_id"], row["label"], row["kind"]])

    print(f"wrote {len(rows)} rows to {args.out} and {args.labels_out}")


if __name__ == "__main__":
    main()
