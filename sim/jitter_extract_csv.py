"""Stage 18.0: the *jittered-pad* variant of a real extraction CSV.

Real windows follow a fixed per-class convention (event span + 0 bp for
A3/A5/RI, + 100 bp for CSE: `benchmark_build/make_candidate_csvs.PAD`, which
`dsignal.audit_noise padding` measured on the curated benchmark), while the
simulation draws `event span + U[0, 100] bp per edge` label-independently
(`sim/window.py` v3). A real-vs-sim discriminator could therefore read the
domain off the window geometry alone -- or off anything that co-varies with
it (edge coverage, junctions that reach just outside the window). This
writes the same regions with an independent U[0, pad] extension drawn per
edge from a seeded RNG, so 18.1 can extract the pool twice and measure how
much of the domain signal the window convention carries.

Row order and count are preserved, so the original `labels.tsv` (keyed by
`ix`) applies unchanged to the jittered npz directory.

Usage::

    python -m sim.jitter_extract_csv --csv sim/experiment/real_pos_ptbp1/extract.csv \\
        --out sim/experiment/real_pos_ptbp1/extract.jitter.csv --seed 18
"""

from __future__ import annotations

import argparse
import csv

import numpy as np


def jitter_rows(rows, rng: np.random.Generator, pad: int):
    out = []
    for region, bam1, bam2 in rows:
        chrom, span = region.split(":")
        start, end = (int(x) for x in span.split("-"))
        left, right = (int(v) for v in rng.integers(0, pad + 1, size=2))
        out.append((f"{chrom}:{max(1, start - left)}-{end + right}", bam1, bam2))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="region,bam1,bam2 extraction CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=int, default=100, help="U[0, pad] bp added per edge")
    ap.add_argument("--seed", type=int, default=18)
    args = ap.parse_args(argv)

    with open(args.csv, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    jittered = jitter_rows(rows, np.random.default_rng(args.seed), args.pad)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(jittered)
    print(f"wrote {len(jittered)} jittered rows (U[0,{args.pad}] per edge, seed {args.seed}) to {args.out}")


if __name__ == "__main__":
    main()
