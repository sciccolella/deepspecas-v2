"""Extract gene genomic spans (chrom, strand, start, end) from an Ensembl GTF.

Simulation-only input, used by sim/select_events.py to size simulation
windows for genes/events and for NN-negative loci with no catalogue event.

Usage:
    python -m sim.gtf_gene_spans --gtf ref/Homo_sapiens.GRCh38.99.gtf \
        --out sim/gene_spans.tsv

Output TSV columns: gene_id, chrom, strand, start, end
"""

import argparse
import sys

from sim.gtf_transcript_lengths import _parse_attrs


def compute_spans(gtf_path: str):
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attr_field = fields
            attrs = _parse_attrs(attr_field)
            gene_id = attrs.get("gene_id")
            if gene_id is None:
                continue
            yield gene_id, chrom, strand, int(start), int(end)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    n = 0
    with open(args.out, "w") as out:
        out.write("gene_id\tchrom\tstrand\tstart\tend\n")
        for row in compute_spans(args.gtf):
            out.write("\t".join(str(x) for x in row) + "\n")
            n += 1
    print(f"wrote {n} genes to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
