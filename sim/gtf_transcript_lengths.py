"""Compute per-transcript exonic length + gene mapping from an Ensembl GTF.

Simulation-only input (per plans/signal-native-v2.md: "GTF used for
simulation only, never a classification input"). Used to size per-transcript
read counts realistically in sim/abundance.py -- polyester needs approximate
transcript lengths to convert PSI-target counts into believable depth.

Usage:
    python -m sim.gtf_transcript_lengths --gtf ref/Homo_sapiens.GRCh38.99.gtf \
        --out sim/transcript_lengths.tsv

Output TSV columns: transcript_id, gene_id, chrom, strand, exonic_length
"""

import argparse
import re
import sys

_ATTR_RE = re.compile(r'(\w+) "([^"]*)"')


def _parse_attrs(field: str) -> dict:
    return dict(_ATTR_RE.findall(field))


def compute_lengths(gtf_path: str):
    """Yields (transcript_id, gene_id, chrom, strand, exonic_length)."""
    lengths = {}
    meta = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            chrom, _, _, start, end, _, strand, _, attr_field = fields
            attrs = _parse_attrs(attr_field)
            tx_id = attrs.get("transcript_id")
            gene_id = attrs.get("gene_id")
            if tx_id is None or gene_id is None:
                continue
            exon_len = int(end) - int(start) + 1
            lengths[tx_id] = lengths.get(tx_id, 0) + exon_len
            meta[tx_id] = (gene_id, chrom, strand)

    for tx_id, length in lengths.items():
        gene_id, chrom, strand = meta[tx_id]
        yield tx_id, gene_id, chrom, strand, length


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    n = 0
    with open(args.out, "w") as out:
        out.write("transcript_id\tgene_id\tchrom\tstrand\texonic_length\n")
        for row in compute_lengths(args.gtf):
            out.write("\t".join(str(x) for x in row) + "\n")
            n += 1
    print(f"wrote {n} transcripts to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
