"""Per-transcript ordered exon coordinates, for genes already in the experiment.

Noise layer 4 (plans/signal-native-v2-noise-layer4.md) needs real exon
boundaries -- not just exonic length (sim/gtf_transcript_lengths.py) -- to
synthesize novel-junction transcripts (shifted splice sites / exon
skip-joins). Simulation-only input, same status as every other GTF-derived
table here (never a classification input).

Restricted to genes already in the experiment (design + neighbour genes)
rather than parsed genome-wide, since that's all noise layer 4 needs and it
keeps this cheap.

Usage:
    python -m sim.gtf_exon_structure \
        --gtf ref/Homo_sapiens.GRCh38.99.gtf \
        --design sim/design.tsv --neighbors sim/gene_neighbors.tsv \
        --out sim/exon_structure.tsv

Output TSV columns: transcript_id, gene_id, chrom, strand, exon_starts,
exon_ends -- exon_starts/exon_ends are comma-separated 1-based-inclusive GTF
coordinates, in ascending genomic order (i.e. transcript order on + strand,
reverse of transcript order on - strand -- callers that need transcript
order must reverse for - strand transcripts themselves).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

_ATTR_RE = re.compile(r'(\w+) "([^"]*)"')


def _parse_attrs(field: str) -> dict:
    return dict(_ATTR_RE.findall(field))


def _load_genes(design_path: str, neighbors_path: str | None) -> set:
    genes = set()
    with open(design_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            genes.add(row["gene_id"])
    if neighbors_path:
        with open(neighbors_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                genes.add(row["neighbor_gene_id"])
    return genes


def compute_exon_structure(gtf_path: str, genes: set):
    """Yields (transcript_id, gene_id, chrom, strand, exon_starts, exon_ends)
    for every transcript belonging to a gene in `genes`, exons sorted
    ascending by genomic start."""
    exons: dict = {}
    meta: dict = {}
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
            if tx_id is None or gene_id is None or gene_id not in genes:
                continue
            exons.setdefault(tx_id, []).append((int(start), int(end)))
            meta[tx_id] = (gene_id, chrom, strand)

    for tx_id, coords in exons.items():
        gene_id, chrom, strand = meta[tx_id]
        coords.sort()
        starts = ",".join(str(s) for s, _ in coords)
        ends = ",".join(str(e) for _, e in coords)
        yield tx_id, gene_id, chrom, strand, starts, ends


def load_exon_structure(path: str) -> dict:
    """gene_id -> {transcript_id: (chrom, strand, [(start, end), ...])},
    exons in ascending genomic order (as written by compute_exon_structure)."""
    by_gene: dict = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            starts = [int(x) for x in row["exon_starts"].split(",")]
            ends = [int(x) for x in row["exon_ends"].split(",")]
            exon_list = list(zip(starts, ends))
            by_gene.setdefault(row["gene_id"], {})[row["transcript_id"]] = (
                row["chrom"],
                row["strand"],
                exon_list,
            )
    return by_gene


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--neighbors", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    genes = _load_genes(args.design, args.neighbors)
    n = 0
    with open(args.out, "w") as out:
        out.write("transcript_id\tgene_id\tchrom\tstrand\texon_starts\texon_ends\n")
        for row in compute_exon_structure(args.gtf, genes):
            out.write("\t".join(row) + "\n")
            n += 1
    print(f"wrote {n} transcripts across {len(genes)} genes to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
