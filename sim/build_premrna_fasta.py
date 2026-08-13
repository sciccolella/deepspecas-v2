"""Extract full genomic-span pre-mRNA background sequences per gene.

Noise layer 2 (plans/signal-native-v2-noise-layer2.md): an unspliced
transcript spanning each pilot-design gene's full genomic extent, added to
the count matrix at low abundance (sim/abundance.add_premrna_row). Unlike
every other transcript in the experiment, this is not extracted via gffread
from the GTF (it isn't an annotated transcript) -- it's a direct genomic
slice over the gene's full span, reverse-complemented on the minus strand.

With --neighbors (noise layer 3, plans/signal-native-v2-noise-layer3.md),
also emits a PREMRNA_<gene_id> entry for every neighbouring/antisense gene
found by sim/gene_neighbors.py, since those genes transcribe pre-mRNA too.

Usage:
    python -m sim.build_premrna_fasta \
        --design sim/design.tsv --gene-spans sim/gene_spans.tsv \
        --reference ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --neighbors sim/gene_neighbors.tsv \
        --out sim/experiment/premrna.fa
"""

from __future__ import annotations

import argparse
import csv
import sys

import pysam

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def premrna_id(gene_id: str) -> str:
    return f"PREMRNA_{gene_id}"


def load_gene_spans(path: str):
    spans = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            spans[row["gene_id"]] = (
                row["chrom"],
                int(row["start"]),
                int(row["end"]),
                row["strand"],
            )
    return spans


def load_design_genes(path: str):
    genes = set()
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            genes.add(row["gene_id"])
    return genes


def load_neighbor_genes(path: str):
    genes = set()
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            genes.add(row["neighbor_gene_id"])
    return genes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True)
    ap.add_argument("--gene-spans", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--neighbors",
        default=None,
        help="sim/gene_neighbors.py output; when given, also emits "
        "PREMRNA_<gene_id> for every neighbouring/antisense gene (noise "
        "layer 3), not just design genes.",
    )
    args = ap.parse_args(argv)

    genes = load_design_genes(args.design)
    if args.neighbors:
        genes |= load_neighbor_genes(args.neighbors)
    spans = load_gene_spans(args.gene_spans)

    fasta = pysam.FastaFile(args.reference)
    n_written = 0
    n_missing = 0
    with open(args.out, "w") as out:
        for gene_id in sorted(genes):
            if gene_id not in spans:
                n_missing += 1
                continue
            chrom, start, end, strand = spans[gene_id]
            # gene_spans.tsv is 1-based inclusive (GTF convention);
            # pysam.fetch is 0-based half-open.
            seq = fasta.fetch(chrom, start - 1, end)
            if strand == "-":
                seq = revcomp(seq)
            out.write(f">{premrna_id(gene_id)}\n")
            for i in range(0, len(seq), 70):
                out.write(seq[i : i + 70] + "\n")
            n_written += 1

    print(
        f"wrote {n_written} pre-mRNA sequences ({n_missing} genes missing spans) to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
