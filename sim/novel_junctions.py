"""Synthesize one low-abundance, never-annotated-junction transcript per gene.

Noise layer 4 (plans/signal-native-v2-noise-layer4.md): real data always has
some background of splice junctions that never made it into the annotation
(mis-splicing, incomplete annotation). For each gene already in the
experiment (design + neighbour genes), pick its most-exon-rich real
transcript and derive one synthetic variant by one of two operations:

- **shifted splice site**: move one donor or acceptor of a real intron by a
  random offset in [-20,-3] U [3,20] bp, producing a genuinely different
  intron.
- **within-transcript exon skip-join**: drop one interior exon, directly
  joining its flanking exons -- a skipped-exon-style junction that was never
  annotated as a transcript of this gene. (Cross-transcript exon joins were
  considered and deferred -- see the plan doc.)

Either way, the resulting intron is checked against every annotated intron
of every transcript of the gene (from all isoforms, not just the one
picked) and re-drawn if it happens to coincide with a real one -- the whole
point of this layer is a junction that is *not* in the annotation.

Usage:
    from sim.novel_junctions import generate_novel_variants
    variants = generate_novel_variants(exon_structure_by_gene, rng)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

MIN_EXON_LEN = 10  # bp, floor after a boundary shift
MIN_INTRON_LEN = 20  # bp, floor after a boundary shift
SHIFT_LOW, SHIFT_HIGH = 3, 20
MAX_TRIES = 10

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def novel_id(gene_id: str, op: str) -> str:
    return f"NOVEL_{gene_id}_{op}"


def _introns_of(exons: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """exons ascending by genomic start -> [(intron_start, intron_end), ...],
    1-based inclusive intron bounds (first base after exon i's end, last
    base before exon i+1's start)."""
    return [(exons[i][1] + 1, exons[i + 1][0] - 1) for i in range(len(exons) - 1)]


def annotated_introns_by_gene(
    exon_structure_by_gene: Dict[str, Dict[str, Tuple[str, str, List[Tuple[int, int]]]]]
) -> Dict[str, set]:
    """gene_id -> set of every intron (start, end) appearing in any of its
    annotated transcripts."""
    out: Dict[str, set] = {}
    for gene_id, transcripts in exon_structure_by_gene.items():
        introns = set()
        for _chrom, _strand, exons in transcripts.values():
            introns.update(_introns_of(exons))
        out[gene_id] = introns
    return out


def choose_transcript(transcripts: Dict[str, Tuple[str, str, List[Tuple[int, int]]]]) -> str:
    """Most-exon-rich transcript of the gene, ties broken by transcript_id
    for determinism."""
    return max(transcripts, key=lambda tx: (len(transcripts[tx][2]), tx))


@dataclass
class NovelVariant:
    gene_id: str
    transcript_id: str
    chrom: str
    strand: str
    op: str  # "shift_donor" | "shift_acceptor" | "skip_join"
    exons: List[Tuple[int, int]]
    detail: str
    novel_intron: Tuple[int, int]


def _try_shift(
    exons: List[Tuple[int, int]], rng: np.random.Generator, annotated: set
) -> Optional[Tuple[List[Tuple[int, int]], str, str, Tuple[int, int]]]:
    if len(exons) < 2:
        return None
    for _ in range(MAX_TRIES):
        i = int(rng.integers(0, len(exons) - 1))
        left_start, left_end = exons[i]
        right_start, right_end = exons[i + 1]
        side = "donor" if rng.random() < 0.5 else "acceptor"
        sign = 1 if rng.random() < 0.5 else -1
        offset = sign * int(rng.integers(SHIFT_LOW, SHIFT_HIGH + 1))

        new_exons = list(exons)
        if side == "donor":
            new_left_end = left_end + offset
            if new_left_end - left_start + 1 < MIN_EXON_LEN:
                continue
            if right_start - new_left_end - 1 < MIN_INTRON_LEN:
                continue
            new_exons[i] = (left_start, new_left_end)
            novel_intron = (new_left_end + 1, right_start - 1)
        else:
            new_right_start = right_start + offset
            if right_end - new_right_start + 1 < MIN_EXON_LEN:
                continue
            if new_right_start - left_end - 1 < MIN_INTRON_LEN:
                continue
            new_exons[i + 1] = (new_right_start, right_end)
            novel_intron = (left_end + 1, new_right_start - 1)

        if novel_intron in annotated:
            continue
        detail = f"intron{i}_{side}_offset{offset:+d}"
        return new_exons, f"shift_{side}", detail, novel_intron
    return None


def _try_skip_join(
    exons: List[Tuple[int, int]], rng: np.random.Generator, annotated: set
) -> Optional[Tuple[List[Tuple[int, int]], str, str, Tuple[int, int]]]:
    if len(exons) < 3:
        return None
    interior = list(range(1, len(exons) - 1))
    rng.shuffle(interior)
    for m in interior[:MAX_TRIES]:
        novel_intron = (exons[m - 1][1] + 1, exons[m + 1][0] - 1)
        if novel_intron in annotated:
            continue
        new_exons = exons[:m] + exons[m + 1 :]
        detail = f"drop_exon{m}_of{len(exons)}"
        return new_exons, "skip_join", detail, novel_intron
    return None


def generate_gene_variant(
    gene_id: str,
    transcripts: Dict[str, Tuple[str, str, List[Tuple[int, int]]]],
    rng: np.random.Generator,
    annotated: set,
) -> Optional[NovelVariant]:
    tx_id = choose_transcript(transcripts)
    chrom, strand, exons = transcripts[tx_id]

    prefer_skip_join = len(exons) >= 3 and rng.random() < 0.5
    result = None
    if prefer_skip_join:
        result = _try_skip_join(exons, rng, annotated)
        if result is None:
            result = _try_shift(exons, rng, annotated)
    else:
        result = _try_shift(exons, rng, annotated)
        if result is None:
            result = _try_skip_join(exons, rng, annotated)

    if result is None:
        return None
    new_exons, op, detail, novel_intron = result
    return NovelVariant(
        gene_id=gene_id,
        transcript_id=tx_id,
        chrom=chrom,
        strand=strand,
        op=op,
        exons=new_exons,
        detail=detail,
        novel_intron=novel_intron,
    )


def generate_novel_variants(
    exon_structure_by_gene: Dict[str, Dict[str, Tuple[str, str, List[Tuple[int, int]]]]],
    rng: np.random.Generator,
) -> List[NovelVariant]:
    annotated = annotated_introns_by_gene(exon_structure_by_gene)
    variants = []
    for gene_id in sorted(exon_structure_by_gene):
        variant = generate_gene_variant(
            gene_id, exon_structure_by_gene[gene_id], rng, annotated[gene_id]
        )
        if variant is not None:
            variants.append(variant)
    return variants


def main(argv=None) -> None:
    import pysam

    from sim.gtf_exon_structure import load_exon_structure

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exon-structure", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    exon_structure_by_gene = load_exon_structure(args.exon_structure)
    rng = np.random.default_rng(args.seed)
    variants = generate_novel_variants(exon_structure_by_gene, rng)

    fasta = pysam.FastaFile(args.reference)
    n_shift = n_skip = n_failed = 0
    with open(args.out_fasta, "w") as fa, open(args.out_tsv, "w", newline="") as tsv:
        writer = csv.DictWriter(
            tsv,
            fieldnames=[
                "gene_id",
                "novel_id",
                "transcript_id",
                "chrom",
                "strand",
                "op",
                "detail",
                "novel_intron_start",
                "novel_intron_end",
                "n_exons",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for v in variants:
            seq_parts = [fasta.fetch(v.chrom, s - 1, e) for s, e in v.exons]
            seq = "".join(seq_parts)
            if v.strand == "-":
                seq = revcomp(seq)
            nid = novel_id(v.gene_id, v.op)
            fa.write(f">{nid}\n")
            for i in range(0, len(seq), 70):
                fa.write(seq[i : i + 70] + "\n")
            writer.writerow(
                dict(
                    gene_id=v.gene_id,
                    novel_id=nid,
                    transcript_id=v.transcript_id,
                    chrom=v.chrom,
                    strand=v.strand,
                    op=v.op,
                    detail=v.detail,
                    novel_intron_start=v.novel_intron[0],
                    novel_intron_end=v.novel_intron[1],
                    n_exons=len(v.exons),
                )
            )
            if v.op == "skip_join":
                n_skip += 1
            else:
                n_shift += 1

    n_failed = len(exon_structure_by_gene) - len(variants)
    print(
        f"genes: {len(exon_structure_by_gene)}, variants: {len(variants)} "
        f"(shift={n_shift}, skip_join={n_skip}, failed={n_failed})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
