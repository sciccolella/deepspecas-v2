"""Select a balanced pilot simulation design: positives (A3/A5/CSE/RI),
plain NN negatives, and hard NN negatives (real events with target dPSI~0).

One gene is used by at most one selected event, so the per-gene abundance
matrix (sim/abundance.py) never has to reconcile two simulated events in the
same gene. Region windows come straight from the SUPPA event span (positives)
or the gene's genomic span (NN negatives) -- window randomization/jitter is
a later step (plan Phase 2.5), not applied here.

Usage:
    python -m sim.select_events \
        --events-dir sim/events --gene-spans sim/gene_spans.tsv \
        --transcript-lengths sim/transcript_lengths.tsv \
        --n-per-label 60 --n-nn-plain 60 --n-nn-hard 60 \
        --out sim/design.tsv --seed 0

Output TSV columns: event_id, gene_id, label, kind, chrom, strand,
region_start, region_end, psi1, psi2
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from sim.abundance import load_transcript_lengths
from sim.catalogue import Event, build_catalogue
from sim.window import sample_window


def _load_chrom_lengths(path: str):
    """Parses a `samtools faidx` .fai file: chrom, length, offset, linebases, linewidth."""
    lengths = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            lengths[parts[0]] = int(parts[1])
    return lengths


def _load_gene_spans(path: str):
    spans = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            spans[row["gene_id"]] = (row["chrom"], row["strand"], int(row["start"]), int(row["end"]))
    return spans


def _sample_psi_pair(rng: np.random.Generator, dpsi_lo: float, dpsi_hi: float):
    dpsi = rng.uniform(dpsi_lo, dpsi_hi) * rng.choice([-1.0, 1.0])
    mid = rng.uniform(0.25, 0.75)
    psi1 = np.clip(mid - dpsi / 2, 0.02, 0.98)
    psi2 = np.clip(mid + dpsi / 2, 0.02, 0.98)
    return float(psi1), float(psi2)


def select_positives(events, n_per_label, used_genes, rng):
    by_label = {}
    for e in events:
        if e.gene_id in used_genes:
            continue
        by_label.setdefault(e.label, []).append(e)

    rows = []
    for label, candidates in by_label.items():
        idx = rng.permutation(len(candidates))
        picked = 0
        for i in idx:
            e = candidates[i]
            if e.gene_id in used_genes:
                continue
            used_genes.add(e.gene_id)
            psi1, psi2 = _sample_psi_pair(rng, 0.3, 0.8)
            rows.append(
                dict(
                    event_id=e.event_id,
                    gene_id=e.gene_id,
                    label=label,
                    kind="positive",
                    chrom=e.chrom,
                    strand=e.strand,
                    region_start=e.region_start,
                    region_end=e.region_end,
                    psi1=psi1,
                    psi2=psi2,
                )
            )
            picked += 1
            if picked >= n_per_label:
                break
    return rows


def select_hard_negatives(events, n, used_genes, rng):
    idx = rng.permutation(len(events))
    rows = []
    for i in idx:
        e = events[i]
        if e.gene_id in used_genes:
            continue
        used_genes.add(e.gene_id)
        mid = float(rng.uniform(0.15, 0.85))
        rows.append(
            dict(
                event_id=e.event_id,
                gene_id=e.gene_id,
                label="NN",
                kind="nn_hard",
                chrom=e.chrom,
                strand=e.strand,
                region_start=e.region_start,
                region_end=e.region_end,
                psi1=mid,
                psi2=mid,
            )
        )
        if len(rows) >= n:
            break
    return rows


def select_plain_negatives(gene_spans, catalogue_genes, transcripts_by_gene, n, used_genes, rng, min_transcripts=2):
    candidates = [
        g
        for g in gene_spans
        if g not in catalogue_genes
        and g not in used_genes
        and len(transcripts_by_gene.get(g, [])) >= min_transcripts
    ]
    idx = rng.permutation(len(candidates))
    rows = []
    for i in idx:
        g = candidates[i]
        used_genes.add(g)
        chrom, strand, start, end = gene_spans[g]
        rows.append(
            dict(
                event_id=f"{g};NN",
                gene_id=g,
                label="NN",
                kind="nn_plain",
                chrom=chrom,
                strand=strand,
                region_start=start,
                region_end=end,
                psi1="",
                psi2="",
            )
        )
        if len(rows) >= n:
            break
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--gene-spans", required=True)
    ap.add_argument("--transcript-lengths", required=True)
    ap.add_argument("--chrom-lengths", required=True, help="samtools faidx .fai for the reference")
    ap.add_argument("--n-per-label", type=int, default=60)
    ap.add_argument("--n-nn-plain", type=int, default=60)
    ap.add_argument("--n-nn-hard", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # rng drives row selection exactly as before this change (same
    # np.random.default_rng(args.seed) call), so a rerun at the same seed
    # reproduces the identical event/gene set as every prior noise layer --
    # window sampling must not perturb that stream. rng_window is seeded
    # independently (not spawned from the same SeedSequence, which would
    # itself change rng's own stream) and only consumed after all rows are
    # selected.
    rng = np.random.default_rng(args.seed)
    rng_window = np.random.default_rng(args.seed + 1_000_000)
    events = build_catalogue(args.events_dir)
    gene_spans = _load_gene_spans(args.gene_spans)
    transcripts_by_gene = load_transcript_lengths(args.transcript_lengths)
    chrom_lengths = _load_chrom_lengths(args.chrom_lengths)
    catalogue_genes = {e.gene_id for e in events}

    used_genes = set()
    rows = []
    rows += select_positives(events, args.n_per_label, used_genes, rng)
    rows += select_hard_negatives(events, args.n_nn_hard, used_genes, rng)
    rows += select_plain_negatives(
        gene_spans, catalogue_genes, transcripts_by_gene, args.n_nn_plain, used_genes, rng
    )

    for row in rows:
        chrom_len = chrom_lengths[row["chrom"]]
        win_start, win_end = sample_window(row["region_start"], row["region_end"], chrom_len, rng_window)
        row["win_start"] = win_start
        row["win_end"] = win_end

    fields = [
        "event_id", "gene_id", "label", "kind", "chrom", "strand",
        "region_start", "region_end", "win_start", "win_end", "psi1", "psi2",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter

    print(f"wrote {len(rows)} design rows to {args.out}", file=sys.stderr)
    print(f"  by kind: {Counter(r['kind'] for r in rows)}", file=sys.stderr)
    print(f"  by label: {Counter(r['label'] for r in rows)}", file=sys.stderr)
    print(f"  distinct genes: {len(used_genes)}", file=sys.stderr)


if __name__ == "__main__":
    main()
