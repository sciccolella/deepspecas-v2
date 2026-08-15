"""Select a balanced pilot simulation design: positives (A3/A5/CSE/RI),
plain NN negatives, hard NN negatives (real events with target dPSI~0), and
novel-junction-only NN negatives (constitutive-looking genes that get an
elevated never-annotated junction in sim/build_experiment.py, so NN isn't
learnable as "no junction noise at all").

One gene is used by at most one selected event, so the per-gene abundance
matrix (sim/abundance.py) never has to reconcile two simulated events in the
same gene. Region windows come straight from the SUPPA event span (positives)
or the gene's genomic span (NN negatives) -- window randomization/jitter is
a later step (plan Phase 2.5), not applied here.

Usage:
    python -m sim.select_events \
        --events-dir sim/events --gene-spans sim/gene_spans.tsv \
        --transcript-lengths sim/transcript_lengths.tsv \
        --n-per-label 60 --n-nn-plain 60 --n-nn-hard 60 --n-nn-novel 60 \
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
from sim.window import sample_window_detail


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


def _load_excluded_genes(paths) -> set:
    """Gene ids from prior batches' design.tsv files (the `gene_id` column) or
    from bare one-id-per-line lists. Both forms are accepted so a batch can be
    excluded straight from its own design output with no extra step."""
    genes = set()
    for path in paths or []:
        with open(path) as f:
            first = f.readline()
            f.seek(0)
            if "\t" in first and "gene_id" in first:
                for row in csv.DictReader(f, delimiter="\t"):
                    genes.add(row["gene_id"])
            else:
                genes.update(line.strip() for line in f if line.strip())
    return genes


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


def select_constitutive_negatives(
    gene_spans, catalogue_genes, transcripts_by_gene, n, used_genes, rng, kind, event_suffix, min_transcripts=2
):
    """Picks genes with no catalogued AS event and >=min_transcripts isoforms
    -- the shared eligibility pool for both plain NN negatives (kind
    "nn_plain") and novel-junction-only NN negatives (kind "nn_novel", see
    select_novel_negatives). The actual novel junction is synthesized later
    in the pipeline (sim/novel_junctions.py, run over design+neighbour
    genes) and applied at elevated abundance in sim/build_experiment.py --
    selection here only needs eligibility, not the variant itself."""
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
                event_id=f"{g};{event_suffix}",
                gene_id=g,
                label="NN",
                kind=kind,
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


def select_plain_negatives(gene_spans, catalogue_genes, transcripts_by_gene, n, used_genes, rng, min_transcripts=2):
    return select_constitutive_negatives(
        gene_spans, catalogue_genes, transcripts_by_gene, n, used_genes, rng, "nn_plain", "NN", min_transcripts
    )


def select_novel_negatives(gene_spans, catalogue_genes, transcripts_by_gene, n, used_genes, rng, min_transcripts=2):
    return select_constitutive_negatives(
        gene_spans, catalogue_genes, transcripts_by_gene, n, used_genes, rng, "nn_novel", "NN_NOVEL", min_transcripts
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--gene-spans", required=True)
    ap.add_argument("--transcript-lengths", required=True)
    ap.add_argument("--chrom-lengths", required=True, help="samtools faidx .fai for the reference")
    ap.add_argument("--n-per-label", type=int, default=60)
    ap.add_argument("--n-nn-plain", type=int, default=60)
    ap.add_argument("--n-nn-hard", type=int, default=60)
    ap.add_argument("--n-nn-novel", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--exclude-genes",
        action="append",
        default=[],
        metavar="FILE",
        help="pre-seed the used-genes set from a prior batch's design.tsv (or a "
        "bare gene-id list), so a scale-up batch selects genes disjoint from it. "
        "Repeatable. Without this, different seeds re-pick the same genes and "
        "create cross-batch gene leakage that GroupKFold cannot see, because "
        "each batch's folds are computed independently.",
    )
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

    used_genes = _load_excluded_genes(args.exclude_genes)
    if used_genes:
        print(f"excluding {len(used_genes)} genes from "
              f"{len(args.exclude_genes)} file(s)", file=sys.stderr)
    n_excluded = len(used_genes)
    rows = []
    rows += select_positives(events, args.n_per_label, used_genes, rng)
    rows += select_hard_negatives(events, args.n_nn_hard, used_genes, rng)
    rows += select_plain_negatives(
        gene_spans, catalogue_genes, transcripts_by_gene, args.n_nn_plain, used_genes, rng
    )
    rows += select_novel_negatives(
        gene_spans, catalogue_genes, transcripts_by_gene, args.n_nn_novel, used_genes, rng
    )

    n_clamped = 0
    for row in rows:
        chrom_len = chrom_lengths[row["chrom"]]
        win_start, win_end, clamped = sample_window_detail(
            row["region_start"], row["region_end"], chrom_len, rng_window
        )
        row["win_start"] = win_start
        row["win_end"] = win_end
        n_clamped += clamped

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
    print(f"  distinct genes: {len(used_genes) - n_excluded}", file=sys.stderr)
    widths = [r["win_end"] - r["win_start"] + 1 for r in rows]
    log10w = np.log10(np.asarray(widths, dtype=float))
    print(f"  window width: median {int(np.median(widths))} log10 mean {log10w.mean():.3f} "
          f"sd {log10w.std(ddof=1):.3f} (benchmark: median 3063, 3.445, 0.702)", file=sys.stderr)
    print(f"  clamped to event span: {n_clamped}/{len(rows)} rows "
          f"({n_clamped / len(rows):.1%}) -- the residual span->width coupling",
          file=sys.stderr)
    if n_excluded:
        selected = {r["gene_id"] for r in rows}
        overlap = selected & _load_excluded_genes(args.exclude_genes)
        print(f"  overlap with excluded genes: {len(overlap)} (must be 0)", file=sys.stderr)
        assert not overlap, f"excluded genes were selected anyway: {sorted(overlap)[:5]}"


if __name__ == "__main__":
    main()
