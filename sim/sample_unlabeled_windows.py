"""Stage 22b.1: sample unlabeled real windows for self-supervised pretraining.

Every other real-data builder in this repo (`build_real_pos_extract_csv.py`,
`build_real_nn_extract_csv.py`) anchors on a caller-derived *event*: a region
with a known class, taken from `consensus.tsv`/`nn_candidates.tsv`. SSL
pretraining needs no such anchor -- the pretext tasks (masked junction-feature
reconstruction, augmentation contrastive) never see the `label` column, so
any expressed-gene window from an already-vetted real BAM pair is fair game,
at whatever scale the pretext tasks want. This module builds that pool: one
random sub-interval per draw, gene-disjoint from every eval set, `k_per_gene`
draws per gene.

**Gene exclusion** reuses the exclusion machinery already run for this
cohort's own `real_pos_*`/`real_nn_*` batches (`--exclude-genes`, repeatable,
same TSV-or-bare-list shape as `sim.select_events`'s flag of the same name --
imported from there directly rather than reimplemented). The committed
`exclude_genes_{u2af2,nn_u2af2}.tsv` are the most complete pair available (per
`plans/signal-native-v2-stage20-real-finetune.md`, each cohort's own exclusion
set was built with `--union-with` chaining onto the previous cohort's, so
u2af2's is a superset of aqr's `exclude_genes_stage16.tsv` and prpf4's
`exclude_genes_prpf4.tsv`) and are the recommended default for all three
cohorts -- passing the same union to aqr/prpf4/u2af2 alike costs nothing
(it only ever removes candidate genes) and keeps one auditable exclusion set
across the whole SSL pool instead of three slightly different ones.

**Sub-interval width** is drawn from `--span-pool-csv` (repeatable): one or
more `region,bam1,bam2` CSVs (the same shape `sim.build_real_pos_extract_csv`
emits) whose region widths become the draw pool -- in practice, this cohort's
own `real_pos_<cohort>/extract.csv`, i.e. this cohort's own caller-derived
positive event windows (already padded per class). This is the same "draw a
length from a pooled span distribution, place it at a random offset in the
gene" pattern `sim.select_events.shrink_constitutive_regions` uses for
constitutive NN negatives, just sourced from the cohort's *own* real positive
spans instead of the simulated positive catalogue -- real, local, and not the
held-out curated benchmark (so none of the `DPSI_BETA`-style calibration
caveat applies: nothing here is fitted to `benchmark/`). The v3 window
sampler (`sim.window.sample_window_detail`) is then applied to that
sub-interval exactly as it is to any catalogued event span, adding the usual
label-independent U[0, 100] bp/edge flank.

**Expression filter**: `--min-reads` (>0) keeps only genes where
`bam1.count(gene_span) + bam2.count(gene_span) >= min_reads`, via
`pysam.AlignmentFile.count` -- the same contig-naming convention as
`dsignal.extract` (`gene_spans.tsv` is GTF-derived, so `chrom` has no "chr"
prefix, matching the Ensembl-referenced STAR BAMs). Requires actual BAM
access, so this only runs somewhere the BAMs are mounted (pasteur); pass
`--min-reads 0` to skip it entirely (useful for a dry run against fabricated
paths, e.g. this module's own local tests, where no BAM exists).

Output is written in the exact `region,bam1,bam2` extract-CSV shape and
`ix,event_id,label,kind` labels-TSV shape the rest of the pipeline expects
(`dsignal.extract`, `sim.merge_batches`, `dsignal.dataset.read_labels`).
`label` is always empty (no class is assigned or assignable) and `kind`
must start with `real_` (`dsignal.dataset.read_labels`'s
`is_real = kind.startswith("real_")` domain-label convention) -- the
recommended value is `real_ssl_<cohort>`, distinguishing this pool from the
labelled `real_pos_<cohort>`/`real_nn_<cohort>` ones sharing the same prefix.
`event_id` has no catalogued counterpart; it is synthesized as
`"{gene_id};SSL:{chrom}:{win_start}-{win_end}"` so `sim.gene_split.gene_id_of`
(split on the first `;`) still recovers the gene for `GroupKFold`-style
grouping if this pool is ever folded into a group-aware split.

Usage::

    python -m sim.sample_unlabeled_windows \\
        --gene-spans sim/gene_spans.tsv --chrom-lengths ref/GRCh38.fa.fai \\
        --bam1 aqr_work/align/c1_aqr.bam --bam2 aqr_work/align/c2_aqr.bam \\
        --exclude-genes sim/experiment/exclude_genes_u2af2.tsv \\
        --exclude-genes sim/experiment/exclude_genes_nn_u2af2.tsv \\
        --span-pool-csv sim/experiment/real_pos_aqr/extract.csv \\
        --k-per-gene 4 --min-reads 20 --max-windows 20000 --seed 0 \\
        --kind real_ssl_aqr \\
        --out sim/experiment/real_ssl_aqr/extract.csv \\
        --labels-out sim/experiment/real_ssl_aqr/labels.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import pysam

from dsignal.extract import parse_region
from dsignal.predict import read_prediction_csv
from sim.select_events import _load_chrom_lengths, _load_excluded_genes, _load_gene_spans
from sim.window import sample_window_detail


def load_span_pool(csv_paths) -> list:
    """Region widths (inclusive bp) pooled across one or more extract-CSVs."""
    widths = []
    for path in csv_paths:
        for region, _bam1, _bam2 in read_prediction_csv(path):
            _chrom, start, end = parse_region(region)
            widths.append(end - start + 1)
    if not widths:
        raise ValueError(f"no regions found in {csv_paths!r}")
    return widths


def expressed_genes(bam1_path: str, bam2_path: str, genes: dict, min_reads: int) -> set:
    """``genes``: gene_id -> (chrom, strand, start, end), 1-based inclusive
    (as `sim.select_events._load_gene_spans` returns). Keeps a gene iff the
    combined read count over both conditions across its whole span is
    >= min_reads. `pysam.count` is 0-based half-open, hence `start - 1`."""
    kept = set()
    with pysam.AlignmentFile(bam1_path, "rb") as bam1, pysam.AlignmentFile(bam2_path, "rb") as bam2:
        contigs = set(bam1.references)
        for gene_id, (chrom, _strand, start, end) in genes.items():
            if chrom not in contigs:
                continue
            n = bam1.count(contig=chrom, start=start - 1, stop=end) + \
                bam2.count(contig=chrom, start=start - 1, stop=end)
            if n >= min_reads:
                kept.add(gene_id)
    return kept


def sample_gene_windows(gene_start: int, gene_end: int, chrom_len: int, span_pool,
                         k: int, rng: np.random.Generator) -> list:
    """k independent (win_start, win_end) draws: a random sub-interval of
    [gene_start, gene_end] whose width comes from span_pool (clamped to the
    gene's own length), then the v3 window sampler's flank on top."""
    gene_len = gene_end - gene_start + 1
    out = []
    for _ in range(k):
        width = max(1, min(int(rng.choice(span_pool)), gene_len))
        offset = int(rng.integers(0, gene_len - width + 1))
        sub_start = gene_start + offset
        sub_end = sub_start + width - 1
        win_start, win_end, _clamped = sample_window_detail(sub_start, sub_end, chrom_len, rng)
        out.append((win_start, win_end))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-spans", required=True)
    ap.add_argument("--chrom-lengths", required=True, help="samtools faidx .fai for the reference")
    ap.add_argument("--bam1", required=True)
    ap.add_argument("--bam2", required=True)
    ap.add_argument("--exclude-genes", action="append", default=[], metavar="FILE",
                     help="repeatable; TSV with a gene_id column or a bare one-id-per-line list "
                     "(same loader as sim.select_events --exclude-genes)")
    ap.add_argument("--span-pool-csv", nargs="+", required=True,
                     help="one or more region,bam1,bam2 CSVs whose region widths seed the "
                     "sub-interval width draw (recommended: this cohort's own real_pos_<cohort>/extract.csv)")
    ap.add_argument("--k-per-gene", type=int, default=4)
    ap.add_argument("--min-reads", type=int, default=20,
                     help="combined bam1+bam2 read count required over a gene's span to keep it; "
                     "0 skips the BAM expression filter entirely (needs no BAM access -- for a dry run)")
    ap.add_argument("--max-windows", type=int, default=None,
                     help="stop once this many windows are written (genes are shuffled first, so "
                     "this is an unbiased subsample, not a prefix of gene-id order)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="real_ssl_aqr",
                     help="labels.tsv `kind` value (default %(default)s); must start with 'real_'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-out", required=True)
    args = ap.parse_args(argv)
    if not args.kind.startswith("real_"):
        ap.error("--kind must start with 'real_'")

    rng = np.random.default_rng(args.seed)

    gene_spans = _load_gene_spans(args.gene_spans)
    chrom_lengths = _load_chrom_lengths(args.chrom_lengths)
    exclude = _load_excluded_genes(args.exclude_genes)
    span_pool = load_span_pool(args.span_pool_csv)
    print(f"{len(gene_spans)} genes, {len(exclude)} excluded, "
          f"span pool n={len(span_pool)} (median {int(np.median(span_pool))} bp)", file=sys.stderr)

    candidates = {g: v for g, v in gene_spans.items()
                  if g not in exclude and v[0] in chrom_lengths}
    n_dropped_chrom = len(gene_spans) - len(exclude) - len(candidates)
    if n_dropped_chrom:
        print(f"{n_dropped_chrom} non-excluded genes dropped: chrom not in --chrom-lengths", file=sys.stderr)

    if args.min_reads > 0:
        n_before = len(candidates)
        keep = expressed_genes(args.bam1, args.bam2, candidates, args.min_reads)
        candidates = {g: v for g, v in candidates.items() if g in keep}
        print(f"{len(candidates)}/{n_before} genes pass the "
              f">= {args.min_reads}-read expression filter", file=sys.stderr)
    else:
        print("--min-reads 0: expression filter skipped", file=sys.stderr)

    gene_ids = list(candidates)
    rng.shuffle(gene_ids)

    rows = []
    for gene_id in gene_ids:
        if args.max_windows is not None and len(rows) >= args.max_windows:
            break
        chrom, _strand, start, end = candidates[gene_id]
        chrom_len = chrom_lengths[chrom]
        windows = sample_gene_windows(start, end, chrom_len, span_pool, args.k_per_gene, rng)
        for win_start, win_end in windows:
            if args.max_windows is not None and len(rows) >= args.max_windows:
                break
            event_id = f"{gene_id};SSL:{chrom}:{win_start}-{win_end}"
            region = f"{chrom}:{win_start}-{win_end}"
            rows.append((event_id, region))

    print(f"{len(gene_ids)} genes considered, {len(rows)} windows sampled "
          f"(k_per_gene={args.k_per_gene})", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        for _event_id, region in rows:
            writer.writerow([region, args.bam1, args.bam2])

    with open(args.labels_out, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ix", "event_id", "label", "kind"])
        for ix, (event_id, _region) in enumerate(rows):
            writer.writerow([f"{ix:05d}", event_id, "", args.kind])

    print(f"wrote {len(rows)} rows to {args.out} and {args.labels_out}")


if __name__ == "__main__":
    main()
