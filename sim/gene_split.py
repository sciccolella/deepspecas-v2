"""Deterministic gene-level train/val/test split (plan Phase 2 step 7:
"Split by gene. Not by event, not by row.").

The partition is drawn over the *distinct gene set*, not over rows/events,
so a gene with several events (e.g. after scale-up gives up the pilot's
one-event-per-gene constraint) always lands entirely in one split. gene_id
is recovered from event_id by splitting on ";" (event_id is always
"{gene_id};{...}", see sim/select_events.py/sim/catalogue.py), so this works
directly off a labels TSV (ix, event_id, label, kind) without needing
design.tsv.

Usage:
    python -m sim.gene_split --labels sim/experiment/labels.tsv \
        --out sim/experiment/gene_split.tsv --val-frac 0.2 --test-frac 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np


def gene_id_of(event_id: str) -> str:
    return event_id.split(";", 1)[0]


def split_genes(gene_ids, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 0) -> dict:
    """gene_id -> "train"/"val"/"test", via a seeded shuffle over the
    distinct gene set. Safe to call on any subset of genes -- the partition
    only depends on what's passed in, not on selection order."""
    uniq = sorted(set(gene_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_test = int(round(len(uniq) * test_frac))
    n_val = int(round(len(uniq) * val_frac))
    assignment = {}
    for rank, i in enumerate(order):
        gene = uniq[i]
        if rank < n_test:
            assignment[gene] = "test"
        elif rank < n_test + n_val:
            assignment[gene] = "val"
        else:
            assignment[gene] = "train"
    return assignment


def assert_no_gene_leak(rows, gene_key: str = "gene_id", split_key: str = "split") -> None:
    """Split-integrity check from the top-level plan's Verification section:
    raises if any gene_id appears under more than one split value across
    `rows` (dicts or objects supporting __getitem__)."""
    seen: dict = {}
    for row in rows:
        gene, split = row[gene_key], row[split_key]
        if gene in seen and seen[gene] != split:
            raise AssertionError(f"gene {gene!r} spans splits {seen[gene]!r} and {split!r}")
        seen[gene] = split


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="ix,event_id,label,kind TSV (sim/build_extract_csv.py output)")
    ap.add_argument("--out", required=True, help="gene_id,split TSV")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(args.labels) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    gene_ids = [gene_id_of(row["event_id"]) for row in rows]

    assignment = split_genes(gene_ids, args.val_frac, args.test_frac, args.seed)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene_id", "split"])
        for gene, split in sorted(assignment.items()):
            writer.writerow([gene, split])

    row_splits = [dict(gene_id=gene_id_of(r["event_id"]), split=assignment[gene_id_of(r["event_id"])]) for r in rows]
    assert_no_gene_leak(row_splits)

    from collections import Counter

    counts = Counter(assignment.values())
    print(f"wrote {len(assignment)} genes to {args.out}: {dict(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
