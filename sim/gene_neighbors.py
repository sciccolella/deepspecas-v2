"""Find genes overlapping each design row's randomized extraction window
(noise layer 3: neighbouring/antisense genes).

The window is `win_start`/`win_end` from `sim/design.tsv` (plan Phase 2
step 5's per-row window, see `sim/window.py`/`sim/select_events.py`),
optionally padded by a flat `--flank` on top.

**`--flank` defaults to 0, and that default predates window sampler v3.** It
was justified by the v1 sampler, whose window varied 0.1x-10x the event span
and so already reached well beyond the event on its own. Under v3 the window is
the event span plus at most 100 bp per edge, so a design row's window is far
tighter and genuinely neighbouring genes just outside it are no longer picked
up. Neighbour detection may therefore now be under-scoped relative to what
noise layer 3 was calibrated for. The default is left unchanged so existing
batches stay reproducible -- but if layer 3 is revisited, widening `--flank` is
the first thing to try, and the audit (`dsignal.audit_noise compare`, rows
`antisense_frac` / `background_frac`) is where the effect would show up.

For every design row, finds every other gene on the same chromosome whose
span overlaps the (possibly padded) window, excluding the row's own gene and
any gene already used as another row's own design gene (real design genes
are simulated at "real" depth already and must not also be treated as
background). Neighbours are deduplicated globally: a gene near two design
windows is reported/simulated once.

Usage:
    python -m sim.gene_neighbors \
        --design sim/design.tsv --gene-spans sim/gene_spans.tsv \
        --out sim/gene_neighbors.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict


def _load_design(path: str):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_gene_spans(path: str):
    spans = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            spans[row["gene_id"]] = (row["chrom"], row["strand"], int(row["start"]), int(row["end"]))
    return spans


def find_neighbors(design_rows, gene_spans, flank):
    """Returns list of dicts: design_gene_id, neighbor_gene_id, relation,
    overlap_bp. One row per (design_gene, neighbor_gene) pair -- a neighbor
    near several design windows appears once per design gene it overlaps,
    the caller dedups by neighbor_gene_id for simulation purposes."""
    used_genes = {row["gene_id"] for row in design_rows}

    by_chrom = defaultdict(list)
    for gene_id, (chrom, strand, start, end) in gene_spans.items():
        by_chrom[chrom].append((gene_id, strand, start, end))

    rows = []
    for row in design_rows:
        gene_id = row["gene_id"]
        chrom = row["chrom"]
        strand = row["strand"]
        win_start = int(row["win_start"]) - flank
        win_end = int(row["win_end"]) + flank

        for other_id, other_strand, other_start, other_end in by_chrom.get(chrom, []):
            if other_id == gene_id or other_id in used_genes:
                continue
            overlap = min(win_end, other_end) - max(win_start, other_start)
            if overlap <= 0:
                continue
            relation = "same" if other_strand == strand else "antisense"
            rows.append(
                dict(
                    design_gene_id=gene_id,
                    neighbor_gene_id=other_id,
                    relation=relation,
                    overlap_bp=overlap,
                )
            )
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True)
    ap.add_argument("--gene-spans", required=True)
    ap.add_argument("--flank", type=int, default=0, help="extra flat pad beyond win_start/win_end")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    design_rows = _load_design(args.design)
    gene_spans = _load_gene_spans(args.gene_spans)
    rows = find_neighbors(design_rows, gene_spans, args.flank)

    fields = ["design_gene_id", "neighbor_gene_id", "relation", "overlap_bp"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    design_genes_with_neighbor = len({r["design_gene_id"] for r in rows})
    distinct_neighbors = len({r["neighbor_gene_id"] for r in rows})
    n_antisense = sum(1 for r in rows if r["relation"] == "antisense")
    print(
        f"wrote {len(rows)} (design_gene, neighbor) pairs to {args.out}",
        file=sys.stderr,
    )
    print(
        f"  design genes with >=1 neighbor: {design_genes_with_neighbor}/{len(design_rows)}",
        file=sys.stderr,
    )
    print(f"  distinct neighbor genes: {distinct_neighbors}", file=sys.stderr)
    print(f"  same-strand pairs: {len(rows) - n_antisense}, antisense pairs: {n_antisense}", file=sys.stderr)


if __name__ == "__main__":
    main()
