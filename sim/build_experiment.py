"""Assemble the full experiment count matrix for a pilot design (sim/select_events.py output).

Combines per-gene count matrices (sim/abundance.py) for every design row
into one transcript_id x sample matrix, and writes the flat transcript list
needed to extract a spliced-transcript FASTA subset (via gffread, on the
machine holding the genome/GTF -- not done here). This is deliberately not
whole-genome: only transcripts belonging to a selected gene get simulated,
which keeps FASTA extraction and the polyester run small for the pilot.

Usage:
    python -m sim.build_experiment \
        --design sim/design.tsv --events-dir sim/events \
        --transcript-lengths sim/transcript_lengths.tsv \
        --out-matrix sim/experiment/count_matrix.tsv \
        --out-transcripts sim/experiment/transcript_ids.txt \
        --out-psi-check sim/experiment/psi_check.tsv \
        --seed 0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sim.abundance import (
    AbundanceConfig,
    add_premrna_row,
    build_count_matrix,
    build_count_matrix_baseline,
    empirical_psi,
    load_transcript_lengths,
    premrna_id,
)
from sim.catalogue import build_catalogue


def _load_design(path: str):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True)
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--transcript-lengths", required=True)
    ap.add_argument("--out-matrix", required=True)
    ap.add_argument("--out-transcripts", required=True)
    ap.add_argument("--out-psi-check", required=True)
    ap.add_argument("--gene-mean-depth", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-premrna",
        action="store_true",
        help="Skip noise layer 2 (pre-mRNA background row per gene); layer-1-only mode.",
    )
    ap.add_argument("--premrna-frac-low", type=float, default=0.01)
    ap.add_argument("--premrna-frac-high", type=float, default=0.10)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    config = AbundanceConfig(gene_mean_depth=args.gene_mean_depth)

    design_rows = _load_design(args.design)
    events_by_id = {e.event_id: e for e in build_catalogue(args.events_dir)}
    transcripts_by_gene = load_transcript_lengths(args.transcript_lengths)

    matrices = []
    psi_check_rows = []
    n_missing_gene = 0
    for row in design_rows:
        gene_id = row["gene_id"]
        if gene_id not in transcripts_by_gene:
            n_missing_gene += 1
            continue

        if row["kind"] == "nn_plain":
            df = build_count_matrix_baseline(gene_id, transcripts_by_gene, config, rng)
            obs1 = obs2 = float("nan")
        else:
            event = events_by_id[row["event_id"]]
            psi1, psi2 = float(row["psi1"]), float(row["psi2"])
            df = build_count_matrix(event, transcripts_by_gene, psi1, psi2, config, rng)
            c1_cols = [c for c in df.columns if c.startswith("cond1")]
            c2_cols = [c for c in df.columns if c.startswith("cond2")]
            obs1 = empirical_psi(df, event, c1_cols)
            obs2 = empirical_psi(df, event, c2_cols)

        premrna_frac = float("nan")
        if not args.no_premrna:
            depth_before = df.sum(axis=0)
            df = add_premrna_row(
                df, gene_id, config, rng, args.premrna_frac_low, args.premrna_frac_high
            )
            # recover the drawn fraction from the row actually added, for QC
            premrna_frac = float(
                (df.loc[premrna_id(gene_id)] / depth_before).mean()
            )

        matrices.append(df)
        psi_check_rows.append(
            dict(
                event_id=row["event_id"],
                gene_id=gene_id,
                kind=row["kind"],
                target_psi1=row["psi1"],
                target_psi2=row["psi2"],
                observed_psi1=obs1,
                observed_psi2=obs2,
                premrna_frac=premrna_frac,
            )
        )

    full = pd.concat(matrices).astype(int)
    assert full.index.is_unique, "duplicate transcript_id across selected genes"

    out_matrix = Path(args.out_matrix)
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_matrix, sep="\t")

    with open(args.out_transcripts, "w") as f:
        f.write("\n".join(full.index) + "\n")

    with open(args.out_psi_check, "w", newline="") as f:
        fields = [
            "event_id",
            "gene_id",
            "kind",
            "target_psi1",
            "target_psi2",
            "observed_psi1",
            "observed_psi2",
            "premrna_frac",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(psi_check_rows)

    print(f"genes: {len(design_rows) - n_missing_gene}/{len(design_rows)} (missing_gene={n_missing_gene})", file=sys.stderr)
    print(f"transcripts: {full.shape[0]}, samples: {full.shape[1]}", file=sys.stderr)
    print(f"total simulated fragments: {full.to_numpy().sum():,}", file=sys.stderr)
    print(f"wrote {out_matrix}, {args.out_transcripts}, {args.out_psi_check}", file=sys.stderr)


if __name__ == "__main__":
    main()
