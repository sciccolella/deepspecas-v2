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
    add_novel_junction_row,
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


def _load_neighbors(path: str):
    """neighbor_gene_id -> list of design_gene_id it overlaps (sim/gene_neighbors.py
    output), deduplicated: each neighbor gene is simulated once regardless of
    how many design windows it overlaps."""
    by_neighbor: dict[str, list[str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by_neighbor.setdefault(row["neighbor_gene_id"], []).append(row["design_gene_id"])
    return by_neighbor


def _load_novel_junctions(path: str):
    """gene_id -> novel_id, from sim/novel_junctions.py's --out-tsv (one
    novel-junction transcript per gene, at most)."""
    by_gene = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by_gene[row["gene_id"]] = row["novel_id"]
    return by_gene


def build_neighbor_matrix(
    gene_id: str,
    transcripts_by_gene,
    rng_base,
    depth_low: float,
    depth_high: float,
    premrna_frac_low: float,
    premrna_frac_high: float,
    transcript_dispersion: float,
    gene_depth_dispersion: float,
    rng_premrna,
    rng_novel,
    novel_by_gene=None,
    novel_frac_low: float = 0.005,
    novel_frac_high: float = 0.03,
):
    """Background count matrix for one neighbouring/antisense gene (noise
    layer 3): plain baseline expression (no forced event, since the gene
    isn't the labelled locus) at its own log-uniform depth -- independent of
    and generally lower than a design gene's fixed depth, matching real
    between-gene expression heterogeneity -- plus its own pre-mRNA
    background row (real neighbouring genes transcribe pre-mRNA too), plus
    its own novel-junction row (noise layer 4) if one was generated.

    `rng_base`/`rng_premrna`/`rng_novel` are separate per-layer generators
    (see `main`'s RNG-isolation comment) -- neighbour depth/baseline draws
    never share a stream with the premrna or novel-junction draws, so
    toggling those layers on/off doesn't perturb this layer's own draws."""
    depth = float(np.exp(rng_base.uniform(np.log(depth_low), np.log(depth_high))))
    config = AbundanceConfig(
        gene_mean_depth=depth,
        gene_depth_dispersion=gene_depth_dispersion,
        transcript_dispersion=transcript_dispersion,
    )
    df = build_count_matrix_baseline(gene_id, transcripts_by_gene, config, rng_base)
    df = add_premrna_row(df, gene_id, config, rng_premrna, premrna_frac_low, premrna_frac_high)
    if novel_by_gene and gene_id in novel_by_gene:
        df = add_novel_junction_row(
            df, novel_by_gene[gene_id], config, rng_novel, novel_frac_low, novel_frac_high
        )
    return df


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
    ap.add_argument(
        "--neighbors",
        default=None,
        help="sim/gene_neighbors.py output; when given, adds noise layer 3 "
        "(neighbouring/antisense gene background transcription + their own "
        "pre-mRNA rows).",
    )
    ap.add_argument("--neighbor-depth-low", type=float, default=100.0)
    ap.add_argument("--neighbor-depth-high", type=float, default=3000.0)
    ap.add_argument(
        "--novel-junctions",
        default=None,
        help="sim/novel_junctions.py --out-tsv output; when given, adds noise "
        "layer 4 (one never-annotated-junction background row per gene, "
        "applied to design genes and neighbour genes alike).",
    )
    ap.add_argument("--novel-frac-low", type=float, default=0.005)
    ap.add_argument("--novel-frac-high", type=float, default=0.03)
    args = ap.parse_args(argv)

    # RNG isolation: each noise layer gets its own independent Generator,
    # spawned from one SeedSequence rather than sharing a single sequential
    # stream. Previously every layer pulled from the same `rng`, so adding a
    # layer (e.g. novel junctions) inserted extra draws that shifted every
    # subsequent draw for every other layer downstream in the loop order --
    # meaning a layer-N-vs-layer-(N+1) comparison wasn't a clean ceteris
    # paribus ablation (see plans/signal-native-v2-checkpoint.md's noise
    # layer 4 update). With independent streams, turning any single layer
    # on/off leaves every other layer's draws bit-for-bit identical.
    seed_seq = np.random.SeedSequence(args.seed)
    rng_design, rng_premrna, rng_neighbor_base, rng_novel = (
        np.random.default_rng(s) for s in seed_seq.spawn(4)
    )
    config = AbundanceConfig(gene_mean_depth=args.gene_mean_depth)

    design_rows = _load_design(args.design)
    events_by_id = {e.event_id: e for e in build_catalogue(args.events_dir)}
    transcripts_by_gene = load_transcript_lengths(args.transcript_lengths)
    novel_by_gene = _load_novel_junctions(args.novel_junctions) if args.novel_junctions else {}

    matrices = []
    psi_check_rows = []
    n_missing_gene = 0
    for row in design_rows:
        gene_id = row["gene_id"]
        if gene_id not in transcripts_by_gene:
            n_missing_gene += 1
            continue

        if row["kind"] == "nn_plain":
            df = build_count_matrix_baseline(gene_id, transcripts_by_gene, config, rng_design)
            obs1 = obs2 = float("nan")
        else:
            event = events_by_id[row["event_id"]]
            psi1, psi2 = float(row["psi1"]), float(row["psi2"])
            df = build_count_matrix(event, transcripts_by_gene, psi1, psi2, config, rng_design)
            c1_cols = [c for c in df.columns if c.startswith("cond1")]
            c2_cols = [c for c in df.columns if c.startswith("cond2")]
            obs1 = empirical_psi(df, event, c1_cols)
            obs2 = empirical_psi(df, event, c2_cols)

        premrna_frac = float("nan")
        if not args.no_premrna:
            depth_before = df.sum(axis=0)
            df = add_premrna_row(
                df, gene_id, config, rng_premrna, args.premrna_frac_low, args.premrna_frac_high
            )
            # recover the drawn fraction from the row actually added, for QC
            premrna_frac = float(
                (df.loc[premrna_id(gene_id)] / depth_before).mean()
            )

        if gene_id in novel_by_gene:
            df = add_novel_junction_row(
                df, novel_by_gene[gene_id], config, rng_novel, args.novel_frac_low, args.novel_frac_high
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

    n_neighbor_genes = 0
    if args.neighbors:
        design_genes = {row["gene_id"] for row in design_rows}
        neighbors_by_gene = _load_neighbors(args.neighbors)
        for neighbor_gene in neighbors_by_gene:
            if neighbor_gene not in transcripts_by_gene or neighbor_gene in design_genes:
                continue
            matrices.append(
                build_neighbor_matrix(
                    neighbor_gene,
                    transcripts_by_gene,
                    rng_neighbor_base,
                    args.neighbor_depth_low,
                    args.neighbor_depth_high,
                    args.premrna_frac_low,
                    args.premrna_frac_high,
                    config.transcript_dispersion,
                    config.gene_depth_dispersion,
                    rng_premrna,
                    rng_novel,
                    novel_by_gene,
                    args.novel_frac_low,
                    args.novel_frac_high,
                )
            )
            n_neighbor_genes += 1

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
    if args.neighbors:
        print(f"neighbor genes added (layer 3): {n_neighbor_genes}", file=sys.stderr)
    if args.novel_junctions:
        n_novel_rows = sum(1 for idx in full.index if idx.startswith("NOVEL_"))
        print(f"novel-junction rows added (layer 4): {n_novel_rows}", file=sys.stderr)
    print(f"transcripts: {full.shape[0]}, samples: {full.shape[1]}", file=sys.stderr)
    print(f"total simulated fragments: {full.to_numpy().sum():,}", file=sys.stderr)
    print(f"wrote {out_matrix}, {args.out_transcripts}, {args.out_psi_check}", file=sys.stderr)


if __name__ == "__main__":
    main()
