"""PSI-targeted per-transcript count matrix for one labelled AS event.

Per plans/signal-native-v2.md Phase 2 step 2: for the labelled event, split
abundance between its inclusion/exclusion transcript sets to hit a target PSI
per condition (dPSI is a knob). All other transcripts of the gene get
nonzero abundance at their own random PSI, so every real AS event in the gene
fires simultaneously and only one is labelled -- this is the confounder
design decision.

Counts are drawn as independent negative binomials per transcript per
replicate (mean = gene depth x isoform proportion), which is where the
"negative-binomial overdispersion across replicates" from the plan enters --
polyester's `simulate_experiment_countmat` takes a finished count matrix and
does not add its own biological-replicate noise, so that noise has to be
generated here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from sim.catalogue import Event


@dataclass
class AbundanceConfig:
    n_replicates: int = 3
    gene_mean_depth: float = 2000.0  # mean total fragment count/gene/replicate
    gene_depth_dispersion: float = 4.0  # NB size, gene-level depth across replicates
    transcript_dispersion: float = 10.0  # NB size, per-transcript counts
    dirichlet_alpha: float = 0.5  # baseline isoform-proportion concentration
    #: |dPSI| range for co-occurring (unlabelled) events of the same gene when
    #: build_experiment passes any -- see build_count_matrix's `cooccurring`
    cooc_dpsi_range: tuple = (0.10, 0.60)


def load_transcript_lengths(path: str) -> Dict[str, List[str]]:
    """gene_id -> list of transcript_id, from sim/gtf_transcript_lengths.py output."""
    by_gene: Dict[str, List[str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by_gene.setdefault(row["gene_id"], []).append(row["transcript_id"])
    return by_gene


def sample_baseline_weights(
    transcripts: List[str], rng: np.random.Generator, alpha: float
) -> Dict[str, float]:
    """Dirichlet-distributed isoform proportions summing to 1; low alpha
    favours a dominant isoform, matching typical real-gene usage patterns."""
    if len(transcripts) == 1:
        return {transcripts[0]: 1.0}
    props = rng.dirichlet(np.full(len(transcripts), alpha))
    return dict(zip(transcripts, props))


def targeted_weights(
    baseline: Dict[str, float], event: Event, target_psi: float
) -> Dict[str, float]:
    """Rescale the event's inclusion/exclusion subsets to hit target_psi
    (PSI defined as inclusion-mass / (inclusion+exclusion)-mass, restricted
    to event.total_transcripts), leaving every other transcript's weight --
    and the overall sum-to-1 -- untouched."""
    incl = event.inclusion_transcripts
    excl = event.exclusion_transcripts
    incl_mass = sum(baseline.get(t, 0.0) for t in incl)
    excl_mass = sum(baseline.get(t, 0.0) for t in excl)
    subset_mass = incl_mass + excl_mass
    if subset_mass <= 0:
        return dict(baseline)

    target_incl_mass = target_psi * subset_mass
    target_excl_mass = subset_mass - target_incl_mass
    scale_incl = target_incl_mass / incl_mass if incl_mass > 0 else 0.0
    scale_excl = target_excl_mass / excl_mass if excl_mass > 0 else 0.0

    weights = dict(baseline)
    for t in incl:
        if t in weights:
            weights[t] = baseline[t] * scale_incl
    for t in excl:
        if t in weights:
            weights[t] = baseline[t] * scale_excl
    return weights


def sample_nb(mean: float, dispersion: float, rng: np.random.Generator) -> int:
    """NB parameterized by mean and size (dispersion); var = mean + mean^2/dispersion."""
    if mean <= 0:
        return 0
    p = dispersion / (dispersion + mean)
    return int(rng.negative_binomial(dispersion, p))


def build_count_matrix(
    event: Event,
    transcripts_by_gene: Dict[str, List[str]],
    target_psi_c1: float,
    target_psi_c2: float,
    config: AbundanceConfig,
    rng: np.random.Generator,
    cooccurring: Sequence[Event] = (),
) -> pd.DataFrame:
    """Returns transcript_id x sample count matrix, columns
    cond{1,2}_rep{1..n_replicates}, for every transcript of event.gene_id.

    ``cooccurring`` (plan Stage 3.3) are *other* catalogued events of the same
    gene that should also change between conditions. Without them the design
    says something real windows never say: exactly one event per window is
    differential and every other event in the gene is perfectly
    non-differential. Curated ES windows carry a mean of 2.4 catalogued CSE and
    0.7 catalogued A5 events, and the ES windows this model calls A5 contain
    2.2 catalogued A5 events against 0.7 for those it calls CSE
    (``runs/cse_a3_diagnosis.log``) -- i.e. the model is reading co-occurring
    structure that the simulation always presents as static.

    Order matters and is not incidental: the co-occurring perturbations are
    applied to the baseline *first*, and the labelled event's targeting last,
    so the labelled event still hits ``target_psi_c1``/``c2`` exactly. Applied
    the other way round, an overlapping neighbour would silently move the
    label's own PSI and ``empirical_psi`` would disagree with the design.
    """
    transcripts = transcripts_by_gene[event.gene_id]
    baseline = sample_baseline_weights(transcripts, rng, config.dirichlet_alpha)

    base_by_cond = {1: baseline, 2: baseline}
    if cooccurring:
        b1, b2 = dict(baseline), dict(baseline)
        for other in cooccurring:
            # each neighbour gets its own level and its own effect, drawn the
            # same way a positive is -- these are real events, not noise
            mid = float(rng.uniform(0.25, 0.75))
            dpsi = float(rng.uniform(*config.cooc_dpsi_range)) * float(rng.choice([-1.0, 1.0]))
            p1 = float(np.clip(mid - dpsi / 2, 0.02, 0.98))
            p2 = float(np.clip(mid + dpsi / 2, 0.02, 0.98))
            b1 = targeted_weights(b1, other, p1)
            b2 = targeted_weights(b2, other, p2)
        base_by_cond = {1: b1, 2: b2}

    weights_by_cond = {
        1: targeted_weights(base_by_cond[1], event, target_psi_c1),
        2: targeted_weights(base_by_cond[2], event, target_psi_c2),
    }

    columns = {}
    for cond, weights in weights_by_cond.items():
        for rep in range(1, config.n_replicates + 1):
            total = sample_nb(config.gene_mean_depth, config.gene_depth_dispersion, rng)
            colname = f"cond{cond}_rep{rep}"
            columns[colname] = {
                t: sample_nb(total * weights[t], config.transcript_dispersion, rng)
                for t in transcripts
            }

    df = pd.DataFrame(columns)
    df = df.reindex(transcripts).fillna(0).astype(int)
    df.index.name = "transcript_id"
    return df


def build_count_matrix_baseline(
    gene_id: str,
    transcripts_by_gene: Dict[str, List[str]],
    config: AbundanceConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Plain NN negative: no forced event, both conditions draw from the same
    Dirichlet baseline proportions (natural replicate/condition noise only)."""
    transcripts = transcripts_by_gene[gene_id]
    baseline = sample_baseline_weights(transcripts, rng, config.dirichlet_alpha)

    columns = {}
    for cond in (1, 2):
        for rep in range(1, config.n_replicates + 1):
            total = sample_nb(config.gene_mean_depth, config.gene_depth_dispersion, rng)
            colname = f"cond{cond}_rep{rep}"
            columns[colname] = {
                t: sample_nb(total * baseline[t], config.transcript_dispersion, rng)
                for t in transcripts
            }

    df = pd.DataFrame(columns)
    df = df.reindex(transcripts).fillna(0).astype(int)
    df.index.name = "transcript_id"
    return df


def premrna_id(gene_id: str) -> str:
    return f"PREMRNA_{gene_id}"


def _add_low_abundance_row(
    df: pd.DataFrame,
    row_id: str,
    config: AbundanceConfig,
    rng: np.random.Generator,
    frac_low: float,
    frac_high: float,
) -> pd.DataFrame:
    """Append a background row to a per-gene count matrix at a fraction of
    that sample's total gene depth. The fraction is log-uniform in
    [frac_low, frac_high], drawn once per gene (not per sample/replicate) so
    it represents one fixed background/total ratio for that gene, then each
    sample's count is an independent NB draw around fraction x that sample's
    total -- the same per-replicate noise model as every other transcript in
    the matrix."""
    frac = float(np.exp(rng.uniform(np.log(frac_low), np.log(frac_high))))
    totals = df.sum(axis=0)
    row = {
        col: sample_nb(totals[col] * frac, config.transcript_dispersion, rng)
        for col in df.columns
    }
    df.loc[row_id] = row
    return df


def add_premrna_row(
    df: pd.DataFrame,
    gene_id: str,
    config: AbundanceConfig,
    rng: np.random.Generator,
    frac_low: float = 0.01,
    frac_high: float = 0.10,
) -> pd.DataFrame:
    """Append an unspliced pre-mRNA background row (noise layer 2) to a
    per-gene count matrix -- see `_add_low_abundance_row`."""
    return _add_low_abundance_row(df, premrna_id(gene_id), config, rng, frac_low, frac_high)


def add_novel_junction_row(
    df: pd.DataFrame,
    novel_id: str,
    config: AbundanceConfig,
    rng: np.random.Generator,
    frac_low: float = 0.005,
    frac_high: float = 0.03,
) -> pd.DataFrame:
    """Append a novel-junction background row (noise layer 4) to a per-gene
    count matrix -- see `_add_low_abundance_row`. Default fraction range is
    lower than pre-mRNA's (0.5-3% vs. 1-10%): a never-annotated junction
    should be rarer than a general incomplete-splicing background."""
    return _add_low_abundance_row(df, novel_id, config, rng, frac_low, frac_high)


def empirical_psi(counts_df: pd.DataFrame, event: Event, columns: List[str]) -> float:
    """Count-based PSI proxy over the given columns, for verification only."""
    incl = [t for t in event.inclusion_transcripts if t in counts_df.index]
    excl = [t for t in event.exclusion_transcripts if t in counts_df.index]
    incl_sum = counts_df.loc[incl, columns].to_numpy().sum() if incl else 0
    excl_sum = counts_df.loc[excl, columns].to_numpy().sum() if excl else 0
    denom = incl_sum + excl_sum
    return float(incl_sum / denom) if denom > 0 else float("nan")
