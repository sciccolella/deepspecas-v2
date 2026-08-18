"""Select a balanced pilot simulation design: positives (A3/A5/CSE/RI),
plain NN negatives, hard NN negatives (real events with target dPSI~0), and
novel-junction-only NN negatives (constitutive-looking genes that get an
elevated never-annotated junction in sim/build_experiment.py, so NN isn't
learnable as "no junction noise at all").

One gene is used by at most one selected event, so the per-gene abundance
matrix (sim/abundance.py) never has to reconcile two simulated events in the
same gene. Regions come straight from the SUPPA event span (positives, hard
negatives); constitutive NN negatives get a random sub-interval of their gene
whose length is drawn from the pooled positive event-span distribution, so NN
is not "the wide class" (see `shrink_constitutive_regions`). The extraction
window around each region is then randomized by `sim/window.py`.

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


#: Effect-size and PSI-level distributions, fitted by moments to the curated
#: benchmark's own events (n=125 matched to the SUPPA catalogue, measured with
#: the identical junction logic sim/psi_recovery_check.py applies to the
#: simulation). Fitted quantiles track the observed ones closely:
#:
#:   |dPSI|     q05 .010/.034  q25 .111/.098  q50 .180/.172  q75 .277/.268
#:   PSI level  q05 .083/.075  q25 .373/.362  q50 .668/.666  q75 .893/.891
#:
#: **Caveat, and it is not small.** These two marginals are calibrated *on the
#: benchmark*, so the benchmark is no longer a fully held-out set along the
#: effect-size axis -- results after this change are domain-calibrated, not
#: zero-shot, and should be reported that way. What leaks is two 2-parameter
#: marginals, not per-event labels or structures; the honest mitigation is a
#: held-out split of the benchmark, or an independent dPSI distribution from
#: another dataset, neither of which exists here yet.
DPSI_BETA = (1.781, 7.396)
PSI_LEVEL_BETA = (0.981, 0.620)

#: Null |dPSI| scale, **measured from the simulation's own replicate noise**
#: (``python -m sim.null_dpsi_scale``, n=399 nn_hard events across b1-b5, whose
#: two conditions share a PSI target exactly): median 0.0505, p90 0.133, max
#: 0.279. A half-normal with this sigma has median ``0.6745 * sigma`` = 0.0506,
#: i.e. it reproduces the measured null median by construction.
#:
#: This is the honest counterpart to DPSI_BETA: nothing here comes from the
#: benchmark, so a batch drawn this way stays zero-shot along the effect-size
#: axis. That matters because the alternative -- pulling positives down onto
#: negatives still pinned at exactly 0 -- is what the beta batch did, and it
#: collapsed sim-CV (0.784 -> 0.693) without helping transfer.
NULL_DPSI_SIGMA = 0.075
#: measured maximum; the clip keeps a negative from wandering into positive
#: territory on a fat tail
NULL_DPSI_MAX = 0.28

#: Per-class positive |dPSI| ranges for ``--psi-dist paired`` (plan Stage 3.2).
#: Two measurements set these, neither from the benchmark's marginals:
#:
#: * the null floor above -- a positive at |dPSI| = 0.05 is *by construction*
#:   indistinguishable from a negative, so the positive floor sits at 0.10,
#:   twice the null median;
#: * the pooled beta batch traded per class rather than gaining, and RI was
#:   where it hurt worst (82.4% -> 64.7%): a retained intron at |dPSI| ~ 0.1
#:   carries almost no signal, because the intron's coverage barely moves. So
#:   RI keeps a higher floor while the junction-defined classes go low.
#: *Measured (2026-08-18, runs/stage3.* vs runs/geomwin_b1_refit.*, 960 events
#: each, same protocol):* the whole Stage 3 batch -- paired per-class positives,
#: non-zero nulls and co-occurrence together -- made things WORSE.
#:
#:     sim-CV macro-F1   0.756 -> 0.642
#:     benchmark dev     48/70 -> 42/70 collapsed (68.6% -> 60.0%),
#:                       15 lost / 9 gained, McNemar p = 0.308
#:
#: Three axes moved at once, so this cannot say which one did the damage, and
#: the plan called for exactly that (the multi-axis batch was meant to be the
#: honest test of "realism only pays when several axes move together"). The
#: answer is that it does not pay: the realism premise is now 0 for 3, counting
#: depth invariance and effect-size realism before it. Read alongside the Stage
#: 2 crossover, which found v2's simulation trains v1's representation as well
#: as v1's own data does (104/140 vs 106/140) -- the simulation is not what
#: limits transfer, and making it harder without making it more informative
#: costs sim-CV and transfer together.
PAIRED_DPSI_RANGE = {
    "A3": (0.10, 0.60),
    "A5": (0.10, 0.60),
    "CSE": (0.10, 0.60),
    "RI": (0.25, 0.80),
}


def _null_dpsi(rng: np.random.Generator, sigma: float) -> float:
    """Signed null effect for a hard negative: half-normal magnitude at the
    measured replicate-noise scale, random direction, clipped at the measured
    maximum. ``sigma = 0`` returns exactly 0, i.e. the original design."""
    if sigma <= 0:
        return 0.0
    mag = min(abs(float(rng.normal(0.0, sigma))), NULL_DPSI_MAX)
    return mag * float(rng.choice([-1.0, 1.0]))


def _sample_psi_pair(rng: np.random.Generator, dpsi_lo: float, dpsi_hi: float,
                     dist: str = "uniform", label: str | None = None):
    """(psi1, psi2) for one simulated positive.

    ``dist="uniform"`` is the original draw: ``|dPSI| ~ U(dpsi_lo, dpsi_hi)``
    with the level at ``U(0.25, 0.75)``. At the shipped 0.3-0.8 that floor is
    the problem -- **81.6% of curated benchmark events have |dPSI| below 0.30**,
    so four out of five real events are of a kind the simulation has never
    produced. The level draw is wrong in the same direction: real PSI is
    U-shaped and piles up near 0 and 1 (median 0.668, q05 0.083, q95 0.999),
    while U(0.25, 0.75) is flat and central.

    ``dist="beta"`` draws both from the fitted marginals above.

    ``dist="paired"`` (plan Stage 3.1/3.2) draws per class from
    ``PAIRED_DPSI_RANGE`` and is meant to be used *together with* a non-zero
    null for the hard negatives -- moving positives down while negatives stay
    pinned at exactly 0 is what the beta batch did, and it collapsed the margin
    without moving the boundary. Nothing in it is fitted to the benchmark.
    """
    if dist == "beta":
        dpsi = float(rng.beta(*DPSI_BETA))
        mid = float(rng.beta(*PSI_LEVEL_BETA))
    elif dist == "paired":
        lo, hi = PAIRED_DPSI_RANGE[label] if label in PAIRED_DPSI_RANGE else (dpsi_lo, dpsi_hi)
        dpsi = float(rng.uniform(lo, hi))
        mid = float(rng.uniform(0.25, 0.75))
    else:
        dpsi = float(rng.uniform(dpsi_lo, dpsi_hi))
        mid = float(rng.uniform(0.25, 0.75))
    dpsi *= rng.choice([-1.0, 1.0])
    psi1 = np.clip(mid - dpsi / 2, 0.02, 0.98)
    psi2 = np.clip(mid + dpsi / 2, 0.02, 0.98)
    return float(psi1), float(psi2)


def select_positives(events, n_per_label, used_genes, rng, psi_dist: str = "uniform"):
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
            psi1, psi2 = _sample_psi_pair(rng, 0.3, 0.8, dist=psi_dist, label=label)
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


def select_hard_negatives(events, n, used_genes, rng, null_sigma: float = 0.0):
    """Catalogued AS events used as negatives.

    ``null_sigma = 0`` is the original behaviour: ``psi1 == psi2`` exactly, a
    mathematically perfect non-event. With ``null_sigma > 0`` the two conditions
    differ by ``|dPSI| ~ HalfNormal(null_sigma)`` clipped at NULL_DPSI_MAX --
    the technical floor the simulation already produces between replicates
    (sim/null_dpsi_scale.py). A negative is then "no *meaningful* change", which
    is what a negative means in the deployment setting, rather than "no change
    representable in floating point".
    """
    idx = rng.permutation(len(events))
    rows = []
    for i in idx:
        e = events[i]
        if e.gene_id in used_genes:
            continue
        used_genes.add(e.gene_id)
        mid = float(rng.uniform(0.15, 0.85))
        # ONE draw, split across the two conditions. Drawing inside each of the
        # psi1/psi2 expressions below would take two independent samples, and
        # the realised |dPSI| would be |d1 + d2| / 2 -- sd 0.075/sqrt(2), median
        # 0.036 rather than the 0.0505 that was measured. That bug shipped
        # briefly on 2026-08-17 and was caught by checking the design before the
        # expensive steps ran.
        null_dpsi = _null_dpsi(rng, null_sigma)
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
                psi1=float(np.clip(mid - null_dpsi / 2, 0.02, 0.98)),
                psi2=float(np.clip(mid + null_dpsi / 2, 0.02, 0.98)),
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


def shrink_constitutive_regions(rows, span_pool, rng, kinds=("nn_plain", "nn_novel")):
    """Replace whole-gene regions on constitutive NN rows with a random
    sub-interval whose length is drawn from the pooled *positive* event-span
    distribution.

    Without this, NN is "the wide class": nn_plain/nn_novel take the entire gene
    span (median ~18kb on the 1260-event design, q90 ~145kb) where a real AS
    event spans ~1.4kb, so window width -- which contains the region -- carries
    the label. This matters *more* under the v3 window sampler, not less: the
    window is now the region plus <=100bp, so an unshrunk NN row would be a
    ~18kb window in a set whose positives are ~1.4kb. nn_hard is left alone: it
    already uses a real event span.

    ``rng`` must be a stream of its own: these draws happen after row selection
    and must not perturb it (see the RNG note in `main`). Returns the number of
    rows whose region was shrunk.
    """
    span_pool = np.asarray(span_pool, dtype=np.int64)
    n = 0
    for row in rows:
        if row["kind"] not in kinds:
            continue
        gene_start, gene_end = row["region_start"], row["region_end"]
        gene_len = gene_end - gene_start + 1
        length = min(int(rng.choice(span_pool)), gene_len)
        offset = int(rng.integers(0, gene_len - length + 1))
        row["region_start"] = gene_start + offset
        row["region_end"] = row["region_start"] + length - 1
        n += 1
    return n


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
        "--psi-dist", choices=("uniform", "beta", "paired"), default="uniform",
        help="effect-size draw for positives. 'uniform' is |dPSI| ~ U(0.3, 0.8) "
        "with level U(0.25, 0.75) -- the original, whose floor excludes the 81.6%% "
        "of curated benchmark events below 0.30. 'beta' draws both from marginals "
        "fitted to the benchmark (see DPSI_BETA; note the calibration caveat). "
        "'uniform' remains the DEFAULT despite that: 'beta' was tried and was a "
        "null on transfer (70.0%% vs 72.9%% at 960 events, McNemar p=0.572), "
        "trading A3/A5 gains against ES/IR losses, and it costs sim-CV "
        "(0.693 vs 0.784). The measurement stands; the fix on its own does not. "
        "See plans/signal-native-v2-audit-geometry-results.md (addendum). "
        "'paired' is the Stage 3 draw: per-class ranges (PAIRED_DPSI_RANGE), "
        "floored at twice the measured null, and meant to be used with "
        "--null-dpsi-sigma so positives and negatives move TOGETHER. Nothing "
        "in it is fitted to the benchmark.",
    )
    ap.add_argument(
        "--null-dpsi-sigma", type=float, default=0.0,
        help="hard negatives get |dPSI| ~ HalfNormal(sigma) instead of exactly 0. "
        f"0 keeps the original design; {NULL_DPSI_SIGMA} reproduces the null "
        "median measured from this simulation's own replicate noise "
        "(python -m sim.null_dpsi_scale). The point is that a negative should "
        "mean 'no meaningful change', not 'no change representable in floating "
        "point' -- see plan Stage 3.1.",
    )
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
    rng_negspan = np.random.default_rng(args.seed + 2_000_000)
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
    rows += select_positives(events, args.n_per_label, used_genes, rng, args.psi_dist)
    rows += select_hard_negatives(events, args.n_nn_hard, used_genes, rng,
                              null_sigma=args.null_dpsi_sigma)
    rows += select_plain_negatives(
        gene_spans, catalogue_genes, transcripts_by_gene, args.n_nn_plain, used_genes, rng
    )
    rows += select_novel_negatives(
        gene_spans, catalogue_genes, transcripts_by_gene, args.n_nn_novel, used_genes, rng
    )

    # Constitutive NN rows stop being whole-gene spans before windows are
    # sampled around them (their region is only ever used for window sampling
    # and the extraction CSV -- psi_recovery_check.py skips non-positive rows).
    span_pool = [e.region_end - e.region_start + 1 for e in events]
    n_shrunk = shrink_constitutive_regions(rows, span_pool, rng_negspan)

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
    print(f"  constitutive NN regions shrunk to a sampled event span: {n_shrunk}", file=sys.stderr)
    widths = [r["win_end"] - r["win_start"] + 1 for r in rows]
    spans = [r["region_end"] - r["region_start"] + 1 for r in rows]
    log10w = np.log10(np.asarray(widths, dtype=float))
    frac = np.asarray(spans, dtype=float) / np.asarray(widths, dtype=float)
    print(f"  window width: median {int(np.median(widths))} log10 mean {log10w.mean():.3f} "
          f"sd {log10w.std(ddof=1):.3f} (benchmark: median 3063, 3.445, 0.702)", file=sys.stderr)
    # Under the v3 sampler the window is the event span plus <=100bp per edge,
    # so this ratio should sit just under 1 for every label -- that is the
    # benchmark's geometry (median 1.000), and its per-label spread is the
    # thing the v2 sampler got wrong (median 0.567, RI 0.22 vs CSE 0.94).
    print(f"  event span / window width: median {np.median(frac):.3f} "
          f"(benchmark: 1.000)", file=sys.stderr)
    print(f"  clamped at a chromosome edge: {n_clamped}/{len(rows)} rows "
          f"({n_clamped / len(rows):.1%})", file=sys.stderr)
    if n_excluded:
        selected = {r["gene_id"] for r in rows}
        overlap = selected & _load_excluded_genes(args.exclude_genes)
        print(f"  overlap with excluded genes: {len(overlap)} (must be 0)", file=sys.stderr)
        assert not overlap, f"excluded genes were selected anyway: {sorted(overlap)[:5]}"


if __name__ == "__main__":
    main()
