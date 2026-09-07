#!/usr/bin/env python
"""Stage 12 / Track B (B2): automated NN candidates for a model-blind arena.

`consensus.py` emits only *positive* candidates -- loci where rMATS and SUPPA2
both call a significant change. An arena built from those alone cannot measure
NN, which is the weakness every v1 and v2 checkpoint measured on `benchmark_new`
shares (8.9%-13.9% recall), so a comparison run on positives only covers the
classes both models already handle.

This produces the other half, by the same two callers with the criterion
inverted: **a locus both callers see and both call clearly unchanged**, in a
gene that is demonstrably expressed. That is deliberately the *hard* negative --
a real catalogued event with real splicing at it, just no differential usage --
which is what `sim/select_events.py` calls `nn_hard`, and what most of
`benchmark_new`'s curated NN turned out to be (candidates that did not hold up).

## Criteria (fixed before any model saw this arena)

    SUPPA2:  dPSI is measurable (not NaN)  and  p-value > 0.20  and |dPSI| < 0.05
    rMATS:   PValue > 0.20  and  |IncLevelDifference| < 0.05
    both, matched at the same locus by overlap (TOLERANCE, from consensus.py)
    expressed: summed TPM over the event's transcripts >= 1.0 in BOTH conditions
    and not overlapping any positive candidate in consensus.tsv

**Amended once, 2026-09-07, before any model was scored.** The first version
required `p > 0.5` on both sides. That is unsatisfiable on the SUPPA2 side, for
a reason worth recording: **`p == 1.0` in a SUPPA2 `.dpsi` means "not
measurable", not "measured and unchanged"** -- `dPSI` is NaN for exactly those
rows and for no others (verified in all four classes: A3 6684/19347, SE
17501/47833, RI 1963/7322). Among rows that *are* measurable the empirical
p-value never exceeds ~0.485, since it is a permutation p-value and that is its
resolution. So `p > 0.5` selected precisely the unquantifiable events and
intersected with `|dPSI| < 0.05` to give the empty set.

`p > 0.20` keeps a real gap from the 0.05 the positives use (it is roughly the
median of the measurable p-values) while leaving thousands of candidates per
class. The NaN check is explicit rather than implicit, so an unmeasurable locus
can never enter the NN pool by a future threshold change.

The expression floor is what makes "no change" informative rather than "no
data": an unexpressed locus has nothing happening at it for a trivial reason,
and a model that learns to call those NN has learned a coverage detector.

## Known, unquantified caveat

Positives' label noise is measured -- 67.2% of consensus candidates survive
curation (Stage 12, C2). **The NN side's noise level is not measured**, since
no curated set of automatically-derived NN exists to check it against. A locus
both callers call unchanged may still be one a curator would label a real
event. Treat NN accuracy here as indicative and paired-only, exactly like the
positives.

## Why this duplicates a little of consensus.py

`rmats_regions`/`suppa_regions` there hardcode the `p < ALPHA` significance
filter. Inverting them in place would mean touching the script that produced an
already-frozen positive pool; the shared geometry helpers are imported instead
and only the row selection is rewritten here.

Usage:
    make_nn_candidates.py <rmats_dir> <suppa_dir> <ioe_dir> <consensus_tsv> \\
        <tpm_dir> <out_dir> <bam1> <bam2> [--n-total 800] [--seed 0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from benchmark_build.consensus import (
    CLASS_MAP,
    LABEL_MAP,
    TOLERANCE,
    norm_chr,
    overlap_len,
    parse_event_id,
)
from benchmark_build.make_candidate_csvs import CSV_NAME, PAD

#: "clearly not significant" -- a real gap above the 0.05 the positives use,
#: but below ~0.485, which is where a SUPPA2 permutation p-value tops out for
#: an event it could actually measure (see the amendment note above)
NS_PMIN = 0.20
#: and effect sizes indistinguishable from zero on both tools' scales
NS_DPSI_MAX = 0.05
NS_INCDIFF_MAX = 0.05
#: summed TPM over the event's transcripts, required in both conditions
TPM_FLOOR = 1.0


def rmats_unchanged(rmats_dir: str, suppa_type: str) -> pd.DataFrame:
    """rMATS rows the tool calls clearly unchanged. Region conventions are
    copied from consensus.rmats_regions; only the row filter differs."""
    df = pd.read_csv(f"{rmats_dir}/{CLASS_MAP[suppa_type]}.MATS.JC.txt", sep="\t")
    df["chr"] = df["chr"].apply(norm_chr)
    ns = df[
        (pd.to_numeric(df["PValue"], errors="coerce") > NS_PMIN)
        & (pd.to_numeric(df["IncLevelDifference"], errors="coerce").abs() < NS_INCDIFF_MAX)
    ].copy()
    if suppa_type == "SE":
        ns["region_start"] = ns["exonStart_0base"] + 1
        ns["region_end"] = ns["exonEnd"]
    elif suppa_type in ("A3", "A5"):
        ns["region_start"] = ns[["shortES", "longExonStart_0base"]].min(axis=1) + 1
        ns["region_end"] = ns[["shortEE", "longExonEnd"]].max(axis=1)
    elif suppa_type == "RI":
        ns["region_start"] = ns["riExonStart_0base"] + 1
        ns["region_end"] = ns["riExonEnd"]
    return ns[["chr", "strand", "region_start", "region_end", "GeneID", "geneSymbol",
               "PValue", "IncLevelDifference"]]


def suppa_unchanged(suppa_dir: str, ioe_dir: str, suppa_type: str, tpm: pd.DataFrame) -> pd.DataFrame:
    ioe = pd.read_csv(f"{ioe_dir}/events_{suppa_type}_strict.ioe", sep="\t")
    dpsi = pd.read_csv(f"{suppa_dir}/diffsplice_{suppa_type}.dpsi", sep="\t")
    dpsi = dpsi.reset_index().rename(columns={"index": "event_id"})
    dpsi.columns = ["event_id", "dPSI", "pval"]
    dpsi["event_id"] = dpsi["event_id"].str.strip()
    merged = ioe.merge(dpsi, on="event_id", how="inner")
    dpsi_val = pd.to_numeric(merged["dPSI"], errors="coerce")
    merged = merged[
        dpsi_val.notna()  # NaN dPSI == p 1.0 == SUPPA2 could not measure this locus
        & (pd.to_numeric(merged["pval"], errors="coerce") > NS_PMIN)
        & (dpsi_val.abs() < NS_DPSI_MAX)
    ].copy()

    # expression floor: the locus must actually be transcribed in both
    # conditions, or "unchanged" only means "unobserved"
    keep = []
    for _, r in merged.iterrows():
        tx = [t for t in str(r["total_transcripts"]).split(",") if t]
        present = tpm.reindex(tx).dropna()
        # summed over the event's transcripts, per condition -- an earlier
        # version took the min *across transcripts*, which demanded every
        # isoform be expressed and rejected essentially everything
        keep.append(bool(len(present)) and bool((present.sum(axis=0) >= TPM_FLOOR).all()))
    merged = merged[np.array(keep, dtype=bool)].copy()

    parsed = merged["event_id"].apply(parse_event_id)
    ok = parsed.notna()
    merged = merged[ok].copy()
    chrom, strand, start, end = zip(*parsed[ok]) if len(merged) else ((), (), (), ())
    merged["chr"] = list(chrom)
    merged["strand"] = list(strand)
    merged["region_start"] = list(start)
    merged["region_end"] = list(end)
    return merged[["event_id", "gene_id", "chr", "strand", "region_start", "region_end", "dPSI", "pval"]]


def load_tpm(tpm_dir: str) -> pd.DataFrame:
    """Per-transcript TPM, one column per condition (min across replicates, so
    the floor means "expressed in every replicate", not "on average")."""
    out = {}
    for cond in ("cond1", "cond2"):
        df = pd.read_csv(f"{tpm_dir}/{cond}_tpm.tsv", sep="\t", index_col=0)
        out[cond] = df.min(axis=1)
    return pd.DataFrame(out)


def match_unchanged(rmats_dir, suppa_dir, ioe_dir, suppa_type, tpm) -> pd.DataFrame:
    """Same best-overlap join as consensus.match_class, over the unchanged rows."""
    rm = rmats_unchanged(rmats_dir, suppa_type)
    su = suppa_unchanged(suppa_dir, ioe_dir, suppa_type, tpm)
    by_key = {}
    for _, s in su.iterrows():
        by_key.setdefault((s["chr"], s["strand"]), []).append(s)
    rows = []
    for _, r in rm.iterrows():
        best, best_len = None, 0
        for s in by_key.get((r["chr"], r["strand"]), ()):
            ol = overlap_len(r["region_start"], r["region_end"],
                             s["region_start"], s["region_end"], TOLERANCE)
            if ol > 0 and ol > best_len:
                best, best_len = s, ol
        if best is None:
            continue
        rows.append({
            "source_class": LABEL_MAP[suppa_type],
            "chr": r["chr"], "strand": r["strand"],
            "region_start": min(r["region_start"], best["region_start"]),
            "region_end": max(r["region_end"], best["region_end"]),
            "gene_id": r["GeneID"], "gene_symbol": r["geneSymbol"],
            "rmats_pvalue": r["PValue"], "rmats_incleveldiff": r["IncLevelDifference"],
            "suppa_event_id": best["event_id"], "suppa_dpsi": best["dPSI"],
            "suppa_pvalue": best["pval"],
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates("suppa_event_id", keep="first")
    return df


def drop_overlapping_positives(nn: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """An NN window must not sit on a positive candidate: the same locus cannot
    be both, and a model scored on both would be graded twice for one call."""
    if not len(nn):
        return nn
    pos = {}
    for _, p in consensus.iterrows():
        pos.setdefault((str(p["chr"]), p["strand"]), []).append((p["region_start"], p["region_end"]))
    keep = []
    for _, r in nn.iterrows():
        hits = pos.get((str(r["chr"]), r["strand"]), ())
        keep.append(not any(
            overlap_len(r["region_start"], r["region_end"], s, e, TOLERANCE) > 0 for s, e in hits
        ))
    return nn[np.array(keep, dtype=bool)].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rmats_dir")
    ap.add_argument("suppa_dir")
    ap.add_argument("ioe_dir")
    ap.add_argument("consensus_tsv")
    ap.add_argument("tpm_dir")
    ap.add_argument("out_dir")
    ap.add_argument("bam1")
    ap.add_argument("bam2")
    ap.add_argument("--n-total", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    consensus = pd.read_csv(args.consensus_tsv, sep="\t")
    consensus["chr"] = consensus["chr"].astype(str)
    tpm = load_tpm(args.tpm_dir)
    print(f"loaded TPM for {len(tpm)} transcripts")

    pools = {}
    for t in ("A3", "A5", "SE", "RI"):
        m = match_unchanged(args.rmats_dir, args.suppa_dir, args.ioe_dir, t, tpm)
        before = len(m)
        m = drop_overlapping_positives(m, consensus)
        cls = LABEL_MAP[t]
        pools[cls] = m
        print(f"{cls}: {before} unchanged-consensus loci -> {len(m)} after dropping positives")

    # Allocate proportionally to the positive class mix, so NN window widths
    # follow the positives' distribution by construction rather than by luck --
    # width is a known leak on this branch (sim/width_leak_check.py).
    pos_counts = consensus["class"].value_counts()
    total_pos = int(pos_counts.sum())
    rng = np.random.default_rng(args.seed)
    picked = []
    for cls, pool in pools.items():
        want = int(round(args.n_total * pos_counts.get(cls, 0) / total_pos))
        take = min(want, len(pool))
        if take < want:
            print(f"  {cls}: wanted {want}, pool has only {len(pool)}")
        if take:
            idx = rng.choice(len(pool), size=take, replace=False)
            picked.append(pool.iloc[np.sort(idx)])
    nn = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()
    print(f"selected {len(nn)} NN candidates (target {args.n_total})")
    print(nn["source_class"].value_counts())

    # padded with the *source* class's convention, so an NN window drawn at a
    # CSE locus is geometrically a CSE window -- otherwise padding alone
    # separates NN from the positives
    rows, bed = [], []
    for _, r in nn.iterrows():
        pad = PAD[r["source_class"]]
        start = max(1, int(r["region_start"]) - pad)
        end = int(r["region_end"]) + pad
        rows.append(f"{r['chr']}:{start}-{end},{args.bam1},{args.bam2}")
        bed.append((r["chr"], start - 1, end))

    out_csv = f"{args.out_dir}/NN.csv"
    with open(out_csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {out_csv}: {len(rows)} events")

    nn.to_csv(f"{args.out_dir}/nn_candidates.tsv", sep="\t", index=False)
    with open(f"{args.out_dir}/nn_events.bed", "w") as f:
        for chrom, start, end in bed:
            f.write(f"{chrom}\t{start}\t{end}\n")

    widths_nn = np.array([int(b[2] - b[1]) for b in bed], dtype=float)
    pad_pos = consensus["class"].map(PAD)
    widths_pos = (consensus["region_end"] + pad_pos - (consensus["region_start"] - pad_pos)).to_numpy(float)
    qs = [10, 25, 50, 75, 90]
    print("\nwidth check (NN must not be separable from positives by width alone):")
    print("  percentile   positives         NN")
    for q in qs:
        print(f"  {q:>9}   {np.percentile(widths_pos, q):>9.0f}  {np.percentile(widths_nn, q):>9.0f}")


if __name__ == "__main__":
    main()
