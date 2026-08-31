#!/usr/bin/env python
"""Stage 8 / Track A2: rMATS x SUPPA2 consensus for the fresh benchmark.

Intersects rMATS's *.MATS.JC.txt and SUPPA2's diffsplice_*.dpsi calls at
p<0.05, per class, matching by genomic locus rather than by exact coordinate
identity -- the two tools use different 0/1-based and intron/exon boundary
conventions for the "same" event, so an exact-coordinate join silently drops
real matches. Instead each event is reduced to a (chr, strand, core_region)
triple and matched by region overlap with a tolerance, mirroring how
sim/window.py already treats event geometry as approximate up to ~100bp.

rMATS uses "chr21"-style names; the GTF/SUPPA2 side (Ensembl) uses "21" --
normalized here, not assumed identical.

Usage: consensus.py <rmats_dir> <suppa_dir> <ioe_dir> <out_tsv>
"""
import re
import sys

import pandas as pd

TOLERANCE = 25  # bp, matches the ~25bp scale of exon/intron boundary convention drift
ALPHA = 0.05

CLASS_MAP = {"A3": "A3SS", "A5": "A5SS", "SE": "SE", "RI": "RI"}
LABEL_MAP = {"A3": "A3", "A5": "A5", "SE": "CSE", "RI": "RI"}


def norm_chr(c):
    return c[3:] if c.startswith("chr") else c


def rmats_regions(rmats_dir, suppa_type):
    rmats_type = CLASS_MAP[suppa_type]
    df = pd.read_csv(f"{rmats_dir}/{rmats_type}.MATS.JC.txt", sep="\t")
    df["chr"] = df["chr"].apply(norm_chr)
    sig = df[df["PValue"] < ALPHA].copy()
    if suppa_type == "SE":
        sig["region_start"] = sig["exonStart_0base"] + 1
        sig["region_end"] = sig["exonEnd"]
    elif suppa_type in ("A3", "A5"):
        sig["region_start"] = sig[["shortES", "longExonStart_0base"]].min(axis=1) + 1
        sig["region_end"] = sig[["shortEE", "longExonEnd"]].max(axis=1)
    elif suppa_type == "RI":
        sig["region_start"] = sig["riExonStart_0base"] + 1
        sig["region_end"] = sig["riExonEnd"]
    return sig[["chr", "strand", "region_start", "region_end", "GeneID", "geneSymbol", "PValue", "FDR", "IncLevelDifference"]]


def parse_event_id(event_id):
    """gene;TYPE:chr:seg1:seg2[:seg3]:strand -- SE/A3/A5 have 2 middle segments,
    each "a-b"; RI has 3, alternating single position and "a-b" range. Every
    numeric token across the middle segments is pooled and reduced to
    (chr, strand, min, max), which is robust to the type-specific segment
    count/shape without needing a separate parser per class."""
    _, rest = event_id.split(";", 1)
    parts = rest.split(":")
    chrom, strand = parts[1], parts[-1]
    nums = []
    for seg in parts[2:-1]:
        nums.extend(int(x) for x in seg.split("-"))
    if not nums:
        return None
    return chrom, strand, min(nums), max(nums)


def suppa_regions(suppa_dir, ioe_dir, suppa_type):
    ioe = pd.read_csv(f"{ioe_dir}/events_{suppa_type}_strict.ioe", sep="\t")
    dpsi = pd.read_csv(f"{suppa_dir}/diffsplice_{suppa_type}.dpsi", sep="\t")
    dpsi = dpsi.reset_index().rename(columns={"index": "event_id"})
    dpsi.columns = ["event_id", "dPSI", "pval"]
    dpsi["event_id"] = dpsi["event_id"].str.strip()
    merged = ioe.merge(dpsi, on="event_id", how="inner")
    merged = merged[pd.to_numeric(merged["pval"], errors="coerce") < ALPHA].copy()

    parsed = merged["event_id"].apply(parse_event_id)
    ok = parsed.notna()
    merged = merged[ok].copy()
    chrom, strand, start, end = zip(*parsed[ok])
    merged["chr"] = chrom
    merged["strand"] = strand
    merged["region_start"] = start
    merged["region_end"] = end
    return merged[["event_id", "gene_id", "chr", "strand", "region_start", "region_end", "dPSI", "pval"]]


def overlap_len(a_start, a_end, b_start, b_end, tol):
    lo = max(a_start - tol, b_start - tol)
    hi = min(a_end + tol, b_end + tol)
    return hi - lo + 1


def match_class(rmats_dir, suppa_dir, ioe_dir, suppa_type):
    """Best-overlap match (not first-hit): for each rMATS row, keep the
    SUPPA2 candidate with the largest overlap, not whichever happened to
    come first in an arbitrary row order -- the latter silently mismatched
    nearby-but-distinct events in multi-isoform loci."""
    rm = rmats_regions(rmats_dir, suppa_type)
    su = suppa_regions(suppa_dir, ioe_dir, suppa_type)
    rows = []
    for _, r in rm.iterrows():
        cand = su[(su["chr"] == r["chr"]) & (su["strand"] == r["strand"])]
        best, best_len = None, 0
        for _, s in cand.iterrows():
            ol = overlap_len(r["region_start"], r["region_end"], s["region_start"], s["region_end"], TOLERANCE)
            if ol > 0 and ol > best_len:
                best, best_len = s, ol
        if best is None:
            continue
        rows.append({
            "class": LABEL_MAP[suppa_type],
            "chr": r["chr"], "strand": r["strand"],
            "region_start": min(r["region_start"], best["region_start"]),
            "region_end": max(r["region_end"], best["region_end"]),
            "gene_id": r["GeneID"], "gene_symbol": r["geneSymbol"],
            "rmats_pvalue": r["PValue"], "rmats_fdr": r["FDR"], "rmats_incleveldiff": r["IncLevelDifference"],
            "suppa_event_id": best["event_id"], "suppa_dpsi": best["dPSI"], "suppa_pvalue": best["pval"],
        })
    return pd.DataFrame(rows)


def clean(df):
    """Two problems found by inspection (not assumed): (1) rMATS reports the
    same underlying event many times over via different flanking-exon/isoform
    choices, all best-matching the same SUPPA2 event -- collapsed here to one
    row per suppa_event_id, keeping the smallest rMATS p-value. (2) rMATS's
    IncLevelDifference and SUPPA2's dPSI disagree in raw sign ~85-95% of the
    time (checked per class: A3 94.5%, A5 94.9%, RI 89.9%, CSE 81.9% -- and
    correlation is strongly *negative*, -0.41 to -0.85, not near zero), i.e.
    the two tools use opposite "which isoform is the alternative one"
    conventions for these event types -- not random mismatched pairs, which
    would show no sign relationship at all. Direction agreement is checked
    against the corrected (flipped) convention and rows that still disagree
    are dropped as low-confidence/likely-mismatched."""
    before = len(df)
    df = df.sort_values("rmats_pvalue").drop_duplicates("suppa_event_id", keep="first")
    deduped = len(df)
    agree = (df["rmats_incleveldiff"] > 0) != (df["suppa_dpsi"] > 0)
    df = df[agree]
    return df, before, deduped


def main():
    rmats_dir, suppa_dir, ioe_dir, out_tsv = sys.argv[1:5]
    frames = [match_class(rmats_dir, suppa_dir, ioe_dir, t) for t in ("A3", "A5", "SE", "RI")]
    out = pd.concat(frames, ignore_index=True)
    cleaned = []
    for cls, sub in out.groupby("class", sort=False):
        c, before, deduped = clean(sub)
        print(f"{cls}: {before} matched -> {deduped} after dedup -> {len(c)} after direction filter")
        cleaned.append(c)
    out = pd.concat(cleaned, ignore_index=True)
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"wrote {out_tsv}: {len(out)} consensus events")
    print(out["class"].value_counts())


if __name__ == "__main__":
    main()
