"""Noise-calibration audit: is the simulation's injected noise *the right
amount*, or merely present?

Track A of ``plans/signal-native-v2-audit-geometry.md``. Every noise layer in
``sim/`` was validated by showing it degraded the baseline model, which was read
as proof it added signal-relevant noise. Nobody ever checked whether the
*amount* matches the real data -- and step 0 of the scale-up plan found the
first counter-instance pointing the other way (simulated CSE events carry ~4x
the skip-junction support curated ES events do). If the simulation is noisier
than reality along the axes we added and cleaner along the ones we never varied,
the distribution shift is larger than v1's cruder simulator had, which is what
the transfer numbers say.

So: measure the same quantities on both sides with the same code.

``measure``  one row per window, from a ``region,bam1,bam2`` CSV -- the format
             the benchmark CSVs and every batch's ``extract.csv`` already use.
``compare``  two such TSVs in, a sim-vs-real table out, with the quantities off
             by more than 2x called out by name.

Every quantity is **annotation-free and window-local**, so the identical code
path runs on ``benchmark/c1.bam`` and on a simulated batch's BAM. Structure
metrics reuse ``dsignal.features.normalize_record`` and the three motif helpers
in ``dsignal/diagnose_cse_a3.py``; the extractor itself supplies coverage.

Two traps, both real and both silent if you get them wrong:

1. **``dsignal/extract.py`` masks off STAR's ``+20`` annotated-junction bit**
   (``motif_raw % 20``), so the novel-junction rate -- the whole point of noise
   layer 4 -- is *not* recoverable from the stored ``.npz``. It needs its own
   BAM pass, which is what ``_raw_bam_stats`` is for. Both sides carry the bit:
   the benchmark BAM's reads show ``jM=[0,22]``, and the simulated BAMs were
   aligned against a STAR index built from ``Homo_sapiens.GRCh38.99.gtf``.
2. **``benchmark/c1.bam`` is a ``samtools view -L events.bed`` subset of a
   three-run merge** (SRR1513329/30/31). Reads that fall outside the curated
   windows are simply absent, so ``background_frac`` and ``antisense_frac`` are
   a *lower bound* on the real side -- ``compare`` labels them as such instead
   of pretending the ratio means what the others do. The merge also means
   duplicates must be keyed **within a source library**, or three runs'
   independent reads at the same coordinate get counted as PCR duplicates of
   each other and the real rate comes out inflated.

Usage:
    python -m dsignal.audit_noise measure --csv benchmark/{A3,A5,ES,IR}.csv \\
        --out runs/noise_audit_real.tsv
    python -m dsignal.audit_noise measure --csv sim/batches/b1/extract.csv \\
        --labels sim/batches/b1/labels.tsv --subsample 200 \\
        --out runs/noise_audit_sim_b1.tsv
    python -m dsignal.audit_noise compare --real runs/noise_audit_real.tsv \\
        --sim runs/noise_audit_sim.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from os.path import basename, splitext

import numpy as np
import pysam

from dsignal import extract as dsextract
from dsignal import features as dsfeatures
from dsignal.diagnose_cse_a3 import (
    DEFAULT_MIN_COUNT,
    DEFAULT_MIN_MINOR_FRAC,
    _best_se_triplet,
    _best_shared_pair,
    _supported,
)


def read_prediction_csv(path: str):
    """``region,bam1,bam2``, no header. Duplicated from ``dsignal.predict``
    rather than imported: the sim side of this audit runs on pasteur, whose
    ``DSpoly2v2`` env has no torch, and importing predict.py would drag it in
    for three lines of csv reading."""
    with open(path) as f:
        return [(r[0], r[1], r[2]) for r in csv.reader(f) if r]
#: bp of exonic context on each side of a junction, for the intron/exon
#: coverage ratio and for "is this window base near any spliced structure".
#: Matches dsignal.features.DEFAULT_FLANK so the audit and the model agree on
#: what "flank" means.
DEFAULT_FLANK = dsfeatures.DEFAULT_FLANK

#: an intron shorter than this gives a median-coverage ratio dominated by read
#: overhang at the boundaries rather than by genuine intronic signal
MIN_INTRON_FOR_COV = 100

#: STAR sets bit +20 on jM for junctions present in the index's GTF. The
#: extractor masks it off; the audit is the one place that needs it.
_JM_ANNOTATED_BIT = 20

def _library_of(query_name: str) -> str:
    """Source library of a read, for scoping duplicate detection.

    benchmark/c1.bam merges three SRA runs and names reads
    ``SRR1513329.6348221``, so three runs' independent fragments can land on
    the same coordinate without being PCR duplicates of each other -- the
    prefix separates them.

    Simulated reads are named ``read2270765``, with **no dot**. Splitting those
    on "." returns the whole (unique) name, which made every simulated read its
    own library and drove the measured duplicate rate to ~0 by construction.
    A name with no dot therefore means one library, not one library per read.
    """
    prefix, sep, _ = query_name.partition(".")
    return prefix if sep else ""


# --------------------------------------------------------------------------
# the BAM pass extract.py cannot give us: annotated-junction bit + duplicates
# --------------------------------------------------------------------------


def _raw_bam_stats(bam: pysam.AlignmentFile, region: tuple, filters: dsextract.ReadFilters) -> dict:
    """Per-junction annotated/novel status from the *unmasked* jM tag, plus the
    coordinate-duplicate rate over primary alignments.

    Read filtering mirrors ``dsignal.extract.extract_condition`` (same anchor
    and NH rules) so the junction set counted here is the same one the model
    sees -- an audit measured over a different read set would not be comparable
    to anything.
    """
    chrom, start, end = region

    # (start, end) -> {"count": n, "annotated": bool | None}
    junctions: dict = {}
    n_no_tag = 0

    seen: set = set()
    n_primary = 0
    n_dup = 0

    for read in bam.fetch(chrom, start, end):
        cigartuples = read.cigartuples
        if cigartuples is None or read.is_unmapped:
            continue
        try:
            nh = int(read.get_tag("NH"))
        except KeyError:
            nh = 1
        if not filters.keep_read(read, nh):
            continue

        # --- duplicates: primary alignments only, keyed within a library. A
        # multimapping read contributes NH alignment records; counting the
        # secondaries would report the multimapping rate as a duplicate rate.
        if not read.is_secondary and not read.is_supplementary:
            n_primary += 1
            key = (
                _library_of(read.query_name),
                read.reference_start,
                read.cigarstring,
                read.next_reference_id,
                read.next_reference_start,
                read.is_read1,
                read.is_reverse,
            )
            if key in seen:
                n_dup += 1
            else:
                seen.add(key)

        # --- junctions, with the annotated bit intact
        raw = _tag_junctions_raw(read)
        if raw is None:
            if _SKIP_OP_IN(cigartuples):
                n_no_tag += 1
            continue
        for i, (s1b, e1b, motif_raw) in enumerate(raw):
            if not _anchor_ok(cigartuples, i, filters.min_anchor):
                continue
            rec = junctions.setdefault((s1b, e1b), {"count": 0, "annotated": None})
            rec["count"] += 1
            if motif_raw >= 0:
                rec["annotated"] = motif_raw >= _JM_ANNOTATED_BIT

    return {
        "junctions": junctions,
        "n_reads_no_tag": n_no_tag,
        "n_primary": n_primary,
        "n_dup": n_dup,
    }


def _SKIP_OP_IN(cigartuples) -> bool:
    return any(op == dsextract._SKIP_OP for op, _ in cigartuples)


def _tag_junctions_raw(read: pysam.AlignedSegment):
    """``dsignal.extract._tag_junctions`` without the ``% 20`` mask. Returns
    None when the read carries no jI/jM (not a STAR BAM, or tags stripped)."""
    try:
        jI = read.get_tag("jI")
        jM = read.get_tag("jM")
    except KeyError:
        return None
    if len(jI) < 2 or jI[0] == -1:
        return []
    return [
        (int(jI[i]), int(jI[i + 1]), int(jM[i // 2]))
        for i in range(0, len(jI), 2)
    ]


def _anchor_ok(cigartuples, n_idx: int, min_anchor: int) -> bool:
    """Same per-occurrence anchor rule ``extract_condition`` applies, indexed by
    which N-op this is within the read."""
    seen = 0
    for i, (op, _) in enumerate(cigartuples):
        if op != dsextract._SKIP_OP:
            continue
        if seen == n_idx:
            left = cigartuples[i - 1][1] if i > 0 and cigartuples[i - 1][0] in dsextract._COV_OPS else 0
            right = (
                cigartuples[i + 1][1]
                if i + 1 < len(cigartuples) and cigartuples[i + 1][0] in dsextract._COV_OPS
                else 0
            )
            return min(left, right) >= min_anchor
        seen += 1
    return False


def _novel_stats(raw_1: dict, raw_2: dict, min_count: float) -> dict:
    """Novel-junction rate and read share over the junctions that clear
    ``min_count`` in either condition -- the same support bar the structural
    metrics use, so "rate" means the same population throughout."""
    pooled: dict = defaultdict(lambda: {"c1": 0, "c2": 0, "annotated": None})
    for raw, slot in ((raw_1, "c1"), (raw_2, "c2")):
        for key, rec in raw["junctions"].items():
            p = pooled[key]
            p[slot] += rec["count"]
            if p["annotated"] is None:
                p["annotated"] = rec["annotated"]

    n_sup = n_novel = 0
    reads_sup = reads_novel = 0.0
    n_unknown = 0
    for rec in pooled.values():
        if max(rec["c1"], rec["c2"]) < min_count:
            continue
        n_sup += 1
        reads = rec["c1"] + rec["c2"]
        reads_sup += reads
        if rec["annotated"] is None:
            n_unknown += 1
        elif not rec["annotated"]:
            n_novel += 1
            reads_novel += reads

    nan = float("nan")
    # every supported junction lacking the tag makes both rates unknowable;
    # returning 0.0 would read as "no novel junctions", the opposite claim
    if n_sup == 0 or n_unknown == n_sup:
        return {"novel_junction_rate": nan, "novel_junction_read_frac": nan,
                "n_supported_tagged": 0}
    return {
        "novel_junction_rate": n_novel / n_sup,
        "novel_junction_read_frac": reads_novel / reads_sup if reads_sup else nan,
        "n_supported_tagged": n_sup - n_unknown,
    }


# --------------------------------------------------------------------------
# coverage-shape metrics off the normalized record
# --------------------------------------------------------------------------


def _intron_cov_frac(norm, idx, flank: int) -> tuple:
    """Median over supported junctions of (mean intronic coverage) / (mean
    coverage on the flanking exonic anchors) -- the layer-2 pre-mRNA readout,
    expressed the way it is injected (a fraction of the spliced abundance).

    Both terms are **means**. The first version used the intron's *median*,
    which measures 0 for essentially every curated benchmark window: real depth
    there is ~15x, so most bases of a long intron are genuinely uncovered even
    when pre-mRNA is present. A mean keeps the sparse intronic reads that are
    the actual quantity layer 2 injects.

    Junctions whose intron is short or reaches outside the window are skipped:
    the ratio there is dominated by read overhang and window truncation, not by
    intronic transcription.
    """
    cov = norm.cov_1 + norm.cov_2
    n = len(cov)
    ratios = []
    for i in idx:
        s, e = int(norm.s_rel[i]), int(norm.e_rel[i])
        if s < flank or e >= n - flank or (e - s + 1) < MIN_INTRON_FOR_COV:
            continue
        interior = cov[s:e + 1]
        flanks = np.concatenate((cov[s - flank:s], cov[e + 1:e + 1 + flank]))
        denom = float(flanks.mean())
        if denom <= 0:
            continue
        ratios.append(float(interior.mean()) / denom)
    if not ratios:
        return float("nan"), 0
    return float(np.median(ratios)), len(ratios)


def _background_frac(norm, idx, flank: int) -> float:
    """Share of window coverage sitting on bases explained by no supported
    junction: neither inside one's intron nor within ``flank`` bp of its
    exonic anchors. On the real side this is a lower bound -- see the module
    docstring's note on the ``-L`` subsetting.
    """
    cov = norm.cov_1 + norm.cov_2
    n = len(cov)
    if n == 0 or len(idx) == 0:
        return float("nan")
    explained = np.zeros(n, dtype=bool)
    for i in idx:
        s, e = int(norm.s_rel[i]), int(norm.e_rel[i])
        lo, hi = max(s - flank, 0), min(e + flank + 1, n)
        if hi > lo:
            explained[lo:hi] = True
    total = float(cov.sum())
    if total <= 0:
        return float("nan")
    return float(cov[~explained].sum()) / total


def _antisense_frac(record: dict) -> float:
    """NH-weighted share of junction reads whose motif-implied strand disagrees
    with the window's strand vote -- the readout of noise layer 3's antisense
    and overlapping neighbours."""
    vote = str(record["strand"])
    if vote not in ("+", "-"):
        return float("nan")
    agree = disagree = 0.0
    for arr in (record["junctions_1"], record["junctions_2"]):
        for row in arr:
            s = row["strand"].decode()
            if s == vote:
                agree += float(row["nh_weighted_count"])
            elif s in ("+", "-"):
                disagree += float(row["nh_weighted_count"])
    tot = agree + disagree
    return disagree / tot if tot > 0 else float("nan")


def _flanks(norm, idx) -> dict:
    """Where the spliced structure sits inside the window (Track B's input).

    ``left_flank`` / ``right_flank`` are the gaps from the window edges to the
    outermost supported junction, in the canonicalized (transcription-direction)
    orientation; ``struct_span_frac`` is how much of the window the structure
    spans. The benchmark's window *is* the curated event span, so these are the
    empirical padding a curator leaves -- which is what ``sim/window.py`` should
    be drawing from.
    """
    nan = float("nan")
    if len(idx) == 0 or norm.n == 0:
        return {"left_flank": nan, "right_flank": nan, "struct_span_frac": nan}
    s = int(np.min(norm.s_rel[idx]))
    e = int(np.max(norm.e_rel[idx]))
    return {
        "left_flank": float(s),
        "right_flank": float(norm.n - 1 - e),
        "struct_span_frac": float(min(e, norm.n - 1) - max(s, 0) + 1) / norm.n,
    }


# --------------------------------------------------------------------------


def audit_event(region: str, bam1: str, bam2: str, filters: dsextract.ReadFilters,
                min_count: float, min_minor_frac: float, flank: int) -> dict:
    reg = dsextract.parse_region(region)
    record = dsextract.extract_event(reg, bam1, bam2, filters=filters)
    norm = dsfeatures.normalize_record(record)
    idx = _supported(norm, min_count)

    with pysam.AlignmentFile(bam1, "rb") as b1, pysam.AlignmentFile(bam2, "rb") as b2:
        raw_1 = _raw_bam_stats(b1, reg, filters)
        raw_2 = _raw_bam_stats(b2, reg, filters)

    cov = (norm.cov_1 + norm.cov_2) / 2.0
    pooled = norm.count_1 + norm.count_2
    n_primary = raw_1["n_primary"] + raw_2["n_primary"]
    n_dup = raw_1["n_dup"] + raw_2["n_dup"]

    intron_frac, n_intron = _intron_cov_frac(norm, idx, flank)
    row = {
        "region": region,
        "width": reg[2] - reg[1],
        "strand": norm.strand,
        "n_junctions": int(len(norm.s_rel)),
        "n_supported": int(len(idx)),
        "cov_mean": float(cov.mean()) if norm.n else 0.0,
        "cov_median": float(np.median(cov)) if norm.n else 0.0,
        "top_junction_count": float(pooled.max()) if pooled.size else 0.0,
        "intron_cov_frac": intron_frac,
        "n_intron_measured": n_intron,
        "background_frac": _background_frac(norm, idx, flank),
        "antisense_frac": _antisense_frac(record),
        "dup_rate": (n_dup / n_primary) if n_primary else float("nan"),
        "n_primary": n_primary,
    }
    row.update(_novel_stats(raw_1, raw_2, min_count))
    row.update(_flanks(norm, idx))

    for name, best in (
        ("a3", _best_shared_pair(norm, idx, "s_rel", min_minor_frac)),
        ("a5", _best_shared_pair(norm, idx, "e_rel", min_minor_frac)),
        ("se", _best_se_triplet(norm, idx, min_minor_frac)),
    ):
        row[f"has_{name}"] = int(best is not None)
        row[f"{name}_minor_frac"] = best["minor_frac"] if best else float("nan")
        row[f"{name}_minor_count"] = best["minor_count"] if best else float("nan")
        row[f"{name}_delta"] = best["delta"] if best else float("nan")
    return row


# --------------------------------------------------------------------------
# what `compare` reports, and how each quantity is summarized
# --------------------------------------------------------------------------

#: (column, kind, sim knob it audits). ``kind`` picks the summary: "rate"
#: quantities are shares whose median is often 0, so they are compared on the
#: mean; "level" quantities are compared on the median, which is what the
#: heavy-tailed depth distributions call for.
METRICS = [
    ("intron_cov_frac", "level", "layer 2: --premrna-frac-low/high 0.01/0.10"),
    ("novel_junction_rate", "rate", "layer 4: --novel-frac 0.005-0.03, --novel-nn-frac-high 0.20"),
    ("novel_junction_read_frac", "rate", "layer 4 (read share)"),
    ("dup_rate", "rate", "layer 5: PCR_RATE=0.1 PCR_LAMBDA=1"),
    ("background_frac", "rate", "layer 3: --neighbor-depth 100-3000  [LOWER BOUND on real]"),
    ("antisense_frac", "rate", "layer 3, antisense component  [LOWER BOUND on real]"),
    ("cov_mean", "level", "--gene-mean-depth 2000"),
    ("cov_median", "level", "--gene-mean-depth 2000"),
    ("top_junction_count", "level", "--gene-mean-depth 2000"),
    ("n_supported", "level", "-"),
    ("a3_minor_count", "level", "_sample_psi_pair: |dPSI| ~ U(0.3, 0.8)"),
    ("a3_delta", "level", "_sample_psi_pair"),
    ("se_minor_count", "level", "_sample_psi_pair"),
    ("se_delta", "level", "_sample_psi_pair"),
    ("left_flank", "level", "sim/window.py (Track B)"),
    ("right_flank", "level", "sim/window.py (Track B)"),
    ("struct_span_frac", "rate", "sim/window.py (Track B)"),
]

#: quantities the real side under-measures because benchmark/c1.bam only keeps
#: reads overlapping events.bed; their ratios are directional, not calibrated
LOWER_BOUND_ON_REAL = {"background_frac", "antisense_frac"}

#: A coordinate-duplicate rate rises with depth on its own -- there are finitely
#: many start positions, so a deep window collides by saturation and not by PCR.
#: (Measured: 0/103 primaries in one benchmark window, 2073/12513 = 16.6% in a
#: deep one.) Read `dup_rate` next to `cov_mean`, never alone.
DEPTH_CONFOUNDED = {"dup_rate"}


def _summarize(vals: np.ndarray, kind: str) -> float:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return float("nan")
    return float(v.mean()) if kind == "rate" else float(np.median(v))


def compare(real_rows, sim_rows, threshold: float = 2.0) -> None:
    print(f"\n=== noise calibration: simulation vs curated benchmark ===")
    print(f"real n={len(real_rows)}   sim n={len(sim_rows)}")
    print("\n'rate' quantities are compared on the mean (their median is often "
          "0), 'level' quantities on the median (heavy tails).\n")
    print(f"{'quantity':<26} {'kind':<6} {'real':>10} {'sim':>10} {'sim/real':>9}  audits")
    offenders = []
    for name, kind, knob in METRICS:
        r = _summarize(np.array([row.get(name, np.nan) for row in real_rows], dtype=float), kind)
        s = _summarize(np.array([row.get(name, np.nan) for row in sim_rows], dtype=float), kind)
        ratio = s / r if (np.isfinite(r) and r != 0 and np.isfinite(s)) else float("nan")
        flag = ""
        if np.isfinite(ratio) and (ratio >= threshold or ratio <= 1 / threshold):
            flag = " <<<" if name not in LOWER_BOUND_ON_REAL else " (?)"
            offenders.append((name, ratio, knob))
        print(f"{name:<26} {kind:<6} {r:>10.4g} {s:>10.4g} {ratio:>9.2f}  {knob}{flag}")

    print(f"\n=== off by more than {threshold}x ===")
    if not offenders:
        print("  none")
    for name, ratio, knob in sorted(offenders, key=lambda t: -abs(np.log2(t[1]))):
        note = ""
        if name in LOWER_BOUND_ON_REAL:
            note = "  [real side is a lower bound -- direction only]"
        elif name in DEPTH_CONFOUNDED:
            note = "  [rises with depth on its own -- read next to cov_mean]"
        direction = "simulation OVER-injects" if ratio > 1 else "simulation UNDER-injects"
        print(f"  {name:<26} {ratio:>6.2f}x  {direction}  ({knob}){note}")

    # per-class, for the quantities where the label is the point
    if any("label" in row for row in sim_rows):
        print("\n=== per-label medians, simulation (the sim-side breakdown) ===")
        _by_group(sim_rows, "label")
    if any("csv" in row for row in real_rows):
        print("\n=== per-CSV medians, curated benchmark ===")
        _by_group(real_rows, "csv")


def _by_group(rows, key: str) -> None:
    cols = ["cov_mean", "top_junction_count", "n_supported", "se_minor_count",
            "a3_minor_count", "intron_cov_frac", "left_flank", "right_flank",
            "struct_span_frac"]
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key, "?")].append(row)
    print(f"{key:<8} {'n':>5} " + " ".join(f"{c[:12]:>12}" for c in cols))
    for g in sorted(groups):
        sub = groups[g]
        vals = []
        for c in cols:
            v = np.array([r.get(c, np.nan) for r in sub], dtype=float)
            v = v[np.isfinite(v)]
            vals.append(f"{np.median(v):>12.4g}" if v.size else f"{'-':>12}")
        print(f"{g:<8} {len(sub):>5} " + " ".join(vals))


# --------------------------------------------------------------------------
# Track B input: how much padding does a curator leave around the event?
# --------------------------------------------------------------------------


def padding_report(benchmark_dir: str, ioe_dir: str) -> None:
    """Match every curated benchmark window to the SUPPA catalogue event it was
    drawn from, and report the padding on each edge.

    This is the empirical answer to what ``sim/window.py`` should be drawing.
    The simulation's design regions come from *this same catalogue*, so the
    comparison is exact rather than distributional: if the curator's window is
    the event span plus a constant, the simulation can reproduce that directly.

    Matching is by best boundary agreement among catalogue events of the
    matching type on the same chromosome. Benchmark events absent from the
    strict catalogue (rMATS-only calls) simply do not match and are counted.
    """
    from sim.catalogue import build_catalogue

    csv_label = {"A3": "A3", "A5": "A5", "ES": "CSE", "IR": "RI"}
    events = build_catalogue(ioe_dir, types=("SE", "A3", "A5", "RI"))
    by_key = defaultdict(list)
    for e in events:
        by_key[(e.chrom, e.label)].append(e)

    print(f"=== curated window vs SUPPA event span ({len(events)} catalogued events) ===")
    print("padding > 0 means the curated window extends beyond the event span.\n")
    print(f"{'csv':<5} {'matched':>9} {'pad_left':>26} {'pad_right':>26} {'span/width':>11}")
    print(f"{'':<5} {'':>9} {'med [q10, q90]':>26} {'med [q10, q90]':>26}")
    pooled = []
    for name, label in csv_label.items():
        rows = []
        n_total = 0
        for line in open(f"{benchmark_dir}/{name}.txt"):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            chrom, span = line.split(":")
            ws, we = (int(x) for x in span.split("-"))
            best = None
            for e in by_key.get((chrom, label), ()):
                if min(we, e.region_end) - max(ws, e.region_start) <= 0:
                    continue
                cost = abs(e.region_start - ws) + abs(e.region_end - we)
                if best is None or cost < best[0]:
                    best = (cost, e)
            if best is None:
                continue
            e = best[1]
            rows.append((e.region_start - ws, we - e.region_end,
                         (e.region_end - e.region_start + 1) / (we - ws + 1)))
        pooled += rows
        if not rows:
            print(f"{name:<5} {0:>4}/{n_total:<4}")
            continue
        L = np.array([r[0] for r in rows], dtype=float)
        R = np.array([r[1] for r in rows], dtype=float)
        F = np.array([r[2] for r in rows], dtype=float)
        print(f"{name:<5} {len(rows):>4}/{n_total:<4} "
              f"{np.median(L):>10.0f} [{np.quantile(L, .1):>6.0f},{np.quantile(L, .9):>6.0f}] "
              f"{np.median(R):>10.0f} [{np.quantile(R, .1):>6.0f},{np.quantile(R, .9):>6.0f}] "
              f"{np.median(F):>11.3f}")

    L = np.array([r[0] for r in pooled], dtype=float)
    R = np.array([r[1] for r in pooled], dtype=float)
    F = np.array([r[2] for r in pooled], dtype=float)
    n = len(pooled)
    print(f"\npooled n={n}")
    print(f"  window == event span on both edges exactly: "
          f"{sum(1 for r in pooled if r[0] == 0 and r[1] == 0)}/{n}")
    print(f"  both edges within 10 bp:                    "
          f"{sum(1 for r in pooled if abs(r[0]) <= 10 and abs(r[1]) <= 10)}/{n}")
    print(f"  both edges within 200 bp:                   "
          f"{sum(1 for r in pooled if abs(r[0]) <= 200 and abs(r[1]) <= 200)}/{n}")
    print(f"  event span / window width: median {np.median(F):.3f}")
    print(f"  pooled padding: left median {np.median(L):.0f}, right median {np.median(R):.0f}, "
          f"max |pad| {max(np.abs(L).max(), np.abs(R).max()):.0f}")
    print("\nRead this as the specification for sim/window.py: the curated window is\n"
          "the event span plus a small constant (per type: ES +100 bp on each edge,\n"
          "IR exactly 0, A3/A5 -1 -- an off-by-one coordinate convention). Drawing a\n"
          "label-independent flank of the same magnitude reproduces the geometry\n"
          "without hard-coding a per-class constant, which would be pure leakage.")


def _read_tsv(path: str) -> list:
    with open(path) as fh:
        rows = []
        for r in csv.DictReader(fh, delimiter="\t"):
            out = {}
            for k, v in r.items():
                if k in ("region", "strand", "label", "kind", "csv", "source"):
                    out[k] = v
                else:
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        out[k] = float("nan")
            rows.append(out)
    return rows


def _measure(args) -> None:
    filters = dsextract.ReadFilters(
        min_anchor=args.min_anchor, max_nh=args.max_nh, min_mapq=args.min_mapq
    )

    jobs = []
    for path in args.csv:
        stem = splitext(basename(path))[0]
        labels = None
        if args.labels:
            with open(args.labels) as fh:
                labels = list(csv.DictReader(fh, delimiter="\t"))
        for i, (region, bam1, bam2) in enumerate(read_prediction_csv(path)):
            meta = {"source": stem}
            if labels is not None and i < len(labels):
                meta["label"] = labels[i]["label"]
                meta["kind"] = labels[i]["kind"]
            else:
                meta["csv"] = stem
            jobs.append((region, bam1, bam2, meta))

    if args.subsample and args.subsample < len(jobs):
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(jobs), size=args.subsample, replace=False)
        jobs = [jobs[i] for i in sorted(pick)]
        print(f"subsampled to {len(jobs)} windows (seed {args.seed})", file=sys.stderr)

    rows = []
    for i, (region, bam1, bam2, meta) in enumerate(jobs):
        row = audit_event(region, bam1, bam2, filters, args.min_count,
                          args.min_minor_frac, args.flank)
        row.update(meta)
        rows.append(row)
        if args.verbose or (i % 100 == 0):
            print(f"  [{i + 1}/{len(jobs)}] {region} cov={row['cov_mean']:.1f} "
                  f"novel={row['novel_junction_rate']:.3f} dup={row['dup_rate']:.3f}",
                  file=sys.stderr, flush=True)

    fields = list(rows[0].keys()) if rows else []
    for extra in ("source", "csv", "label", "kind"):
        if any(extra in r for r in rows) and extra not in fields:
            fields.append(extra)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out} ({len(rows)} windows)", file=sys.stderr)

    n_untagged = sum(1 for r in rows if not np.isfinite(r["novel_junction_rate"]))
    if n_untagged:
        print(f"NOTE: {n_untagged}/{len(rows)} windows had no jM-tagged supported "
              f"junction -- their novel-junction rate is NaN, not zero", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="one row per window from a region,bam1,bam2 CSV")
    m.add_argument("--csv", nargs="+", required=True)
    m.add_argument("--labels", default=None,
                   help="a batch's labels.tsv; row order must match --csv (both "
                        "come from sim/build_extract_csv.py)")
    m.add_argument("--out", required=True)
    m.add_argument("--subsample", type=int, default=0, help="0 = all windows")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--min-count", type=float, default=DEFAULT_MIN_COUNT)
    m.add_argument("--min-minor-frac", type=float, default=DEFAULT_MIN_MINOR_FRAC)
    m.add_argument("--flank", type=int, default=DEFAULT_FLANK)
    m.add_argument("--min-anchor", type=int, default=5)
    m.add_argument("--max-nh", type=int, default=10)
    m.add_argument("--min-mapq", type=int, default=0)
    m.add_argument("-v", "--verbose", action="store_true")

    c = sub.add_parser("compare", help="sim-vs-real table from two measure outputs")
    c.add_argument("--real", required=True)
    c.add_argument("--sim", nargs="+", required=True)
    c.add_argument("--threshold", type=float, default=2.0)

    p = sub.add_parser("padding", help="curated window vs SUPPA event span (Track B)")
    p.add_argument("--benchmark-dir", default="benchmark")
    p.add_argument("--ioe-dir", default="sim/events")

    args = ap.parse_args(argv)
    if args.cmd == "measure":
        _measure(args)
    elif args.cmd == "padding":
        padding_report(args.benchmark_dir, args.ioe_dir)
    else:
        sim_rows = []
        for p in args.sim:
            sim_rows += _read_tsv(p)
        compare(_read_tsv(args.real), sim_rows, args.threshold)


if __name__ == "__main__":
    main()
