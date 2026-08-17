"""Step 0 of ``plans/signal-native-v2-scaleup-7500.md``: why does the model call
curated exon-skipping (ES/CSE) events ``A3``?

The concrete hypothesis to test: **a curated ES window often also contains a
genuine alternative-3' junction pair (two junctions sharing a donor), so the
model sees real A3 evidence and the "right" answer depends on which event the
curator annotated, not on which junctions exist.**

This script re-extracts the junction set of every benchmark event straight from
``benchmark/c1.bam``/``c2.bam``, canonicalizes it exactly the way
``dsignal/features.py`` does (minus-strand records mirrored, so "donor" always
means the transcription-direction donor), and scores three *structural* motifs
on the junction set:

``a3``   two junctions sharing a **donor** and differing in acceptor
``a5``   two junctions sharing an **acceptor** and differing in donor
``se``   an exon-skipping triplet: ``(s1,e1)``, ``(s2,e2)`` with an exon between
         them (``e1 < s2``) *and* the skip junction ``(s1,e2)``

Each motif also gets a *differential* strength -- ``|frac_1 - frac_2|`` of the
minor branch across the two conditions -- because the benchmark's truth set is
a **differential** splicing truth set, not merely "this structure exists".

Nothing here is a model: it is annotation-free junction bookkeeping, so it can
disagree with the classifier without circularity.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from os.path import dirname, join as pjoin

import numpy as np

sys.path.insert(0, dirname(dirname(__file__)))

from dsignal import extract as dsextract
from dsignal import features as dsfeatures

#: a junction is "supported" if it reaches this count in *either* condition
DEFAULT_MIN_COUNT = 5
#: minor branch must hold at least this share of its group, pooled over
#: conditions, for the pair to count as a real alternative (rather than noise
#: hanging off a dominant junction)
DEFAULT_MIN_MINOR_FRAC = 0.05


def _supported(norm, min_count: float):
    tot = np.maximum(norm.count_1, norm.count_2)
    return np.flatnonzero(tot >= min_count)


def _best_shared_pair(norm, idx, key_attr: str, min_minor_frac: float):
    """Best pair of supported junctions sharing ``key_attr`` (``s_rel`` = shared
    donor -> A3; ``e_rel`` = shared acceptor -> A5).

    Returns ``None`` or a dict with the pooled minor-branch fraction, the
    per-condition minor fractions and their absolute difference.
    """
    key = getattr(norm, key_attr)
    groups = defaultdict(list)
    for i in idx:
        groups[int(key[i])].append(i)

    best = None
    for members in groups.values():
        if len(members) < 2:
            continue
        pooled = np.array([norm.count_1[i] + norm.count_2[i] for i in members])
        order = np.argsort(pooled)[::-1]
        top, second = members[order[0]], members[order[1]]
        tot = pooled.sum()
        if tot <= 0:
            continue
        minor_frac = float(pooled[order[1]] / tot)
        if minor_frac < min_minor_frac:
            continue
        # differential strength of the minor branch, per condition
        g1 = sum(norm.count_1[i] for i in members)
        g2 = sum(norm.count_2[i] for i in members)
        f1 = float(norm.count_1[second] / g1) if g1 > 0 else float("nan")
        f2 = float(norm.count_2[second] / g2) if g2 > 0 else float("nan")
        delta = abs(f1 - f2) if np.isfinite(f1) and np.isfinite(f2) else float("nan")
        cand = {
            "n_in_group": len(members),
            "minor_frac": minor_frac,
            "minor_count": float(pooled[order[1]]),
            "frac_1": f1,
            "frac_2": f2,
            "delta": delta,
            "top": (int(norm.s_rel[top]), int(norm.e_rel[top])),
            "alt": (int(norm.s_rel[second]), int(norm.e_rel[second])),
        }
        if best is None or cand["minor_frac"] > best["minor_frac"]:
            best = cand
    return best


def _best_se_triplet(norm, idx, min_minor_frac: float):
    """Best exon-skipping triplet: inclusion junctions ``(s1,e1)``/``(s2,e2)``
    flanking an exon, plus the skip junction ``(s1,e2)``.

    ``minor_frac`` here is the *skip* share, i.e. ``1 - PSI`` pooled; ``delta``
    is the per-condition change in that share -- the differential SE signal.
    """
    s, e = norm.s_rel, norm.e_rel
    by_key = {(int(s[i]), int(e[i])): i for i in idx}
    starts = defaultdict(list)
    ends = defaultdict(list)
    for i in idx:
        starts[int(s[i])].append(i)
        ends[int(e[i])].append(i)

    best = None
    for (s1, e2), skip in by_key.items():
        for up in starts[s1]:  # (s1, e1)
            e1 = int(e[up])
            if e1 >= e2:
                continue
            for down in ends[e2]:  # (s2, e2)
                s2 = int(s[down])
                if s2 <= e1:  # need a real exon between the two introns
                    continue
                if down == skip or up == skip:
                    continue
                incl_1 = min(norm.count_1[up], norm.count_1[down])
                incl_2 = min(norm.count_2[up], norm.count_2[down])
                sk_1, sk_2 = norm.count_1[skip], norm.count_2[skip]
                tot = incl_1 + incl_2 + sk_1 + sk_2
                if tot <= 0:
                    continue
                minor_frac = float(min(incl_1 + incl_2, sk_1 + sk_2) / tot)
                if minor_frac < min_minor_frac:
                    continue
                d1 = incl_1 + sk_1
                d2 = incl_2 + sk_2
                f1 = float(sk_1 / d1) if d1 > 0 else float("nan")
                f2 = float(sk_2 / d2) if d2 > 0 else float("nan")
                delta = abs(f1 - f2) if np.isfinite(f1) and np.isfinite(f2) else float("nan")
                cand = {
                    "exon_len": s2 - e1 - 1,
                    "minor_frac": minor_frac,
                    "minor_count": float(min(incl_1 + incl_2, sk_1 + sk_2)),
                    "frac_1": f1,
                    "frac_2": f2,
                    "delta": delta,
                }
                if best is None or cand["minor_frac"] > best["minor_frac"]:
                    best = cand
    return best


def analyse_event(region: str, bam1: str, bam2: str, min_count: float, min_minor_frac: float):
    record = dsextract.extract_event(dsextract.parse_region(region), bam1, bam2)
    norm = dsfeatures.normalize_record(record)
    idx = _supported(norm, min_count)
    pooled = norm.count_1 + norm.count_2
    row = {
        "region": region,
        "strand": norm.strand,
        "n_junctions": int(len(norm.s_rel)),
        "n_supported": int(len(idx)),
        # depth proxies: a weakly-supported motif and a strongly-supported one
        # are very different evidence even at the same fraction
        "cov_mean": float((norm.cov_1.mean() + norm.cov_2.mean()) / 2) if norm.n else 0.0,
        "top_junction_count": float(pooled.max()) if pooled.size else 0.0,
    }
    for name, best in (
        ("a3", _best_shared_pair(norm, idx, "s_rel", min_minor_frac)),
        ("a5", _best_shared_pair(norm, idx, "e_rel", min_minor_frac)),
        ("se", _best_se_triplet(norm, idx, min_minor_frac)),
    ):
        row[f"has_{name}"] = int(best is not None)
        row[f"{name}_minor_frac"] = best["minor_frac"] if best else 0.0
        row[f"{name}_minor_count"] = best["minor_count"] if best else 0.0
        row[f"{name}_delta"] = best["delta"] if best else 0.0
    return row


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", default="benchmark")
    ap.add_argument(
        "--predictions",
        default="runs/benchmark_events_geomwin.tsv",
        help="score_benchmark.py per-event TSV to cross-tabulate against",
    )
    ap.add_argument("--out", default="runs/cse_a3_diagnosis.tsv")
    ap.add_argument("--min-count", type=float, default=DEFAULT_MIN_COUNT)
    ap.add_argument("--min-minor-frac", type=float, default=DEFAULT_MIN_MINOR_FRAC)
    ap.add_argument(
        "--ioe-dir",
        default=None,
        help="directory of SUPPA events_<TYPE>_strict.ioe files; enables the "
        "annotation-side check (do catalogued A3 and SE events genuinely "
        "overlap the curated ES windows?)",
    )
    ap.add_argument(
        "--sim-npz",
        default=None,
        help="directory of simulated .npz records (e.g. sim/experiment/npz_windowfix); "
        "enables the sim-vs-benchmark evidence-strength comparison",
    )
    ap.add_argument("--sim-labels", default=None, help="labels TSV matching --sim-npz")
    args = ap.parse_args(argv)

    preds = {}
    with open(args.predictions) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            preds[r["region"]] = r

    rows = []
    for csv_name in ("A3", "A5", "ES", "IR"):
        path = pjoin(args.benchmark_dir, f"{csv_name}.csv")
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                region, bam1, bam2 = line.split(",")[:3]
                row = analyse_event(region, bam1, bam2, args.min_count, args.min_minor_frac)
                p = preds.get(region, {})
                row["csv"] = csv_name
                row["truth"] = p.get("truth", "")
                row["pred"] = p.get("pred", "")
                for k in ("p_A3", "p_A5", "p_CSE", "p_NN", "p_RI"):
                    row[k] = float(p[k]) if p.get(k) else float("nan")
                rows.append(row)
                print(f"  {csv_name} {region} pred={row['pred']:>3} "
                      f"a3={row['has_a3']} a5={row['has_a5']} se={row['has_se']}", flush=True)

    fields = ["csv", "region", "truth", "pred", "strand", "n_junctions", "n_supported",
              "cov_mean", "top_junction_count"]
    for name in ("a3", "a5", "se"):
        fields += [f"has_{name}", f"{name}_minor_frac", f"{name}_minor_count", f"{name}_delta"]
    fields += ["p_A3", "p_A5", "p_CSE", "p_NN", "p_RI"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r[k]) for k in fields})
    print(f"\nwrote {args.out} ({len(rows)} events)")

    _report(rows)
    if args.ioe_dir:
        _catalogue_report(rows, args.ioe_dir)
    if args.sim_npz:
        _sim_report(args.sim_npz, args.sim_labels, args.min_count, args.min_minor_frac)


def _report(rows) -> None:
    es = [r for r in rows if r["csv"] == "ES"]
    print("\n=== ES events, structural motifs present in the curated window ===")
    print(f"{'group':<24} {'n':>4} {'has_a3':>7} {'has_a5':>7} {'has_se':>7} "
          f"{'a3_delta':>9} {'se_delta':>9} {'p_A3':>6} {'p_CSE':>6}")
    groups = [("all ES", es)]
    for pred in ("A3", "A5", "CSE", "NN", "RI"):
        sub = [r for r in es if r["pred"] == pred]
        if sub:
            groups.append((f"ES predicted {pred}", sub))
    for name, sub in groups:
        n = len(sub)
        print(f"{name:<24} {n:>4} "
              f"{sum(r['has_a3'] for r in sub)/n:>7.2f} "
              f"{sum(r['has_a5'] for r in sub)/n:>7.2f} "
              f"{sum(r['has_se'] for r in sub)/n:>7.2f} "
              f"{np.nanmean([r['a3_delta'] for r in sub]):>9.3f} "
              f"{np.nanmean([r['se_delta'] for r in sub]):>9.3f} "
              f"{np.nanmean([r['p_A3'] for r in sub]):>6.3f} "
              f"{np.nanmean([r['p_CSE'] for r in sub]):>6.3f}")

    print("\n=== does p_A3 track A3 evidence, within the ES set? ===")
    for key in ("a3_minor_frac", "a3_delta", "se_minor_frac", "se_delta"):
        x = np.array([r[key] for r in es], dtype=float)
        y = np.array([r["p_A3"] for r in es], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 2 and x[ok].std() > 0:
            print(f"  pearson r(p_A3, {key:<16}) = {np.corrcoef(x[ok], y[ok])[0, 1]:+.3f}")

    print("\n=== motif prevalence by CSV (does A3 evidence discriminate at all?) ===")
    print(f"{'csv':<5} {'n':>4} {'has_a3':>7} {'has_a5':>7} {'has_se':>7}")
    for csv_name in ("A3", "A5", "ES", "IR"):
        sub = [r for r in rows if r["csv"] == csv_name]
        n = len(sub)
        print(f"{csv_name:<5} {n:>4} {sum(r['has_a3'] for r in sub)/n:>7.2f} "
              f"{sum(r['has_a5'] for r in sub)/n:>7.2f} "
              f"{sum(r['has_se'] for r in sub)/n:>7.2f}")

    print("\n=== joint motif presence, ES only ===")
    c = Counter((r["has_a3"], r["has_a5"], r["has_se"]) for r in es)
    for (a3, a5, se), k in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  a3={a3} a5={a5} se={se}: {k}")

    # "present" is a weak bar: the truth set is a *differential* one, so ask
    # instead whether each motif is differentially used between conditions --
    # i.e. whether rMATS/SUPPA would plausibly have called it too.
    print("\n=== differentially-used motifs (delta >= 0.10), by CSV ===")
    print(f"{'csv':<5} {'n':>4} {'diff_a3':>8} {'diff_a5':>8} {'diff_se':>8}")
    for csv_name in ("A3", "A5", "ES", "IR"):
        sub = [r for r in rows if r["csv"] == csv_name]
        n = len(sub)
        print(f"{csv_name:<5} {n:>4} "
              + " ".join(f"{sum(1 for r in sub if r[f'{m}_delta'] >= 0.10) / n:>8.2f}"
                         for m in ("a3", "a5", "se")))

    # The hypothesis is only supported if the events called A3 carry *more* A3
    # evidence than the ones called CSE. Compare the two groups head to head.
    print("\n=== ES->A3 vs ES->CSE, median evidence (the actual hypothesis test) ===")
    keys = ["cov_mean", "top_junction_count", "n_supported",
            "a3_minor_frac", "a3_minor_count", "a3_delta",
            "se_minor_frac", "se_minor_count", "se_delta"]
    a3g = [r for r in es if r["pred"] == "A3"]
    cseg = [r for r in es if r["pred"] == "CSE"]
    print(f"{'metric':<20} {'ES->A3':>12} {'ES->CSE':>12}")
    for k in keys:
        print(f"{k:<20} {np.nanmedian([r[k] for r in a3g]):>12.3f} "
              f"{np.nanmedian([r[k] for r in cseg]):>12.3f}")

    print("\n=== which motif is strongest (largest delta), ES only ===")
    def _strongest(r):
        return max(("a3", "a5", "se"), key=lambda m: r[f"{m}_delta"])
    c = Counter((_strongest(r), r["pred"]) for r in es)
    print(f"{'strongest':<10} " + " ".join(f"{p:>5}" for p in ("A3", "A5", "CSE", "NN", "RI")))
    for m in ("a3", "a5", "se"):
        print(f"{m:<10} " + " ".join(f"{c[(m, p)]:>5}" for p in ("A3", "A5", "CSE", "NN", "RI")))


def _sim_report(npz_dir: str, labels_path: str, min_count: float, min_minor_frac: float) -> None:
    """Same motif detector, run on the *simulated* training records.

    The point is not the sim's accuracy but the **evidence strength** it trains
    on: if simulated CSE events carry far more skip-junction support than the
    curated ones do, the model's CSE detector is calibrated to a depth the real
    data never reaches, and it will fall back to whatever structure is left.
    """
    import glob
    from os.path import basename

    labels = {}
    with open(labels_path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            labels[r["ix"]] = r["label"]

    by_label = defaultdict(list)
    for path in sorted(glob.glob(pjoin(npz_dir, "*.npz"))):
        ix = basename(path).split(".")[0]
        label = labels.get(ix)
        if label is None:
            continue
        norm = dsfeatures.normalize_record(dsextract.load_record(path))
        idx = _supported(norm, min_count)
        a3 = _best_shared_pair(norm, idx, "s_rel", min_minor_frac)
        se = _best_se_triplet(norm, idx, min_minor_frac)
        by_label[label].append({
            "cov_mean": float((norm.cov_1.mean() + norm.cov_2.mean()) / 2) if norm.n else 0.0,
            "n_supported": len(idx),
            "has_a3": int(a3 is not None),
            "has_se": int(se is not None),
            "a3_minor_frac": a3["minor_frac"] if a3 else 0.0,
            "a3_minor_count": a3["minor_count"] if a3 else 0.0,
            "se_minor_frac": se["minor_frac"] if se else 0.0,
            "se_minor_count": se["minor_count"] if se else 0.0,
        })

    print(f"\n=== same motif detector on the simulated training records ({npz_dir}) ===")
    print(f"{'sim label':<10} {'n':>4} {'cov':>7} {'nsup':>5} {'has_a3':>7} {'has_se':>7} "
          f"{'a3frac':>7} {'sefrac':>7} {'a3cnt':>7} {'secnt':>7}")
    for label, vals in sorted(by_label.items()):
        med = lambda k: np.median([v[k] for v in vals])  # noqa: E731
        print(f"{label:<10} {len(vals):>4} {med('cov_mean'):>7.1f} {med('n_supported'):>5.0f} "
              f"{np.mean([v['has_a3'] for v in vals]):>7.2f} "
              f"{np.mean([v['has_se'] for v in vals]):>7.2f} "
              f"{med('a3_minor_frac'):>7.3f} {med('se_minor_frac'):>7.3f} "
              f"{med('a3_minor_count'):>7.0f} {med('se_minor_count'):>7.0f}")


def _catalogue_report(rows, ioe_dir: str) -> None:
    """Annotation-side counterpart of the junction-level check: does SUPPA's own
    catalogue put A3 events inside the curated ES windows?

    This is the same catalogue the simulation's labels come from, so it answers
    "is the label ambiguous in the annotation too, or only in the reads?".
    """
    from sim.catalogue import build_catalogue

    events = build_catalogue(ioe_dir, types=("SE", "A3", "A5", "RI"))
    by_chrom = defaultdict(list)
    for e in events:
        by_chrom[e.chrom].append(e)
    print(f"\n=== SUPPA catalogue overlap ({len(events)} catalogued events) ===")

    def overlapping(region: str):
        chrom, span = region.split(":")
        lo, hi = (int(x) for x in span.split("-"))
        hits = Counter()
        for e in by_chrom.get(chrom, ()):
            if e.region_start <= hi and e.region_end >= lo:
                hits[e.label] += 1
        return hits

    print(f"{'csv':<5} {'n':>4} " + " ".join(f"{lab:>10}" for lab in ("A3", "A5", "CSE", "RI")))
    for csv_name in ("A3", "A5", "ES", "IR"):
        sub = [r for r in rows if r["csv"] == csv_name]
        hits = [overlapping(r["region"]) for r in sub]
        print(f"{csv_name:<5} {len(sub):>4} "
              + " ".join(f"{sum(1 for h in hits if h[lab]) / len(sub):>10.2f}"
                         for lab in ("A3", "A5", "CSE", "RI"))
              + "   (fraction of windows containing >=1 catalogued event of that type)")

    es = [r for r in rows if r["csv"] == "ES"]
    print("\n  ES windows, mean catalogued events per window, by prediction:")
    print(f"  {'group':<20} {'n':>4} " + " ".join(f"{lab:>7}" for lab in ("A3", "A5", "CSE", "RI")))
    for pred in ("A3", "A5", "CSE", "NN", "RI"):
        sub = [r for r in es if r["pred"] == pred]
        if not sub:
            continue
        hits = [overlapping(r["region"]) for r in sub]
        print(f"  {'predicted ' + pred:<20} {len(sub):>4} "
              + " ".join(f"{np.mean([h[lab] for h in hits]):>7.1f}" for lab in ("A3", "A5", "CSE", "RI")))


if __name__ == "__main__":
    main()
