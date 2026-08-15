"""Score a trained signal-native model against the curated 140-event benchmark
(Phase 3 step 5 / top-level plan Verification item 6).

The top-level plan says to reuse ``tools/score_benchmark.py``; that file does
not exist in this repo, so this is it.

The benchmark (``benchmark/{A3,A5,ES,IR}.csv``, 140 events over the local
``benchmark/c1.bam``/``c2.bam``) is the true held-out set: it is real data,
manually curated from an rMATS + SUPPA consensus (see benchmark/README.md), and
nothing in Phase 2 or Phase 3 has been fit to it. Its CSV -> label map is
``A3.csv``->A3, ``A5.csv``->A5, ``ES.csv``->CSE, ``IR.csv``->RI. **There are no
NN events in it**, so it scores the positive classes only -- an NN prediction is
always wrong here, and NN's own recall is unmeasurable.

Both numbers are reported: the 5-class one, and the A3+A5 -> A collapse, which
is what is comparable to v1's 123/140. Per-event region width is included in the
output so accuracy-vs-width is plottable against v1's.

**Expect the sim-only model to do poorly.** This is zero-shot sim->real transfer
with no fine-tuning; Phase 4 is what closes that gap. The number's value is as
the baseline for the Phase-5 "noise realism transfers" ablation, not as a
headline.

Usage:
    python -m dsignal.score_benchmark --net runs/control/fold*.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from os.path import basename, splitext

import numpy as np

from dsignal.extract import ReadFilters, parse_region
from dsignal.predict import Predictor, read_prediction_csv

#: benchmark CSV basename -> our label. ES ("exon skipping") is SUPPA's SE,
#: which Phase 2 labels CSE; IR ("intron retention") is RI.
CSV_LABEL = {"A3": "A3", "A5": "A5", "ES": "CSE", "IR": "RI"}

DEFAULT_CSVS = ["benchmark/A3.csv", "benchmark/A5.csv", "benchmark/ES.csv", "benchmark/IR.csv"]


def collapse(label: str) -> str:
    return "A" if label in ("A3", "A5") else label


def score(predictor: Predictor, csv_paths, verbose: bool = False):
    rows = []
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r} (known: {sorted(CSV_LABEL)})")
        truth = CSV_LABEL[stem]
        for region, bam1, bam2 in read_prediction_csv(path):
            chrom, start, end = parse_region(region)
            pred, probs, _, timing = predictor.predict_region(region, bam1, bam2)
            rows.append(
                {
                    "csv": stem,
                    "region": region,
                    "width": end - start,
                    "truth": truth,
                    "pred": pred,
                    "correct": int(pred == truth),
                    "correct_collapsed": int(collapse(pred) == collapse(truth)),
                    **{f"p_{c}": probs[i] for i, c in enumerate(predictor.classes)},
                    **timing,
                }
            )
            if verbose:
                print(f"  {region:32s} truth={truth:4s} pred={pred:4s} "
                      f"{'ok' if pred == truth else 'XX'}  ({timing['total_s']:.2f}s)", file=sys.stderr)
    return rows


def _confusion(rows, key_true, key_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    idx = {l: i for i, l in enumerate(labels)}
    for r in rows:
        t, p = r[key_true], r[key_pred]
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def report(rows, predictor) -> None:
    n = len(rows)
    print(f"\n=== curated benchmark: {n} events ===")

    print("\nper-CSV accuracy (5-class / A3+A5 collapsed):")
    for stem in ("A3", "A5", "ES", "IR"):
        sub = [r for r in rows if r["csv"] == stem]
        if not sub:
            continue
        acc = np.mean([r["correct"] for r in sub])
        acc_c = np.mean([r["correct_collapsed"] for r in sub])
        print(f"  {stem:3s} (truth {CSV_LABEL[stem]:4s}) n={len(sub):3d}  "
              f"{acc:6.1%} / {acc_c:6.1%}   ({sum(r['correct'] for r in sub)}/{len(sub)} correct)")

    n_correct = sum(r["correct"] for r in rows)
    n_correct_c = sum(r["correct_collapsed"] for r in rows)
    print(f"\naggregate 5-class:  {n_correct}/{n} = {n_correct / n:.1%}")
    print(f"aggregate collapsed: {n_correct_c}/{n} = {n_correct_c / n:.1%}   "
          f"(this is the number comparable to v1's 123/140 = 87.9%)")

    labels = list(predictor.classes)
    print(f"\nconfusion matrix, 5-class (rows=truth, cols=pred): {labels}")
    print("  NB: the benchmark contains no NN events, so the NN row is empty by")
    print("  construction and every NN *prediction* is a false positive.")
    print(_confusion(rows, "truth", "pred", labels))

    clabels = sorted({collapse(l) for l in labels})
    for r in rows:
        r["truth_c"], r["pred_c"] = collapse(r["truth"]), collapse(r["pred"])
    print(f"\nconfusion matrix, A3+A5 collapsed (rows=truth, cols=pred): {clabels}")
    print(_confusion(rows, "truth_c", "pred_c", clabels))

    widths = np.array([r["width"] for r in rows])
    print("\naccuracy vs region width (5-class), for plotting against v1's:")
    edges = [0, 1000, 5000, 20000, 100000, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["width"] < hi]
        if not sel:
            continue
        hi_s = "inf" if hi == np.inf else f"{int(hi)}"
        print(f"  [{int(lo):>6}, {hi_s:>6}) n={len(sel):3d}  "
              f"{np.mean([r['correct'] for r in sel]):6.1%} / "
              f"{np.mean([r['correct_collapsed'] for r in sel]):6.1%}")

    tot = np.array([r["total_s"] for r in rows])
    ext = np.array([r["extract_s"] for r in rows])
    inf = np.array([r["infer_s"] for r in rows])
    print(
        f"\nwall clock: {tot.mean():.3f} s/event "
        f"({ext.mean():.3f} extract + {inf.mean():.3f} inference), {tot.sum():.1f} s total.\n"
        f"  v1's ensemble spends ~17 s/event on image rendering alone, so this is\n"
        f"  the Phase-5 cost comparison in its rawest form."
    )
    print(f"  widest event {widths.max():,} bp, median {int(np.median(widths)):,} bp")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", nargs="+", required=True, help="trained fold checkpoints (.pt)")
    ap.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    ap.add_argument("--out", default=None, help="per-event TSV (region, width, truth, pred, probs, timings)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-anchor", type=int, default=5)
    ap.add_argument("--max-nh", type=int, default=10)
    ap.add_argument("--min-mapq", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    predictor = Predictor(
        args.net,
        device=args.device,
        filters=ReadFilters(min_anchor=args.min_anchor, max_nh=args.max_nh, min_mapq=args.min_mapq),
    )
    print(f"loaded {len(predictor.models)} fold checkpoint(s), classes {predictor.classes}, "
          f"feature subset {predictor.feature_subset!r}", file=sys.stderr)

    rows = score(predictor, args.csv, args.verbose)
    report(rows, predictor)

    if args.out:
        fields = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote per-event results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
