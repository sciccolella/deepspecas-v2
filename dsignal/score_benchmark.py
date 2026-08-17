"""Score a trained signal-native model against the curated 140-event benchmark
(Phase 3 step 5 / top-level plan Verification item 6).

The top-level plan says to reuse ``tools/score_benchmark.py``; that file does
not exist in this repo, so this is it.

The benchmark (``benchmark/{A3,A5,ES,IR}.csv``, 140 events over the local
``benchmark/c1.bam``/``c2.bam``) is the held-out set: it is real data, manually
curated from an rMATS + SUPPA consensus (see benchmark/README.md).

**It is no longer fully held out, and the two leaks must be reported with any
number from here:**

* ``DPSI_BETA`` / ``PSI_LEVEL_BETA`` in ``sim/select_events.py`` are fitted by
  moments to the benchmark's own 125 matched events, so anything simulated with
  ``--psi-dist beta`` is domain-calibrated along the effect-size axis rather
  than zero-shot. What leaks is two 2-parameter marginals, not labels or
  structures -- but it is not nothing.
* the v3 window sampler's flank magnitude (0-100 bp) was measured off the
  curated windows (``python -m dsignal.audit_noise padding``). The per-type
  padding constants were deliberately *not* reproduced, since those would be
  label leakage; the magnitude was.

Neither is fixed by anything currently in the repo. The clean fix is a held-out
split of the benchmark, or an independent dPSI distribution.

Its CSV -> label map is
``A3.csv``->A3, ``A5.csv``->A5, ``ES.csv``->CSE, ``IR.csv``->RI. **There are no
NN events in it**, so it scores the positive classes only -- an NN prediction is
always wrong here, and NN's own recall is unmeasurable.

Both numbers are reported: the 5-class one, and the A3+A5 -> A collapse, which
is what is comparable to v1's 123/140. Per-event region width is included in the
output so accuracy-vs-width is plottable against v1's.

This is sim->real transfer with **no fine-tuning at all** -- the model has never
seen a read from this dataset. Current standing: **110/140 = 78.6% collapsed**
(``runs/benchmark_geomwin.log``), against v1's 123/140 = 87.9%, reproduced
locally in ``runs/v1/``. The gap was 25.8 points before the window sampler was
fixed and is 8.6 now.

Fine-tuning on real data (the original plan's Phase 4) is **not** currently
recommended; see plans/signal-native-v2-resume-2.md for why, and for the two
readings of what the remaining gap is.

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
from dsignal.predict import (
    Predictor,
    PriorCorrection,
    class_counts_from_labels,
    make_target_prior,
    read_prediction_csv,
)

#: benchmark CSV basename -> our label. ES ("exon skipping") is SUPPA's SE,
#: which Phase 2 labels CSE; IR ("intron retention") is RI.
CSV_LABEL = {"A3": "A3", "A5": "A5", "ES": "CSE", "IR": "RI"}

DEFAULT_CSVS = ["benchmark/A3.csv", "benchmark/A5.csv", "benchmark/ES.csv", "benchmark/IR.csv"]


def collapse(label: str) -> str:
    return "A" if label in ("A3", "A5") else label


def score(predictor: Predictor, csv_paths, verbose: bool = False, keep=None):
    """``keep``, when given, is a predicate on ``(csv_stem, region)`` -- used by
    ``--half`` to score only one side of ``benchmark/split.tsv``. Filtering here
    rather than after the fact means the skipped half is never extracted."""
    rows = []
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r} (known: {sorted(CSV_LABEL)})")
        truth = CSV_LABEL[stem]
        for region, bam1, bam2 in read_prediction_csv(path):
            if keep is not None and not keep(stem, region):
                continue
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


def report(rows, predictor, half: str = "both") -> None:
    n = len(rows)
    which = "all 140" if half == "both" else f"{half} half"
    print(f"\n=== curated benchmark ({which}): {n} events ===")

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
    v1_note = ("(this is the number comparable to v1's 123/140 = 87.9%)" if half == "both"
               else f"({half} half only -- v1's 123/140 = 87.9% is over all 140)")
    print(f"aggregate collapsed: {n_correct_c}/{n} = {n_correct_c / n:.1%}   {v1_note}")

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
        f"  For scale: v1's six-net ensemble spends ~17 s/event on image rendering alone\n"
        f"  on the default PIL path (see report.md; --gpu-render cuts that several-fold)."
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
    ap.add_argument(
        "--target-prior", choices=("none", "uniform", "benchmark"), default="none",
        help="opt-in class-prior correction (dsignal.predict.PriorCorrection); "
        "'none' is the default and leaves inference exactly as it was. "
        "'benchmark' (NN excluded) is an oracle upper bound, not a deployable "
        "setting -- see dsignal/prior_sweep.py",
    )
    ap.add_argument("--train-labels", default=None,
                    help="labels TSV the checkpoints were trained on; required by --target-prior")
    ap.add_argument("--tau", type=float, default=1.0, help="prior-correction strength")
    ap.add_argument(
        "--half", choices=("both", "dev", "test"), default="both",
        help="score only one side of benchmark/split.tsv. Every tuning decision is "
        "read off 'dev'; 'test' is scored once per milestone and reported separately. "
        "'both' reproduces the historical 140-event numbers",
    )
    ap.add_argument(
        "--tta-condition-swap", action="store_true",
        help="average the prediction over both condition orientations. The swap "
        "is exactly label-invariant (the label is the event type, not the "
        "direction of the change), so this is free test-time augmentation; it "
        "doubles inference time and leaves extraction untouched",
    )
    ap.add_argument("--split", default="benchmark/split.tsv")
    args = ap.parse_args(argv)

    correction = None
    if args.target_prior != "none":
        if not args.train_labels:
            ap.error("--target-prior needs --train-labels to count the training class prior")

    predictor = Predictor(
        args.net,
        device=args.device,
        filters=ReadFilters(min_anchor=args.min_anchor, max_nh=args.max_nh, min_mapq=args.min_mapq),
        tta_condition_swap=args.tta_condition_swap,
    )
    if args.target_prior != "none":
        train_counts = class_counts_from_labels(args.train_labels, predictor.classes)
        correction = PriorCorrection(
            train_counts, make_target_prior(predictor.classes, args.target_prior), args.tau
        )
        predictor.prior_correction = correction
        print(f"prior correction: {correction}", file=sys.stderr)
    print(f"loaded {len(predictor.models)} fold checkpoint(s), classes {predictor.classes}, "
          f"feature subset {predictor.feature_subset!r}", file=sys.stderr)

    keep = None
    if args.half != "both":
        from dsignal.benchmark_split import load_split  # local: benchmark_split imports us

        split = load_split(args.split)
        keep = lambda stem, region: split.get((stem, region)) == args.half  # noqa: E731
        print(f"scoring the {args.half} half of {args.split}", file=sys.stderr)

    rows = score(predictor, args.csv, args.verbose, keep=keep)
    report(rows, predictor, args.half)

    if args.out:
        fields = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote per-event results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
