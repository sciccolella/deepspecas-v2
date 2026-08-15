"""Does the zero-shot sim->real gap come from a class-prior mismatch?
(Phase 4 probe; plan: plans/signal-native-v2-prior-correction.md)

The dominant benchmark error is a false NN prediction -- 41/140 under the fixed
windows, 29/140 under the old ones -- and NN is 43% of the training design
(540/1260) against a benchmark that contains **no NN events at all**. This
sweeps `dsignal.predict.PriorCorrection` over that mismatch, on already-trained
checkpoints: no retraining, no simulation.

Each benchmark event is extracted and run through the folds **once**; the
per-fold posteriors are cached and every prior setting is then evaluated
analytically from them. So the sweep costs one benchmark pass, not one per
setting.

Read the output as a diagnosis, not a score:

* a large jump on **both** checkpoint sets means the prior mismatch was always
  the dominant error and the window fix is transfer-neutral;
* a jump on only one means the window change moved NN's decision geometry;
* no movement means the errors are genuine and Phase 4 needs representation
  adaptation, not calibration.

`nn_weight=0` is an **oracle upper bound**, not a deployable setting: it
presumes the truth set contains no negatives, which is exactly what makes the
benchmark's NN column empty. The deployable setting is `uniform`.

Usage:
    python -m dsignal.prior_sweep --net runs/windowfix_allaug/fold*.pt \
        --train-labels sim/experiment/labels_windowfix.tsv
"""

from __future__ import annotations

import argparse
import sys
from os.path import basename, splitext

import numpy as np

from dsignal.extract import ReadFilters, extract_event, parse_region
from dsignal.predict import (
    Predictor,
    PriorCorrection,
    class_counts_from_labels,
    make_target_prior,
    read_prediction_csv,
)
from dsignal.score_benchmark import CSV_LABEL, DEFAULT_CSVS, collapse

#: NN prior weights to trace, 1.0 (no correction) down to 0.0 (oracle).
DEFAULT_NN_WEIGHTS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.0)


def cache_fold_probs(predictor: Predictor, csv_paths, verbose: bool = False):
    """[(csv, region, truth, [F, C] per-fold posteriors)] over the benchmark."""
    cached = []
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r}")
        truth = CSV_LABEL[stem]
        for region, bam1, bam2 in read_prediction_csv(path):
            record = extract_event(parse_region(region), bam1, bam2, filters=predictor.filters)
            per_fold, _ = predictor.fold_probs(record)
            cached.append((stem, region, truth, per_fold))
            if verbose:
                print(f"  cached {region}", file=sys.stderr)
    return cached


def evaluate(predictor: Predictor, cached, correction: PriorCorrection | None):
    """(predictions, accuracy, collapsed accuracy) under one correction."""
    saved = predictor.prior_correction
    predictor.prior_correction = correction
    try:
        preds = [predictor.combine(per_fold)[0] for *_, per_fold in cached]
    finally:
        predictor.prior_correction = saved
    truth = [t for _, _, t, _ in cached]
    acc = float(np.mean([p == t for p, t in zip(preds, truth)]))
    acc_c = float(np.mean([collapse(p) == collapse(t) for p, t in zip(preds, truth)]))
    return preds, acc, acc_c


def mcnemar(base_preds, new_preds, truth):
    """(n_base_only_correct, n_new_only_correct, exact p). Paired over the same
    events, so this is the right test for 'did the correction change anything'
    -- an unpaired proportion test would ignore that both runs score the same
    140 events."""
    b = sum(1 for p0, p1, t in zip(base_preds, new_preds, truth) if p0 == t and p1 != t)
    c = sum(1 for p0, p1, t in zip(base_preds, new_preds, truth) if p0 != t and p1 == t)
    if b + c == 0:
        return b, c, 1.0
    try:
        from scipy.stats import binomtest

        p = float(binomtest(min(b, c), b + c, 0.5).pvalue)
    except ImportError:  # scipy is not a hard dependency of this repo
        p = float("nan")
    return b, c, p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", nargs="+", required=True, help="trained fold checkpoints (.pt)")
    ap.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    ap.add_argument(
        "--train-labels",
        required=True,
        help="labels TSV of the set these checkpoints were trained on; the "
        "training class prior is counted from it (checkpoints predating "
        "train_class_counts do not carry it)",
    )
    ap.add_argument("--tau", type=float, default=1.0, help="correction strength; 0 is a no-op")
    ap.add_argument("--nn-weights", type=float, nargs="+", default=list(DEFAULT_NN_WEIGHTS))
    ap.add_argument("--out", default=None, help="per-setting TSV")
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
    classes = list(predictor.classes)
    train_counts = class_counts_from_labels(args.train_labels, classes)
    print(f"loaded {len(predictor.models)} fold checkpoint(s), classes {tuple(classes)}, "
          f"feature subset {predictor.feature_subset!r}", file=sys.stderr)
    print(f"training prior from {args.train_labels}: "
          + ", ".join(f"{c} {int(n)}" for c, n in zip(classes, train_counts))
          + f"  (NN = {train_counts[classes.index('NN')] / train_counts.sum():.1%})", file=sys.stderr)

    cached = cache_fold_probs(predictor, args.csv, args.verbose)
    truth = [t for _, _, t, _ in cached]
    n = len(cached)
    print(f"cached per-fold posteriors for {n} benchmark events", file=sys.stderr)

    base_preds, base_acc, base_acc_c = evaluate(predictor, cached, None)

    settings = [("uncorrected", None)]
    settings.append(("uniform", PriorCorrection(train_counts, make_target_prior(classes, "uniform"), args.tau)))
    for w in args.nn_weights:
        label = f"nn_weight={w:g}" + ("  (oracle)" if w == 0 else "")
        target = make_target_prior(classes, "nn-weight", nn_weight=w)
        settings.append((label, PriorCorrection(train_counts, target, args.tau)))

    print(f"\n=== prior sweep, tau={args.tau:g}, {n} curated events ===")
    header = f"{'setting':24s} {'acc':>7s} {'collapsed':>10s} {'NN pred':>8s} " \
             f"{'base>new':>9s} {'new>base':>9s} {'McNemar p':>10s}"
    print(header)
    print("-" * len(header))
    out_rows = []
    for label, correction in settings:
        preds, acc, acc_c = evaluate(predictor, cached, correction)
        n_nn = sum(1 for p in preds if p == "NN")
        b, c, p = mcnemar(base_preds, preds, truth)
        print(f"{label:24s} {acc:7.1%} {acc_c:10.1%} {n_nn:8d} {b:9d} {c:9d} {p:10.3f}")
        out_rows.append(dict(setting=label, accuracy=acc, accuracy_collapsed=acc_c,
                             n_nn_pred=n_nn, base_only_correct=b, new_only_correct=c, mcnemar_p=p))

    # Where do the recovered (and newly lost) events go? A correction that only
    # converts false NN into a different wrong class is not a fix.
    best = max(out_rows, key=lambda r: r["accuracy"])
    best_correction = dict(settings)[best["setting"]]
    best_preds, _, _ = evaluate(predictor, cached, best_correction)
    print(f"\nbest setting: {best['setting']}  ({best['accuracy']:.1%} vs {base_acc:.1%} uncorrected)")
    print("per-CSV accuracy, uncorrected -> best:")
    for stem in ("A3", "A5", "ES", "IR"):
        idx = [i for i, (s, *_rest) in enumerate(cached) if s == stem]
        if not idx:
            continue
        a0 = np.mean([base_preds[i] == truth[i] for i in idx])
        a1 = np.mean([best_preds[i] == truth[i] for i in idx])
        print(f"  {stem:3s} (truth {CSV_LABEL[stem]:4s}) n={len(idx):3d}  {a0:6.1%} -> {a1:6.1%}")

    moved = [(truth[i], base_preds[i], best_preds[i]) for i in range(n) if base_preds[i] != best_preds[i]]
    print(f"\n{len(moved)} predictions moved. Where the previously-NN calls went:")
    from collections import Counter

    went = Counter(f"{t} was NN -> {new}{'  [correct]' if new == t else ''}"
                   for t, old, new in moved if old == "NN")
    for k, v in went.most_common():
        print(f"  {v:3d}  {k}")
    other = Counter(f"{t}: {old} -> {new}" for t, old, new in moved if old != "NN")
    if other:
        print("non-NN changes (a correction should not be churning these):")
        for k, v in other.most_common(10):
            print(f"  {v:3d}  {k}")

    if args.out:
        import csv as _csv

        with open(args.out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
