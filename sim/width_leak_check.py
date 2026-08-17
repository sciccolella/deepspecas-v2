"""Window-width leak test: how much of the label can be predicted from the
extraction window's *width* alone?

This is the narrowest possible probe of the window sampler. It reads widths
straight off the `.npz` filenames (`{ix}.{chrom}_{start}_{end}.npz`), so it
touches no coverage, no junctions, and no model code.

**Read this number as a measurement, not as a pass/fail.** Its meaning changed
when the window sampler did:

    sampler   set                   n     acc / macro-F1   majority-class floor
    v1        npz_scale1260      1,260    0.457 / 0.235    --
    v2        npz_windowfix      1,260    0.432 / 0.199    0.429 / 0.120
    v2        npz_scaleup        4,800    0.279 / 0.221    0.250 / 0.080
    v3        npz_geomwin        4,800    0.349 / 0.293    0.250 / 0.080

Only the last two rows are like-for-like -- the 1,260-event sets carry a 43% NN
design and therefore a much higher majority-class floor. Compare within a
column, never across the n boundary.

Under v1 and v2 the goal was for this to sit at chance, and under v2 it did (on
accuracy). Under **v3 it is expected to rise** and does: the window is now
deliberately the event span plus a small flank, because that is what the curated
benchmark's windows measurably are, and event span tracks the label. See
`sim/window.py` and `dsignal/baseline_gbm.py` for why that trade was accepted --
briefly, the same coupling exists in the deployment distribution, where window
geometry alone recovers 13 of 17 curated IR events.

So a *rise* here is no longer evidence of a bug; an unexplained rise still is.
Re-run after any change to the window sampler or the design's region definitions
and compare against the row for the sampler you are running.

Usage:
    python -m sim.width_leak_check --npz-dir sim/experiment/npz_geomwin \
        --labels sim/experiment/labels_geomwin.tsv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np


def widths_from_npz_dir(npz_dir: str) -> dict:
    """ix -> window width in bp, parsed from the extractor's output filenames."""
    widths = {}
    for name in os.listdir(npz_dir):
        if not name.endswith(".npz"):
            continue
        ix, region = name[: -len(".npz")].split(".", 1)
        _, start, end = region.rsplit("_", 2)
        widths[ix] = int(end) - int(start) + 1
    return widths


def main(argv=None) -> None:
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    widths = widths_from_npz_dir(args.npz_dir)
    X, y = [], []
    with open(args.labels) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["ix"] in widths:
                X.append(np.log10(widths[row["ix"]]))
                y.append(row["label"])
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    y = np.asarray(y)

    per = collections.defaultdict(list)
    for w, label in zip(10 ** X[:, 0], y):
        per[label].append(w)
    print(f"{len(y)} events, window width by label:")
    for label in sorted(per):
        a = np.asarray(per[label])
        print(f"  {label:4s} n={len(a):5d}  median {np.median(a):9.0f}  q90 {np.percentile(a, 90):10.0f}")
    print(f"pooled: median {10 ** np.median(X):.0f}  log10 mean {X.mean():.3f} sd {X.std(ddof=1):.3f}"
          f"   (pooled curated benchmark: 3063 / 3.445 / 0.702)")

    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.seed), X, y, cv=cv)
    print(f"\nwidth-only decision tree (depth {args.max_depth}, {args.folds}-fold): "
          f"acc {accuracy_score(y, pred):.3f}  macro-F1 {f1_score(y, pred, average='macro', zero_division=0):.3f}")

    print("chance references:")
    for strategy in ("most_frequent", "stratified", "uniform"):
        accs, f1s = [], []
        for seed in range(20):
            p = DummyClassifier(strategy=strategy, random_state=seed).fit(X, y).predict(X)
            accs.append(accuracy_score(y, p))
            f1s.append(f1_score(y, p, average="macro", zero_division=0))
        print(f"  {strategy:14s} acc {np.mean(accs):.3f}  macro-F1 {np.mean(f1s):.3f}")

    print()
    print(classification_report(y, pred, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
