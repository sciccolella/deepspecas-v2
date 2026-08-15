"""LightGBM baseline over the shared junction-topology features (Phase 3 step 2,
top-level plan item 3.1).

Same data, same protocol, same reporting format as ``sim/baseline_model.py`` --
``GroupKFold(5)`` grouped by gene with an ``assert_no_gene_leak`` check -- so its
numbers drop straight into the standing comparison table. The *only* thing that
changes is the feature set: `sim/baseline_model.py`'s 24 crude summary features
(junction count, coverage moments, ratios) are replaced by
``dsignal.features``' topology + coverage-context view.

**This is the honest-reporting gate.** Two outcomes both matter and both get
reported as they come out:

* if topology features alone lift A3/A5 F1 materially above the standing
  reference (0.108 / 0.160), that confirms junction *identity* -- not model
  capacity -- is the missing ingredient, which is the paper's central claim;
* if this baseline approaches the eventual Transformer, that has to be said.

**Window geometry is a simulation artifact and is excluded by default.** The
plan's geometry-only control (``--feature-subset geometry``) was supposed to
score at chance; it does not -- window geometry alone reaches 46.5% accuracy /
0.369 macro-F1 on npz_scale1260, because ``sim/window.py`` draws each window's
flank log-uniform in [0.1x, 10x] the *event span*, leaving median window width
running 1,452bp (RI) to 22,734bp (NN), a 15x spread that tracks the label. That
is a Phase-2 bug, not a Phase-3 one, and it would not transfer to the real
benchmark, whose windows come from a different process entirely. So
``--feature-subset`` defaults to ``no-geometry``: it is the number that is safe
to report. The topology result survives the exclusion nearly intact (macro-F1
0.621 without geometry vs 0.640 with), which is itself the evidence that
topology, not geometry, is doing the work.

Usage:
    python -m dsignal.baseline_gbm --npz-dir sim/experiment/npz_scale1260 \
        --labels sim/experiment/labels_scale1260.tsv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from os.path import join as pjoin

import numpy as np

from dsignal.extract import load_record
from dsignal.features import FeatureConfig, build_all, gbm_feature_names
from sim.gene_split import assert_no_gene_leak, gene_id_of

#: standing reference from sim/baseline_model.py on npz_scale1260, printed
#: alongside every run so the comparison is never done from memory
REFERENCE_RF = {
    "accuracy": 0.488,
    "macro_f1": 0.325,
    "per_class_f1": {"A3": 0.108, "A5": 0.160, "CSE": 0.132, "NN": 0.657, "RI": 0.568},
}


#: Window-geometry columns -- where the event sits in its extraction window and
#: how wide that window is. sim/window.py draws each window's flank log-uniform
#: in [0.1x, 10x] the *event span*, so window width is correlated with event
#: span, which is correlated with label (RI introns are short, CSE spans are
#: not). That makes geometry a potential simulation-artifact shortcut which
#: would not transfer to the real benchmark. The top-level plan's Verification
#: section calls for the geometry-only control; --feature-subset runs it, and
#: `no-geometry` runs the complement (the number that is actually safe to
#: report as the topology result).
_GEOMETRY_EVENT_FEATURES = ("log10_window_width", "frac_junctions_out_of_window")
_GEOMETRY_TOKEN_FEATURES = ("rel_start", "rel_end", "rel_mid", "in_window")


def geometry_columns(feat_names: list[str]) -> np.ndarray:
    """Boolean mask over gbm_feature_names() selecting geometry columns
    (the event-level ones plus every aggregate of a geometry token column)."""
    mask = np.zeros(len(feat_names), dtype=bool)
    for i, name in enumerate(feat_names):
        base = name.split("__", 1)[1] if "__" in name else name
        if name in _GEOMETRY_EVENT_FEATURES or base in _GEOMETRY_TOKEN_FEATURES:
            mask[i] = True
    return mask


def apply_feature_subset(X: np.ndarray, feat_names: list[str], subset: str) -> np.ndarray:
    """Zero out columns rather than dropping them, so feature indices (and
    therefore the importance printout) stay aligned across subsets."""
    if subset == "all":
        return X
    mask = geometry_columns(feat_names)
    X = X.copy()
    if subset == "geometry":
        X[:, ~mask] = 0.0
        print(f"feature subset 'geometry': kept {int(mask.sum())} geometry columns, "
              f"zeroed {int((~mask).sum())} junction/coverage columns", file=sys.stderr)
    elif subset == "no-geometry":
        X[:, mask] = 0.0
        print(f"feature subset 'no-geometry': zeroed {int(mask.sum())} geometry columns",
              file=sys.stderr)
    else:
        raise ValueError(f"unknown feature subset {subset!r}")
    return X


def _npz_index(npz_dir: str) -> dict:
    """ix prefix -> path. Built once; sim/baseline_model.py re-lists the
    directory per row, which is O(n^2) and noticeable at scale."""
    index = {}
    for name in os.listdir(npz_dir):
        if name.endswith(".npz"):
            index[name.split(".", 1)[0]] = pjoin(npz_dir, name)
    return index


def build_dataset(npz_dir: str, labels_path: str, cfg: FeatureConfig | None = None):
    """(X [n, F], y [n], event_ids, gene_ids) over every labelled record."""
    cfg = cfg or FeatureConfig()
    with open(labels_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    index = _npz_index(npz_dir)

    X, y, event_ids, gene_ids = [], [], [], []
    for row in rows:
        path = index.get(row["ix"])
        if path is None:
            continue
        _, _, gbm_row, _ = build_all(load_record(path), cfg)
        X.append(gbm_row)
        y.append(row["label"])
        event_ids.append(row["event_id"])
        gene_ids.append(gene_id_of(row["event_id"]))

    return np.asarray(X, dtype=np.float32), np.asarray(y), event_ids, gene_ids


def grouped_folds(X, y, gene_ids, n_folds: int, verbose: bool = True):
    """GroupKFold fold assignment, with the split-integrity check the
    top-level plan's Verification section requires."""
    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=n_folds)
    splits = list(cv.split(X, y, groups=gene_ids))
    fold_of = {}
    for fold, (_, test_idx) in enumerate(splits):
        for i in test_idx:
            fold_of[gene_ids[i]] = fold
    assert_no_gene_leak(
        [dict(gene_id=g, split=fold_of[g]) for g in gene_ids],
        gene_key="gene_id",
        split_key="split",
    )
    if verbose:
        print(f"gene-grouped {n_folds}-fold CV over {len(set(gene_ids))} genes: no gene leak", file=sys.stderr)
    return splits


def report(y_true, y_pred, labels_sorted, title: str) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, labels=labels_sorted, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("confusion matrix (rows=true, cols=pred):", labels_sorted)
    print(cm)

    per_class = f1_score(y_true, y_pred, labels=labels_sorted, average=None, zero_division=0)
    out = {
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {l: float(f) for l, f in zip(labels_sorted, per_class)},
    }
    return out


def report_collapsed(y_true, y_pred, title: str) -> dict:
    """A3+A5 -> A, for comparability with v1's 123/140 benchmark number."""
    collapse = np.vectorize(lambda l: "A" if l in ("A3", "A5") else l)
    yt, yp = collapse(np.asarray(y_true)), collapse(np.asarray(y_pred))
    return report(yt, yp, sorted(set(yt.tolist())), title)


def print_vs_reference(result: dict) -> None:
    print("\nvs sim/baseline_model.py random forest (same data, same protocol):")
    print(f"  {'metric':10s} {'this':>8s} {'RF ref':>8s} {'delta':>8s}")
    print(f"  {'accuracy':10s} {result['accuracy']:8.3f} {REFERENCE_RF['accuracy']:8.3f} "
          f"{result['accuracy'] - REFERENCE_RF['accuracy']:+8.3f}")
    print(f"  {'macro-F1':10s} {result['macro_f1']:8.3f} {REFERENCE_RF['macro_f1']:8.3f} "
          f"{result['macro_f1'] - REFERENCE_RF['macro_f1']:+8.3f}")
    for label, ref in REFERENCE_RF["per_class_f1"].items():
        got = result["per_class_f1"].get(label)
        if got is None:
            continue
        print(f"  {'F1 ' + label:10s} {got:8.3f} {ref:8.3f} {got - ref:+8.3f}")
    print("\n  A3/A5 F1 is the metric that matters -- that is the class pair the")
    print("  topology features exist to fix.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flank", type=int, default=50, help="coverage-context flank width (bp)")
    ap.add_argument("--max-junctions", type=int, default=None)
    ap.add_argument("--no-canonicalize", action="store_true",
                    help="skip minus-strand canonicalization (ablation: A3/A5 should get worse)")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument(
        "--feature-subset",
        choices=("all", "geometry", "no-geometry"),
        default="no-geometry",
        help="default 'no-geometry' zeroes the window-geometry columns, which "
        "carry a real simulation artifact (see module docstring); 'geometry' is "
        "the plan's geometry-only control; 'all' keeps everything",
    )
    ap.add_argument("--importances", type=int, default=20)
    ap.add_argument("--model-out", default=None, help="save a full-data LightGBM model here")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    cfg = FeatureConfig(
        flank=args.flank,
        max_junctions=args.max_junctions,
        canonicalize=not args.no_canonicalize,
    )
    X, y, event_ids, gene_ids = build_dataset(args.npz_dir, args.labels, cfg)
    feat_names = gbm_feature_names()
    X = apply_feature_subset(X, feat_names, args.feature_subset)
    print(f"loaded {len(y)} events, {X.shape[1]} features", file=sys.stderr)
    labels_sorted = sorted(set(y.tolist()))
    print("class counts:", {l: int((y == l).sum()) for l in labels_sorted})
    print(f"distinct genes: {len(set(gene_ids))}", file=sys.stderr)

    splits = grouped_folds(X, y, gene_ids, args.folds)

    label_to_int = {l: i for i, l in enumerate(labels_sorted)}
    y_int = np.array([label_to_int[l] for l in y])

    y_pred = np.empty(len(y), dtype=object)
    for fold, (train_idx, test_idx) in enumerate(splits):
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(labels_sorted),
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X[train_idx], y_int[train_idx])
        pred = clf.predict(X[test_idx])
        for i, p in zip(test_idx, pred):
            y_pred[i] = labels_sorted[int(p)]
        print(f"  fold {fold}: {len(train_idx)} train / {len(test_idx)} held out", file=sys.stderr)

    result = report(
        y,
        y_pred,
        labels_sorted,
        f"lightgbm [{args.feature_subset}] ({args.folds}-fold group CV, grouped by gene)",
    )
    print_vs_reference(result)
    report_collapsed(y, y_pred, "lightgbm, A3+A5 collapsed to A (comparable to v1's 123/140)")

    # gain importances from a full-data fit -- for inspection only, not a
    # held-out quantity
    full = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(labels_sorted),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
        importance_type="gain",
    )
    full.fit(X, y_int)
    order = np.argsort(full.feature_importances_)[::-1]
    print(f"\ntop {args.importances} gain importances (full-data fit, for inspection only):")
    for i in order[: args.importances]:
        print(f"  {feat_names[i]:45s} {full.feature_importances_[i]:12.1f}")

    if args.model_out:
        full.booster_.save_model(args.model_out)
        print(f"\nwrote full-data model to {args.model_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
