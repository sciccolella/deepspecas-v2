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

**Window geometry is excluded by default -- but the reason has changed, and the
default has not been revisited since.** Read this whole note before touching
``--feature-subset``.

*Originally (window sampler v1, ``npz_scale1260``):* the plan's geometry-only
control was supposed to score at chance and did not -- window geometry alone
reached 46.5% accuracy / 0.369 macro-F1, because ``sim/window.py`` then drew
each window's flank log-uniform in [0.1x, 10x] the *event span*, leaving median
window width running 1,452bp (RI) to 22,734bp (NN), a 15x spread tracking the
label. That was a pure simulation artifact with no counterpart at inference, so
``--feature-subset`` defaulted to ``no-geometry`` as the number safe to report.
The topology result survived the exclusion nearly intact (macro-F1 0.621
without geometry vs 0.640 with), which was itself the evidence that topology,
not geometry, was doing the work.

*Now (window sampler v3):* the window is the event span plus U[0, 100] bp per
edge, because that is measurably what the curated benchmark's own windows are
(``python -m dsignal.audit_noise padding``; see ``sim/window.py``). This
knowingly reintroduces a width->label coupling -- geometry-only sim-CV rose to
0.446 here and 0.486 for the Transformer -- **but the coupling is now partly
shared with the deployment distribution rather than being an artifact**. Scored
on the *real* benchmark, the geometry-only control reaches 31.4% 5-class / 35.0%
collapsed (above a ~20% random floor, below the 46.4% always-CSE baseline), and
recovers 13 of 17 curated IR events from geometry alone. So: not a strong
general signal, but a real RI one.

The default is still ``no-geometry``, which zeroes those columns outright.

*Measured (2026-08-17, ``runs/geomwin_allfeat.*``):* ``--feature-subset all``
under v3 windows buys sim-CV and nothing else. sim-CV macro-F1 0.867 -> **0.873**,
benchmark dev half 55/70 -> **52/70** collapsed (78.6% -> 74.3%, 7 lost / 4
gained, McNemar p = 0.549 -- a null, and the sign is wrong). The loss is
concentrated in RI: 6/8 -> 3/8 on the dev half, with two RI events going to A5
and one to A3. So geometry does not transfer *additively*: geometry alone
recovers 13/17 curated IR events, but handing the model geometry alongside
topology moves it onto the simulated width->label coupling, which is only
partly the real one. The conservative default stands, now on evidence rather
than on caution.

Usage:
    python -m dsignal.baseline_gbm --npz-dir sim/experiment/npz_geomwin \
        --labels sim/experiment/labels_geomwin.tsv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from os.path import join as pjoin

import numpy as np

from dsignal.extract import load_record
from dsignal.features import EXTRA_EVENT_GROUPS, FeatureConfig, build_all, gbm_feature_names, parse_feature_subset
from sim.gene_split import assert_no_gene_leak, gene_id_of

#: Standing reference from sim/baseline_model.py, printed alongside every run so
#: the comparison is never done from memory.
#:
#: **Provenance matters here:** these numbers are from npz_scale1260 -- 1,260
#: events under window sampler *v1*. They are kept as the fixed yardstick the
#: whole layer-by-layer history was measured against, not as a current baseline.
#: Anything trained on npz_geomwin is a different data distribution, so treat
#: this as "the crude-feature floor as of the pilot", not as a like-for-like
#: control.
REFERENCE_RF = {
    "accuracy": 0.488,
    "macro_f1": 0.325,
    "per_class_f1": {"A3": 0.108, "A5": 0.160, "CSE": 0.132, "NN": 0.657, "RI": 0.568},
}


#: Window-geometry columns -- where the event sits in its extraction window and
#: how wide that window is. Under window sampler v3 these are correlated with
#: the label because the window *is* the event span plus a small flank, and
#: event span tracks the label (RI introns are short, CSE spans are not).
#: See the module docstring: that coupling is now partly real rather than
#: purely artifactual, so `--feature-subset geometry` is a measurement of how
#: much of the task geometry alone can do, not a bug detector any more.
#: `no-geometry` runs the complement and remains the default.
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
    base, drop = parse_feature_subset(subset)
    if base == "all" and not drop:
        return X
    mask = geometry_columns(feat_names)
    X = X.copy()
    if drop:
        # Stage 18.5 event groups (event-level only; they have no token aggregates)
        cols = [i for i, n in enumerate(feat_names) if any(n in EXTRA_EVENT_GROUPS[g] for g in drop)]
        X[:, cols] = 0.0
        print(f"feature subset: zeroed {len(cols)} columns of groups {list(drop)}", file=sys.stderr)
    subset = base
    if subset == "geometry":
        # Only the *event-level* geometry features survive. The token-aggregate
        # geometry columns (max/mean/wmean of rel_start/rel_end/rel_mid/
        # in_window) summarize where this record's junctions sit, which is
        # junction structure rather than window geometry -- keeping them made
        # the control score well above width alone. This matches the
        # Transformer control in dsignal/dataset.py, whose dummy token set
        # makes every token aggregate identically zero.
        keep = np.array([n in _GEOMETRY_EVENT_FEATURES for n in feat_names])
        X[:, ~keep] = 0.0
        print(f"feature subset 'geometry': kept {int(keep.sum())} event-level geometry "
              f"columns, zeroed {int((~keep).sum())} junction/coverage/token-aggregate "
              f"columns", file=sys.stderr)
    elif subset == "no-geometry":
        X[:, mask] = 0.0
        print(f"feature subset 'no-geometry': zeroed {int(mask.sum())} geometry columns",
              file=sys.stderr)
    return X


def _npz_index(npz_dir: str) -> dict:
    """ix prefix -> path. Built once; sim/baseline_model.py re-lists the
    directory per row, which is O(n^2) and noticeable at scale."""
    index = {}
    for name in os.listdir(npz_dir):
        if name.endswith(".npz"):
            index[name.split(".", 1)[0]] = pjoin(npz_dir, name)
    return index


def build_dataset(npz_dir: str, labels_path: str, cfg: FeatureConfig | None = None,
                  return_meta: bool = False):
    """(X [n, F], y [n], event_ids, gene_ids) over every labelled record.

    With ``return_meta`` a fifth element is appended: one dict per row with
    the labels TSV's ``kind`` (provenance, "" when the file has no such
    column) and the record's own ``chrom``/``start``/``end`` -- what
    ``dsignal.domain_probe`` clusters on, since gene ids are not consistent
    across real cohorts."""
    cfg = cfg or FeatureConfig()
    with open(labels_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    index = _npz_index(npz_dir)

    X, y, event_ids, gene_ids, meta = [], [], [], [], []
    for row in rows:
        path = index.get(row["ix"])
        if path is None:
            continue
        record = load_record(path)
        _, _, gbm_row, _ = build_all(record, cfg)
        X.append(gbm_row)
        y.append(row["label"])
        event_ids.append(row["event_id"])
        gene_ids.append(gene_id_of(row["event_id"]))
        if return_meta:
            meta.append({"kind": row.get("kind", ""), "ix": row["ix"],
                         "chrom": record["chrom"], "start": record["start"], "end": record["end"]})

    out = (np.asarray(X, dtype=np.float32), np.asarray(y), event_ids, gene_ids)
    return out + (meta,) if return_meta else out


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
    ap.add_argument("--event-groups", nargs="*", default=None, choices=list(EXTRA_EVENT_GROUPS),
                    help="Stage 18.5 opt-in event-feature groups (covshape, delcov)")
    ap.add_argument(
        "--feature-subset",
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
        event_groups=tuple(args.event_groups or ()),
    )
    X, y, event_ids, gene_ids = build_dataset(args.npz_dir, args.labels, cfg)
    feat_names = gbm_feature_names(cfg)
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
