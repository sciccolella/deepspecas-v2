"""Trivial hand-engineered-feature sanity check for the pilot .npz records
(checkpoint step 3: "score a first-pass model, even something trivial,
against the extracted pilot records to sanity-check the label signal is
learnable at all, before scaling up"). Not the eventual signal-native
architecture -- just confirms the raw junction/coverage signal separates
A3/A5/CSE/RI/NN above chance before investing in a real model.

Usage:
    python -m sim.baseline_model --npz-dir sim/experiment/npz \
        --labels sim/experiment/labels.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from os.path import join as pjoin

import numpy as np

from sim.gene_split import assert_no_gene_leak, gene_id_of

FEATURE_NAMES = [
    "n_junctions",
    "top_junction_frac",
    "second_junction_frac",
    "junction_count_ratio",
    "cov_mean",
    "cov_cv",
    "zero_cov_frac",
    "skip_to_cov_ratio",
]


def _condition_features(cov: np.ndarray, junctions: np.ndarray, stats: dict) -> np.ndarray:
    n_junctions = len(junctions)
    weights = np.sort(junctions["nh_weighted_count"])[::-1] if n_junctions else np.array([])
    total_w = weights.sum() if n_junctions else 0.0

    top_frac = weights[0] / total_w if n_junctions >= 1 and total_w > 0 else 0.0
    second_frac = weights[1] / total_w if n_junctions >= 2 and total_w > 0 else 0.0
    ratio = weights[1] / weights[0] if n_junctions >= 2 and weights[0] > 0 else 0.0

    cov_mean = float(cov.mean()) if len(cov) else 0.0
    cov_std = float(cov.std()) if len(cov) else 0.0
    cov_cv = cov_std / cov_mean if cov_mean > 0 else 0.0
    zero_frac = float((cov == 0).mean()) if len(cov) else 0.0
    skip_to_cov = stats["skip_max"] / cov_mean if cov_mean > 0 else 0.0

    return np.array(
        [n_junctions, top_frac, second_frac, ratio, cov_mean, cov_cv, zero_frac, skip_to_cov]
    )


def extract_features(record: dict) -> np.ndarray:
    f1 = _condition_features(record["cov_1"], record["junctions_1"], record["stats_1"])
    f2 = _condition_features(record["cov_2"], record["junctions_2"], record["stats_2"])
    diff = f1 - f2
    return np.concatenate([f1, f2, diff])


def load_record(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "cov_1": data["cov_1"],
        "junctions_1": data["junctions_1"],
        "stats_1": json.loads(str(data["stats_1_json"])),
        "cov_2": data["cov_2"],
        "junctions_2": data["junctions_2"],
        "stats_2": json.loads(str(data["stats_2_json"])),
    }


def build_dataset(npz_dir: str, labels_path: str):
    with open(labels_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    X, y, event_ids, gene_ids = [], [], [], []
    for row in rows:
        ix = row["ix"]
        matches = [p for p in __import__("os").listdir(npz_dir) if p.startswith(ix + ".")]
        if not matches:
            continue
        record = load_record(pjoin(npz_dir, matches[0]))
        X.append(extract_features(record))
        y.append(row["label"])
        event_ids.append(row["event_id"])
        gene_ids.append(gene_id_of(row["event_id"]))

    return np.array(X), np.array(y), event_ids, gene_ids


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    X, y, event_ids, gene_ids = build_dataset(args.npz_dir, args.labels)
    print(f"loaded {len(y)} events, {X.shape[1]} features", file=__import__("sys").stderr)
    labels_sorted = sorted(set(y))
    print("class counts:", {l: int((y == l).sum()) for l in labels_sorted})
    print(f"distinct genes: {len(set(gene_ids))}", file=__import__("sys").stderr)

    # plan Phase 2 step 7 / split-integrity verification: folds are grouped
    # by gene_id so no gene's events can straddle train and held-out --
    # StratifiedKFold alone doesn't guarantee that once a gene can carry more
    # than one event (true once the pilot's one-event-per-gene constraint is
    # lifted at scale-up).
    cv = GroupKFold(n_splits=args.folds)
    fold_of = {}
    for fold, (_, test_idx) in enumerate(cv.split(X, y, groups=gene_ids)):
        for i in test_idx:
            fold_of[gene_ids[i]] = fold
    assert_no_gene_leak(
        [dict(gene_id=g, split=fold_of[g]) for g in gene_ids], gene_key="gene_id", split_key="split"
    )

    for name, clf in [
        ("logistic_regression", LogisticRegression(max_iter=2000)),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=args.seed)),
    ]:
        if name == "logistic_regression":
            from sklearn.pipeline import make_pipeline

            model = make_pipeline(StandardScaler(), clf)
        else:
            model = clf

        y_pred = cross_val_predict(model, X, y, cv=cv, groups=gene_ids)
        print(f"\n=== {name} ({args.folds}-fold group CV, grouped by gene) ===")
        print(classification_report(y, y_pred, labels=labels_sorted, digits=3))
        cm = confusion_matrix(y, y_pred, labels=labels_sorted)
        print("confusion matrix (rows=true, cols=pred):", labels_sorted)
        print(cm)

    # feature importance from the random forest, fit on the full pilot set
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed)
    rf.fit(X, y)
    feat_labels = [f"{n}_1" for n in FEATURE_NAMES] + [f"{n}_2" for n in FEATURE_NAMES] + [
        f"{n}_diff" for n in FEATURE_NAMES
    ]
    order = np.argsort(rf.feature_importances_)[::-1]
    print("\ntop 10 feature importances (full-pilot fit, for inspection only):")
    for i in order[:10]:
        print(f"  {feat_labels[i]:25s} {rf.feature_importances_[i]:.4f}")


if __name__ == "__main__":
    main()
