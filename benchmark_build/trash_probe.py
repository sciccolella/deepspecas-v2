#!/usr/bin/env python
"""Stage 9 / Stage 1: is TRASH-iness recoverable from the signal alone?

Joins ``dsignal.audit_noise measure``'s per-window metrics to
``benchmark_new/curation.tsv``'s ``trash`` annotation, on **dev** only, and
reports:

* per-metric AUC of ``trash``, over the whole dev pool and split by ``truth``
  class (the check that a metric is not simply re-encoding "this is an NN
  row");
* a 5-fold cluster-grouped logistic-regression AUC over the metric block, as
  an upper bound on how much of the annotation is signal-recoverable;
* the same metrics against v2's existing per-event correctness (the
  ``v2_pred`` column already carried in ``curation.tsv`` from the consensus
  review, collapsed A3+A5->A the same way ``score_benchmark`` does), to check
  whether the model's error concentrates where these metrics say it should.

**Discipline** (plan ``plans/signal-native-v2-trash-signal.md``): nothing
here is fit against the TRASH label and then used at inference time. This is
read-only measurement -- the logistic regression is diagnostic (an upper
bound on recoverability), not a model component. ``trash = NA`` rows are
dropped throughout: they are unlabelled along the axis being probed, and
counting them as negatives would inject label noise into the negative class.

Usage::

    python benchmark_build/trash_probe.py \\
        runs/noise_audit_bnew.tsv benchmark_new/curation.tsv \\
        --split benchmark_new/split.tsv --half dev
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

METRICS = [
    "intron_cov_frac", "background_frac", "antisense_frac", "dup_rate",
    "unsupported_junction_frac", "struct_span_frac", "cov_cv",
]
#: excluded from the model feature work (masked-off STAR bit, needs a BAM pass
#: to get right -- see the plan), but still reported here as a diagnostic.
DIAGNOSTIC = ["novel_junction_rate"]


def _f(x):
    try:
        v = float(x)
        return v if not np.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_curation(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if not int(r["ambiguous"])]
    by_region = {}
    for r in rows:
        if r["region"] in by_region:
            raise AssertionError(f"duplicate unambiguous region {r['region']!r} in {path}")
        by_region[r["region"]] = r
    return by_region


def build_table(noise_path, curation_path, split_path, half):
    from dsignal.benchmark_split import load_split
    from dsignal.score_benchmark import collapse

    curation = load_curation(curation_path)
    split = load_split(split_path) if half != "both" else None

    with open(noise_path, newline="") as f:
        noise_rows = list(csv.DictReader(f, delimiter="\t"))

    rows = []
    n_dropped_na, n_dropped_half, n_dropped_nomatch = 0, 0, 0
    for r in noise_rows:
        key = (r["csv"], r["region"])
        if split is not None and split.get(key) != half:
            n_dropped_half += 1
            continue
        c = curation.get(r["region"])
        if c is None:
            n_dropped_nomatch += 1
            continue
        if c["trash"] == "NA":
            n_dropped_na += 1
            continue

        row = {"csv": r["csv"], "region": r["region"], "truth": c["truth"],
               "trash": int(c["trash"])}
        n_j, n_s = _f(r["n_junctions"]), _f(r["n_supported"])
        row["unsupported_junction_frac"] = (1.0 - n_s / n_j) if n_j and n_j > 0 else 0.0
        for m in ("intron_cov_frac", "background_frac", "antisense_frac", "dup_rate",
                  "struct_span_frac", "cov_cv") + tuple(DIAGNOSTIC):
            row[m] = _f(r[m])

        if c["v2_pred"]:
            row["correct_v2"] = int(collapse(c["v2_pred"]) == collapse(c["truth"]))
        else:
            row["correct_v2"] = None
        rows.append(row)

    print(f"[{half}] {len(rows)} rows probed "
          f"({n_dropped_na} trash=NA dropped, {n_dropped_half} off-half, "
          f"{n_dropped_nomatch} unmatched to curation.tsv)", file=sys.stderr)
    return rows


def _auc(y, x):
    """ROC AUC, dropping NaN x. None if too degenerate to define."""
    y, x = np.asarray(y), np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    y, x = y[mask], x[mask]
    if len(set(y)) < 2 or mask.sum() < 4:
        return None, int(mask.sum())
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, x)), int(mask.sum())


def report_auc_table(rows, title):
    print(f"\n=== {title} (n={len(rows)}) ===")
    y = [r["trash"] for r in rows]
    if len(set(y)) < 2:
        print("  single class present -- AUC undefined")
        return
    for m in METRICS + DIAGNOSTIC:
        x = [r[m] for r in rows]
        auc, n = _auc(y, x)
        tag = " [diagnostic, not a feature candidate]" if m in DIAGNOSTIC else ""
        if auc is None:
            print(f"  {m:28s}  AUC undefined (n={n}){tag}")
        else:
            direction = "higher=TRASH" if auc >= 0.5 else "lower=TRASH"
            print(f"  {m:28s}  AUC={auc:.3f}  ({direction}, n={n}){tag}")


def cluster_grouped_logreg_auc(rows, csv_paths, link=50000, seed=0, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from dsignal.benchmark_split import _events, cluster

    events = _events(csv_paths)
    clusters = cluster(events, link)
    key_to_cluster = {}
    for ci, cl in enumerate(clusters):
        for i in cl:
            key_to_cluster[(events[i][0], events[i][1])] = ci

    X, y, groups = [], [], []
    for r in rows:
        key = (r["csv"], r["region"])
        if key not in key_to_cluster:
            continue
        X.append([r[m] for m in METRICS])
        y.append(r["trash"])
        groups.append(key_to_cluster[key])
    X, y, groups = np.array(X, dtype=float), np.array(y), np.array(groups)

    for j in range(X.shape[1]):
        col = X[:, j]
        nan = np.isnan(col)
        if nan.any():
            col[nan] = np.nanmedian(col) if not np.all(nan) else 0.0

    n_groups = len(set(groups))
    splits = min(n_splits, n_groups)
    if splits < 2 or len(set(y)) < 2:
        print(f"\n=== cluster-grouped logistic regression: not enough data "
              f"(n={len(y)}, groups={n_groups}) ===")
        return

    gkf = GroupKFold(n_splits=splits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    auc = roc_auc_score(y[mask], oof[mask])
    print(f"\n=== cluster-grouped {splits}-fold logistic regression over "
          f"{METRICS} ===")
    print(f"  pooled out-of-fold AUC = {auc:.3f}  (n={mask.sum()}, {n_groups} clusters)")
    print("  DECISION GATE: below ~0.70 means TRASH is not signal-recoverable "
          "at window level -- Stages 3-4 do not run.")


def report_vs_v2_correctness(rows):
    scored = [r for r in rows if r["correct_v2"] is not None]
    if not scored:
        print("\n=== vs v2 correctness: no v2_pred available ===")
        return
    print(f"\n=== metrics vs v2's existing per-event correctness (n={len(scored)}) ===")
    correct = [r for r in scored if r["correct_v2"]]
    wrong = [r for r in scored if not r["correct_v2"]]
    print(f"  v2 correct: {len(correct)}/{len(scored)} = {len(correct) / len(scored):.1%}")
    print(f"  {'metric':28s}  {'mean|correct':>14s}  {'mean|wrong':>14s}")
    for m in METRICS + DIAGNOSTIC:
        c_vals = [r[m] for r in correct if not np.isnan(r[m])]
        w_vals = [r[m] for r in wrong if not np.isnan(r[m])]
        c_mean = np.mean(c_vals) if c_vals else float("nan")
        w_mean = np.mean(w_vals) if w_vals else float("nan")
        print(f"  {m:28s}  {c_mean:14.3f}  {w_mean:14.3f}")

    trash_correct = np.mean([r["trash"] for r in correct]) if correct else float("nan")
    trash_wrong = np.mean([r["trash"] for r in wrong]) if wrong else float("nan")
    print(f"\n  mean(trash) | v2 correct = {trash_correct:.3f}   "
          f"mean(trash) | v2 wrong = {trash_wrong:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("noise_tsv")
    ap.add_argument("curation_tsv")
    ap.add_argument("--split", default="benchmark_new/split.tsv")
    ap.add_argument("--half", choices=("dev", "test", "both"), default="dev")
    ap.add_argument("--csv", nargs="+", default=[
        "benchmark_new/A3.csv", "benchmark_new/A5.csv", "benchmark_new/ES.csv",
        "benchmark_new/IR.csv", "benchmark_new/NN.csv",
    ])
    ap.add_argument("--link", type=int, default=50000)
    args = ap.parse_args()

    rows = build_table(args.noise_tsv, args.curation_tsv, args.split, args.half)

    report_auc_table(rows, f"pooled AUC ({args.half})")
    by_truth = defaultdict(list)
    for r in rows:
        by_truth[r["truth"]].append(r)
    for cls in sorted(by_truth):
        report_auc_table(by_truth[cls], f"AUC within truth={cls}")

    cluster_grouped_logreg_auc(rows, args.csv, link=args.link)
    report_vs_v2_correctness(rows)


if __name__ == "__main__":
    main()
