"""Stage 18.1 / 18.2 (plans/signal-native-v2-stage18-domain-probe.md): what
does a real-BAM record look like to the model's own features, and can those
features express curated NN at all?

Stages 14-17 established *that* the Transformer can tell a real record from
a simulated one (474 real rows were enough for a real-vs-sim shortcut to
swamp the NN signal) but never *which features* carry the tell, whether the
window convention alone is one, or how much of the sim-real gap is a
per-cohort signature versus a simulation artefact. This module measures all
of that with LightGBM on the shared ``dsignal.features`` row -- the same
information the Transformer sees, aggregated the way ``baseline_gbm`` does --
so every answer is about the representation, not about a model.

Three subcommands, all on one cached table:

``build``
    One row per event over the real pool (every ``real_*`` npz dir) plus a
    seeded per-class subsample of one simulation matched to the real pool's
    label counts. ``FeatureConfig`` and ``feature_subset`` come from a
    training checkpoint, never from CLI defaults, so the probe sees exactly
    the deployed representation. Rows are 50 kb single-linkage clustered
    over their npz coordinates (``dsignal.benchmark_split.cluster``) -- gene
    ids are not consistent across cohorts, and ``ptbp1_arena`` shares loci
    with ``benchmark_new`` dev, which clustering handles and gene ids would
    not.

``real-vs-sim``
    Cluster-grouped 5-fold OOF real-vs-sim discriminator: pooled AUC,
    per-cohort AUC, class-conditional AUC, leave-one-cohort-out AUC (is the
    tell a sim artefact every cohort shares, or a cohort signature?),
    univariate Mann-Whitney AUC and gain importance per feature, and a
    greedy removal curve that re-ranks after each removal -- how many
    feature *families* (an event feature, or a token feature with its three
    aggregates) must go before the discriminator drops to chance.

``real-nn``
    Real rows only. Within the arena cohorts (``ptbp1`` + ``aqr``, caller
    labels, ~67% pure): cluster-grouped 5-fold 5-class and binary
    NN-vs-positive. Transfer to the curated ``bnew_dev`` rows from a model
    trained on the arena cohorts *minus every cluster that contains a
    ``bnew_dev`` row*; leave-one-cohort-out between the two arenas; and a
    paired McNemar against the Transformer's real-supervision ceiling on the
    same 242 dev events.

Decision rules are pre-registered in the plan doc and are not repeated
here so they cannot drift.

Usage::

    python -m dsignal.domain_probe build --ckpt runs/scale_union2_n12480/fold0.pt \\
        --real sim/experiment/real_pos_ptbp1 sim/experiment/real_nn_ptbp1 \\
               sim/experiment/real_pos_aqr sim/experiment/real_nn_aqr \\
               runs/real_dev_npz runs/real_tra2_npz \\
        --sim-npz sim/experiment/npz_union2 --sim-labels sim/experiment/labels_union2.n12480.tsv \\
        --out runs/domain_probe/table.npz
    python -m dsignal.domain_probe real-vs-sim --table runs/domain_probe/table.npz \\
        --out-dir runs/domain_probe/rvs
    python -m dsignal.domain_probe real-nn --table runs/domain_probe/table.npz \\
        --out-dir runs/domain_probe/realnn --ceiling-oof runs/real_supervision_ceiling/oof.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from os.path import join as pjoin

import numpy as np

from dsignal.baseline_gbm import (
    apply_feature_subset,
    build_dataset,
    grouped_folds,
    report,
    report_collapsed,
)
from dsignal.benchmark_split import cluster
from dsignal.dataset import CLASSES
from dsignal.features import FeatureConfig, gbm_feature_names
from dsignal.prior_sweep import mcnemar
from dsignal.score_benchmark import collapse

DEFAULT_LINK = 50000


def cohort_of(kind: str) -> str:
    """``real_pos_ptbp1``/``real_nn_ptbp1`` -> ``ptbp1``; ``real_bnew_dev`` ->
    ``bnew_dev``; ``real_tra2`` -> ``tra2``; ``real_dev`` (the historical
    ceiling build) -> ``bnew_dev``; anything not ``real_*`` -> ``sim``."""
    if not kind.startswith("real_"):
        return "sim"
    rest = kind[len("real_"):]
    for prefix in ("pos_", "nn_"):
        if rest.startswith(prefix):
            return rest[len(prefix):]
    return "bnew_dev" if rest == "dev" else rest


def _gbm_params(seed: int, n_estimators: int):
    return dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _recall_at_fpr(y, s, fpr: float) -> float:
    """Recall of the positive class (y==1) at the score threshold that lets
    ``fpr`` of the negatives through."""
    y, s = np.asarray(y), np.asarray(s)
    neg = s[y == 0]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr))
    return float((s[y == 1] > thr).mean())


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _labels_in(npz_dir: str) -> str:
    """The labels TSV that belongs to a real npz dir: ``<dir>/../labels.tsv``
    for ``<cohort>/npz`` and ``<cohort>/npz_jitter``, ``<dir>/labels.tsv`` if
    the dir itself holds it."""
    for cand in (pjoin(npz_dir, "labels.tsv"), pjoin(os.path.dirname(npz_dir.rstrip("/")), "labels.tsv")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"no labels.tsv for {npz_dir}")


def cmd_build(args) -> None:
    import torch

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = FeatureConfig(**ck["feature_config"])
    subset = ck["feature_subset"]
    if args.depth_norm is not None:
        cfg.depth_norm = args.depth_norm
    if args.event_groups is not None:
        cfg = FeatureConfig(**{**cfg.__dict__, "event_groups": tuple(args.event_groups)})
    print(f"feature config from {args.ckpt}: {cfg}, subset {subset!r}", file=sys.stderr)

    X, y, ids, kinds, meta = [], [], [], [], []
    for real_dir in args.real:
        npz_dir = pjoin(real_dir, args.npz_subdir) if os.path.isdir(pjoin(real_dir, args.npz_subdir)) else real_dir
        labels = _labels_in(npz_dir)
        Xr, yr, idr, _g, mr = build_dataset(npz_dir, labels, cfg, return_meta=True)
        if not all(m["kind"].startswith("real_") for m in mr):
            raise ValueError(f"{labels}: every kind must start with 'real_'")
        print(f"  {npz_dir}: {len(yr)} rows, kinds {sorted(set(m['kind'] for m in mr))}", file=sys.stderr)
        X.append(Xr); y.extend(yr.tolist()); ids.extend(idr); meta.extend(mr)
    real_counts = Counter(y)
    print(f"real pool: {sum(real_counts.values())} rows, {dict(real_counts)}", file=sys.stderr)

    Xs, ys, ids_s, _g, ms = build_dataset(args.sim_npz, args.sim_labels, cfg, return_meta=True)
    rng = np.random.default_rng(args.seed)
    keep = []
    for c in CLASSES:
        idx = np.flatnonzero(ys == c)
        n = min(len(idx), real_counts.get(c, 0)) if not args.all_sim else len(idx)
        keep.extend(rng.choice(idx, size=n, replace=False).tolist())
    keep = sorted(keep)
    print(f"sim {args.sim_npz}: {len(ys)} rows, kept {len(keep)} matched per class "
          f"{dict(Counter(ys[keep].tolist()))}", file=sys.stderr)
    X.append(Xs[keep]); y.extend(ys[keep].tolist()); ids.extend(ids_s[i] for i in keep)
    meta.extend(ms[i] for i in keep)

    X = np.concatenate(X, axis=0)
    y = np.asarray(y)
    kinds = np.asarray([m["kind"] for m in meta])
    cohorts = np.asarray([cohort_of(k) for k in kinds])
    chrom = np.asarray([str(m["chrom"]) for m in meta])
    start = np.asarray([int(m["start"]) for m in meta])
    end = np.asarray([int(m["end"]) for m in meta])

    events = [(None, None, chrom[i], int(start[i]), int(end[i]), None) for i in range(len(y))]
    clusters = cluster(events, args.link)
    cluster_id = np.zeros(len(y), dtype=np.int64)
    for cid, members in enumerate(clusters):
        for i in members:
            cluster_id[i] = cid
    n_mixed = sum(1 for cl in clusters if len({cohorts[i] for i in cl}) > 1)
    print(f"{len(clusters)} clusters at {args.link} bp over {len(y)} rows; "
          f"{n_mixed} span more than one cohort", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, X=X, y=y, kind=kinds, cohort=cohorts, is_real=(cohorts != "sim").astype(np.int8),
        chrom=chrom, start=start, end=end, event_id=np.asarray(ids), cluster=cluster_id,
        feat_names=np.asarray(gbm_feature_names(cfg)),
        config=json.dumps({"ckpt": args.ckpt, "feature_config": cfg.__dict__, "feature_subset": subset,
                           "sim_npz": args.sim_npz, "sim_labels": args.sim_labels,
                           "real": args.real, "npz_subdir": args.npz_subdir,
                           "link": args.link, "seed": args.seed}),
    )
    print(f"wrote {args.out}: X {X.shape}, cohorts {dict(Counter(cohorts.tolist()))}", file=sys.stderr)


def load_table(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    t = {k: d[k] for k in d.files}
    t["config"] = json.loads(str(t["config"]))
    t["feat_names"] = [str(n) for n in t["feat_names"]]
    return t


# --------------------------------------------------------------------------
# real-vs-sim
# --------------------------------------------------------------------------


def _families(feat_names):
    """feature column -> family name: event features are their own family,
    token aggregates (``max__x``/``mean__x``/``wmean__x``) share ``x``."""
    fam = [n.split("__", 1)[1] if "__" in n else n for n in feat_names]
    members = defaultdict(list)
    for j, f in enumerate(fam):
        members[f].append(j)
    return fam, members


def _oof_binary(X, y, groups, folds, seed, n_estimators):
    import lightgbm as lgb

    splits = grouped_folds(X, y, groups, folds, verbose=False)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in splits:
        clf = lgb.LGBMClassifier(objective="binary", **_gbm_params(seed, n_estimators))
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return oof


def _gain(X, y, seed, n_estimators):
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(objective="binary", importance_type="gain",
                             **_gbm_params(seed, n_estimators))
    clf.fit(X, y)
    return clf.feature_importances_.astype(np.float64)


def cmd_real_vs_sim(args) -> None:
    t = load_table(args.table)
    feat_names = t["feat_names"]
    subset = args.feature_subset or t["config"]["feature_subset"]
    X = apply_feature_subset(t["X"].astype(np.float64), feat_names, subset)
    y = t["is_real"].astype(int)
    cohorts, labels, groups = t["cohort"], t["y"], t["cluster"].tolist()
    if args.cohorts:
        keep = np.isin(cohorts, args.cohorts.split(",") + ["sim"])
        X, y, cohorts, labels = X[keep], y[keep], cohorts[keep], labels[keep]
        groups = [g for g, k in zip(groups, keep) if k]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"real-vs-sim on {len(y)} rows ({int(y.sum())} real / {int((1 - y).sum())} sim), "
          f"subset {subset!r}, table {args.table}", file=sys.stderr)

    # 1. pooled cluster-grouped OOF
    oof = _oof_binary(X, y, groups, args.folds, args.seed, args.n_estimators)
    summary = {"table": args.table, "feature_subset": subset, "n": int(len(y)),
               "n_real": int(y.sum()), "pooled_auc": _auc(y, oof)}
    print(f"\npooled OOF AUC (real vs sim): {summary['pooled_auc']:.3f}")

    sim_mask = cohorts == "sim"
    summary["per_cohort_auc"] = {}
    for c in sorted(set(cohorts.tolist()) - {"sim"}):
        m = (cohorts == c) | sim_mask
        summary["per_cohort_auc"][c] = _auc(y[m], oof[m])
        print(f"  cohort {c:9s} vs sim: AUC {summary['per_cohort_auc'][c]:.3f}  (n_real {int((cohorts == c).sum())})")

    summary["class_conditional_auc"] = {}
    for c in CLASSES:
        m = labels == c
        summary["class_conditional_auc"][c] = _auc(y[m], oof[m])
        print(f"  class {c:4s}: AUC {summary['class_conditional_auc'][c]:.3f}  (n {int(m.sum())})")

    # 2. leave-one-cohort-out: the held-out cohort has never been seen as
    # "real"; its sim counterpart is a cluster-grouped fifth of the sim rows,
    # minus any sim row sharing a cluster with the held-out cohort.
    import lightgbm as lgb

    summary["loco_auc"] = {}
    sim_idx = np.flatnonzero(sim_mask)
    sim_splits = grouped_folds(X[sim_idx], y[sim_idx], [groups[i] for i in sim_idx], args.folds, verbose=False)
    sim_test = set(sim_idx[sim_splits[0][1]].tolist())
    for c in sorted(set(cohorts.tolist()) - {"sim"}):
        held = np.flatnonzero(cohorts == c)
        held_clusters = {groups[i] for i in held}
        test_idx = np.array(sorted(set(held.tolist()) | sim_test))
        train_idx = np.array([i for i in range(len(y)) if i not in set(test_idx.tolist())
                              and groups[i] not in held_clusters])
        clf = lgb.LGBMClassifier(objective="binary", **_gbm_params(args.seed, args.n_estimators))
        clf.fit(X[train_idx], y[train_idx])
        s = clf.predict_proba(X[test_idx])[:, 1]
        summary["loco_auc"][c] = _auc(y[test_idx], s)
        print(f"  leave-{c}-out: AUC {summary['loco_auc'][c]:.3f} "
              f"(train {len(train_idx)} rows from the other cohorts + sim)")

    # 3. per-feature: univariate Mann-Whitney AUC and gain importance
    gain = _gain(X, y, args.seed, args.n_estimators)
    with open(pjoin(args.out_dir, "features.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["feature", "family", "mw_auc", "abs_mw_auc_minus_half", "gain", "real_mean", "sim_mean"])
        fam, _members = _families(feat_names)
        rows = []
        for j, name in enumerate(feat_names):
            col = X[:, j]
            mw = 0.5 if np.all(col == col[0]) else _auc(y, col)
            rows.append((name, fam[j], mw, abs(mw - 0.5), gain[j],
                         float(col[y == 1].mean()), float(col[y == 0].mean())))
        rows.sort(key=lambda r: -r[4])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.1f}", f"{r[5]:.4g}", f"{r[6]:.4g}"])
    print("\ntop 15 by gain:")
    for r in rows[:15]:
        print(f"  {r[0]:45s} gain {r[4]:10.1f}  MW-AUC {r[2]:.3f}")
    summary["top_gain"] = [r[0] for r in rows[:30]]

    # 4. greedy removal curve over families, re-ranked after every removal
    curve = []
    if args.removal_k > 0:
        fam, members = _families(feat_names)
        alive = set(members)
        active = np.ones(X.shape[1], dtype=bool)
        auc_k = summary["pooled_auc"]
        curve.append({"k": 0, "removed": "", "auc": auc_k})
        print(f"\nremoval curve (families; k=0 AUC {auc_k:.3f}):")
        for k in range(1, args.removal_k + 1):
            g = _gain(X[:, active], y, args.seed, args.n_estimators)
            cols = np.flatnonzero(active)
            fam_gain = Counter()
            for gj, j in zip(g, cols):
                fam_gain[fam[j]] += gj
            if not fam_gain or max(fam_gain.values()) <= 0:
                break
            top = max(alive, key=lambda f: fam_gain.get(f, 0.0))
            for j in members[top]:
                active[j] = False
            alive.discard(top)
            oof_k = _oof_binary(X[:, active], y, groups, args.folds, args.seed, args.n_estimators)
            auc_k = _auc(y, oof_k)
            curve.append({"k": k, "removed": top, "auc": auc_k})
            print(f"  k={k:2d} removed {top:40s} -> AUC {auc_k:.3f}")
        with open(pjoin(args.out_dir, "removal_curve.tsv"), "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["k", "removed_family", "auc_after"])
            for r in curve:
                w.writerow([r["k"], r["removed"], f"{r['auc']:.4f}"])
    summary["removal_curve"] = curve

    with open(pjoin(args.out_dir, "oof.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cohort", "label", "is_real", "p_real"])
        for c, l, r, p in zip(cohorts, labels, y, oof):
            w.writerow([c, l, int(r), f"{p:.5f}"])
    with open(pjoin(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out_dir}/{{oof,features,removal_curve}}.tsv and summary.json", file=sys.stderr)


# --------------------------------------------------------------------------
# real-nn
# --------------------------------------------------------------------------


def _fit_multiclass(X, y_int, n_classes, seed, n_estimators):
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes,
                             **_gbm_params(seed, n_estimators))
    clf.fit(X, y_int)
    return clf


def _binary_block(name: str, y_is_nn, p_nn, summary: dict) -> None:
    auc = _auc(y_is_nn, p_nn)
    r05, r10 = _recall_at_fpr(y_is_nn, p_nn, 0.05), _recall_at_fpr(y_is_nn, p_nn, 0.10)
    summary[name] = {"nn_vs_pos_auc": auc, "nn_recall_at_5pct_fpr": r05, "nn_recall_at_10pct_fpr": r10,
                     "n_nn": int(np.sum(y_is_nn)), "n_pos": int(np.sum(1 - np.asarray(y_is_nn)))}
    print(f"  NN-vs-positive AUC {auc:.3f}; NN recall at 5% FPR {r05:.3f}, at 10% FPR {r10:.3f} "
          f"(n_NN {summary[name]['n_nn']}, n_pos {summary[name]['n_pos']})")


def cmd_real_nn(args) -> None:
    t = load_table(args.table)
    feat_names = t["feat_names"]
    subset = args.feature_subset or t["config"]["feature_subset"]
    real = t["is_real"].astype(bool)
    X = apply_feature_subset(t["X"].astype(np.float64), feat_names, subset)[real]
    y, cohorts = t["y"][real], t["cohort"][real]
    groups = t["cluster"][real].tolist()
    event_ids = t["event_id"][real]
    y_int = np.array([CLASSES.index(l) for l in y])
    nn_i = CLASSES.index("NN")
    os.makedirs(args.out_dir, exist_ok=True)
    train_cohorts = args.train_cohorts.split(",")
    summary = {"table": args.table, "feature_subset": subset, "train_cohorts": train_cohorts,
               "dev_cohort": args.dev_cohort}
    print(f"real-only: {len(y)} rows {dict(Counter(cohorts.tolist()))}, subset {subset!r}", file=sys.stderr)

    # (a) within the arena cohorts, cluster-grouped OOF
    tr = np.isin(cohorts, train_cohorts)
    Xt, yt, gt = X[tr], y_int[tr], [g for g, k in zip(groups, tr) if k]
    print(f"\n=== (a) within {train_cohorts}: {len(yt)} rows, {args.folds}-fold cluster-grouped ===")
    oof_p = np.zeros((len(yt), len(CLASSES)))
    for train_idx, test_idx in grouped_folds(Xt, yt, gt, args.folds, verbose=False):
        clf = _fit_multiclass(Xt[train_idx], yt[train_idx], len(CLASSES), args.seed, args.n_estimators)
        oof_p[test_idx] = clf.predict_proba(Xt[test_idx])
    pred = [CLASSES[i] for i in oof_p.argmax(1)]
    truth = [CLASSES[i] for i in yt]
    summary["within_arena_5class"] = report(truth, pred, list(CLASSES), "real-only GBM, within arena cohorts")
    summary["within_arena_collapsed"] = report_collapsed(truth, pred, "real-only GBM, within arena, A3+A5->A")
    _binary_block("within_arena_binary", (yt == nn_i).astype(int), oof_p[:, nn_i], summary)

    # (b) transfer to the curated dev cohort, cluster-excluded
    dev = cohorts == args.dev_cohort
    dev_clusters = {g for g, d in zip(groups, dev) if d}
    tr_x = tr & ~np.isin(np.array(groups), list(dev_clusters))
    n_dropped = int(tr.sum() - tr_x.sum())
    print(f"\n=== (b) transfer {train_cohorts} -> {args.dev_cohort}: {int(tr_x.sum())} train rows "
          f"({n_dropped} dropped for sharing a cluster with a {args.dev_cohort} row), {int(dev.sum())} dev rows ===")
    clf = _fit_multiclass(X[tr_x], y_int[tr_x], len(CLASSES), args.seed, args.n_estimators)
    p_dev = clf.predict_proba(X[dev])
    pred_dev = [CLASSES[i] for i in p_dev.argmax(1)]
    truth_dev = y[dev].tolist()
    summary["transfer_5class"] = report(truth_dev, pred_dev, list(CLASSES), f"real-only GBM -> {args.dev_cohort}")
    summary["transfer_collapsed"] = report_collapsed(truth_dev, pred_dev, f"-> {args.dev_cohort}, A3+A5->A")
    _binary_block("transfer_binary", (y_int[dev] == nn_i).astype(int), p_dev[:, nn_i], summary)
    nn_hit = sum(1 for tr_, pr in zip(truth_dev, pred_dev) if tr_ == "NN" and pr == "NN")
    nn_n = sum(1 for tr_ in truth_dev if tr_ == "NN")
    summary["transfer_nn_recall_argmax"] = nn_hit / nn_n if nn_n else float("nan")
    print(f"  NN recall at argmax: {nn_hit}/{nn_n}")
    summary["transfer_n_train_dropped_for_cluster_overlap"] = n_dropped

    # every other non-training cohort, scored with the same model (tra2 has
    # no NN, so only its 5-class/collapsed accuracy is meaningful)
    summary["transfer_other_cohorts"] = {}
    for c in sorted(set(cohorts.tolist()) - set(train_cohorts) - {args.dev_cohort}):
        m = cohorts == c
        p = clf.predict_proba(X[m])
        pr = [CLASSES[i] for i in p.argmax(1)]
        acc = float(np.mean([a == b for a, b in zip(pr, y[m])]))
        acc_c = float(np.mean([collapse(a) == collapse(b) for a, b in zip(pr, y[m])]))
        summary["transfer_other_cohorts"][c] = {"n": int(m.sum()), "accuracy": acc, "accuracy_collapsed": acc_c}
        print(f"  also -> {c}: n {int(m.sum())}, 5-class {acc:.3f}, collapsed {acc_c:.3f}")

    with open(pjoin(args.out_dir, "dev_pred.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["event_id", "region", "truth", "pred"] + [f"p_{c}" for c in CLASSES])
        for eid, tr_, pr, p in zip(event_ids[dev], truth_dev, pred_dev, p_dev):
            w.writerow([eid, eid.split(";", 1)[1], tr_, pr] + [f"{v:.5f}" for v in p])

    # (c) leave-one-cohort-out between the arenas
    summary["loco"] = {}
    if len(train_cohorts) >= 2:
        print("\n=== (c) leave-one-arena-out ===")
        for held in train_cohorts:
            src = [c for c in train_cohorts if c != held]
            held_clusters = {g for g, c in zip(groups, cohorts) if c == held}
            m_tr = np.isin(cohorts, src) & ~np.isin(np.array(groups), list(held_clusters))
            m_te = cohorts == held
            clf_l = _fit_multiclass(X[m_tr], y_int[m_tr], len(CLASSES), args.seed, args.n_estimators)
            p = clf_l.predict_proba(X[m_te])
            pr = [CLASSES[i] for i in p.argmax(1)]
            acc = float(np.mean([a == b for a, b in zip(pr, y[m_te])]))
            auc = _auc((y_int[m_te] == nn_i).astype(int), p[:, nn_i])
            summary["loco"][f"{'+'.join(src)}->{held}"] = {"n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
                                                            "accuracy": acc, "nn_vs_pos_auc": auc}
            print(f"  {'+'.join(src)} -> {held}: 5-class acc {acc:.3f}, NN-vs-pos AUC {auc:.3f} "
                  f"(train {int(m_tr.sum())}, test {int(m_te.sum())})")

    # (d) paired McNemar vs the Transformer real-supervision ceiling
    if args.ceiling_oof:
        ceil = {}
        with open(args.ceiling_oof, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                ceil[r["event_id"].split(";", 1)[1]] = (r["truth"], r["pred"])
        regions = [eid.split(";", 1)[1] for eid in event_ids[dev]]
        shared = [(i, regions[i]) for i in range(len(regions)) if regions[i] in ceil]
        base_pred = [ceil[r][1] for _, r in shared]
        new_pred = [pred_dev[i] for i, _ in shared]
        tru = [truth_dev[i] for i, _ in shared]
        for i, r in shared:
            assert ceil[r][0] == truth_dev[i], f"truth mismatch at {r}"
        b, c, p = mcnemar(base_pred, new_pred, tru)
        bc, cc, pc = mcnemar([collapse(x) for x in base_pred], [collapse(x) for x in new_pred],
                             [collapse(x) for x in tru])
        n_base = sum(1 for a, t_ in zip(base_pred, tru) if a == t_)
        n_new = sum(1 for a, t_ in zip(new_pred, tru) if a == t_)
        summary["vs_ceiling"] = {"n_shared": len(shared), "ceiling_correct": n_base, "gbm_correct": n_new,
                                 "mcnemar_5class": {"ceiling_only": b, "gbm_only": c, "p": p},
                                 "mcnemar_collapsed": {"ceiling_only": bc, "gbm_only": cc, "p": pc}}
        print(f"\n=== (d) vs Transformer ceiling ({args.ceiling_oof}), {len(shared)} shared events ===")
        print(f"  5-class correct: ceiling {n_base}, real-only GBM {n_new}; "
              f"McNemar ceiling-only {b} / gbm-only {c}, p={p:.3f}")
        print(f"  collapsed: McNemar ceiling-only {bc} / gbm-only {cc}, p={pc:.3f}")
        print("  NOTE: the ceiling was trained *within* bnew_dev (5-fold); this GBM never saw a "
              "bnew_dev row -- not the same protocol, so this is a bound comparison, not an A/B.")

    with open(pjoin(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out_dir}/dev_pred.tsv and summary.json", file=sys.stderr)


# --------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="cache the feature table (real pool + class-matched sim subsample)")
    b.add_argument("--ckpt", required=True, help="checkpoint whose feature_config/feature_subset define the representation")
    b.add_argument("--real", nargs="+", required=True,
                   help="real cohort dirs (each holding labels.tsv and npz/ -- or an npz dir itself)")
    b.add_argument("--npz-subdir", default="npz", help="'npz' (default) or 'npz_jitter' for the jittered-pad pool")
    b.add_argument("--sim-npz", required=True)
    b.add_argument("--sim-labels", required=True)
    b.add_argument("--all-sim", action="store_true", help="keep every sim row instead of class-matching")
    b.add_argument("--depth-norm", choices=("none", "scale", "drop"), default=None,
                   help="override the checkpoint's depth_norm (the 'scale' variant of 18.1)")
    b.add_argument("--event-groups", nargs="*", default=None,
                   help="Stage 18.5: add opt-in event groups (covshape delcov) on top of the checkpoint's config")
    b.add_argument("--link", type=int, default=DEFAULT_LINK)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("real-vs-sim", help="cluster-grouped real-vs-sim discriminator + removal curve")
    r.add_argument("--table", required=True)
    r.add_argument("--feature-subset", default=None,
                   help="default: the checkpoint's (stored in the table); 'no-geometry,no-covshape' etc. also accepted")
    r.add_argument("--cohorts", default=None, help="comma-separated real cohorts to keep (default all)")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-estimators", type=int, default=400)
    r.add_argument("--removal-k", type=int, default=30, help="families to remove greedily (0 = skip)")
    r.add_argument("--out-dir", required=True)
    r.set_defaults(func=cmd_real_vs_sim)

    n = sub.add_parser("real-nn", help="real-only NN expressibility (18.2)")
    n.add_argument("--table", required=True)
    n.add_argument("--feature-subset", default=None)
    n.add_argument("--train-cohorts", default="ptbp1,aqr")
    n.add_argument("--dev-cohort", default="bnew_dev")
    n.add_argument("--ceiling-oof", default=None, help="runs/real_supervision_ceiling/oof.tsv for the paired McNemar")
    n.add_argument("--folds", type=int, default=5)
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--n-estimators", type=int, default=400)
    n.add_argument("--out-dir", required=True)
    n.set_defaults(func=cmd_real_nn)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
