"""Training CLI for the junction-set Transformer (Phase 3 step 3).

Same evaluation protocol as ``sim/baseline_model.py`` and
``dsignal/baseline_gbm.py`` -- ``GroupKFold(5)`` grouped by gene with an
``assert_no_gene_leak`` check, same reporting format -- so every number stays
comparable to the whole layer-by-layer history. Held-out-fold predictions are
pooled across folds into one out-of-fold report, exactly as
``cross_val_predict`` does for the two baselines.

**Epoch selection changed on 2026-08-17 and old numbers are not comparable.**
Until then the best epoch (and early stopping) was chosen by held-out-fold
macro-F1 -- the same fold the run then reported. That makes the reported sim-CV
a best-of-N maximum over epochs rather than an estimate of held-out
performance, which matters here because sim-CV is the steering signal for every
experiment on this branch. The default is now ``--select-on inner``: a
gene-disjoint 15% of each *training* fold picks the epoch, and the held-out
fold is scored exactly once with that epoch. Reported sim-CV drops as a result;
that is the point, and ``--select-on held-out`` reproduces any older number.

*Measured (2026-08-17, ``runs/geomwin_innerval.*`` vs ``runs/geomwin_allaug.*``,
same data and hyperparameters):* sim-CV macro-F1 0.867 -> **0.854**, per-fold
0.843/0.884/0.873/0.872/0.860 -> 0.847/0.863/0.856/0.849/0.855 -- the old
protocol was inflating by ~0.013 and, tellingly, was also *noisier* across
folds. Benchmark transfer fell too: dev half 55/70 -> 48/70 collapsed (78.6% ->
68.6%, McNemar p = 0.065; 5-class p = 0.022). That drop is **not** an accounting
artifact -- it is a real change in the model, and it confounds two things this
switch does at once: it makes selection honest *and* it trains each fold on 15%
less data. ``--refit`` separates them, and the answer is **the data is not it**:
refitting on the whole fold gives back 49/70 vs 48/70 (p = 1.000) and sim-CV
0.853 vs 0.854. The transfer cost belongs to the epoch choice itself.

The uncomfortable reading, recorded rather than buried: the branch's headline
78.6% came from checkpoints whose epoch was picked by peeking at a held-out
fold, and picking on ~960 gene-disjoint events tracked benchmark transfer
better than picking on ~576 does. Part of that 10-point gap is a genuinely
better selection signal and part is the optimism of a maximum -- this
experiment cannot split them. What is certain is that 78.6% was selected
optimistically on both axes, and that the honest re-baseline is ~70% on dev.

Usage:
    python -m dsignal.train --npz-dir sim/experiment/npz_scale1260 \
        --labels sim/experiment/labels_scale1260.tsv --out-dir runs/control
    python -m dsignal.train ... --augment flip condition_swap
    python -m dsignal.train ... --augment all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsignal.baseline_gbm import grouped_folds, print_vs_reference, report, report_collapsed
from dsignal.dataset import (
    CLASSES,
    TYPE_CLASSES,
    AugmentConfig,
    JunctionSetDataset,
    collate,
    read_labels,
)
from dsignal.features import EXTRA_EVENT_GROUPS, FeatureConfig, event_feature_names
from dsignal.model import JunctionSetTransformer, ModelConfig, batch_inputs
from sim.gene_split import assert_no_gene_leak, split_genes


def class_weights(labels, classes, device, boost: dict | None = None):
    """Class-balanced cross-entropy weights from the *training* fold only.

    ``boost`` (``{class: factor}``) multiplies specific classes' weights
    *after* balancing -- the cheap way to shift a decision boundary (e.g.
    CSE vs A3/A5, plans/signal-native-v2-ensemble.md Stage 2's ``ens_cse``
    member) without touching the feature/model plumbing.
    """
    counts = np.array([max(sum(1 for l in labels if l == c), 1) for c in classes], dtype=np.float64)
    w = counts.sum() / (len(classes) * counts)
    if boost:
        for c, factor in boost.items():
            w[classes.index(c)] *= factor
    return torch.tensor(w, dtype=torch.float32, device=device)


def aux_weights(ds, device):
    """``(type_weights [T], pos_weight [])`` for the two-head loss, counted on
    the training fold only -- the same rule ``class_weights`` follows.

    The type head is balanced over the four structural types; the differential
    head gets ``n_negative / n_positive`` so the 25% NN design fraction does not
    tilt it. Both are counted from the dataset's own entries, so a fold with an
    unusual mix is weighted for what it actually contains.
    """
    from dsignal.dataset import aux_targets

    targets = [aux_targets(e[1], e[2]) for e in ds.entries]
    counts = np.array([max(sum(1 for t, _ in targets if t == i), 1)
                       for i in range(len(TYPE_CLASSES))], dtype=np.float64)
    w = counts.sum() / (len(TYPE_CLASSES) * counts)
    n_pos = max(sum(1 for _, d in targets if d > 0.5), 1)
    n_neg = max(len(targets) - n_pos, 1)
    return (torch.tensor(w, dtype=torch.float32, device=device),
            torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device))


def two_head_loss(model, batch, device, type_weights, pos_weight, diff_lambda: float):
    """Type cross-entropy (masked) + differential BCE.

    The type term is averaged over the events that *have* a structural type --
    every positive plus the ``nn_hard`` negatives, which are catalogued events
    held at dPSI = 0. ``nn_plain``/``nn_novel`` carry no event and are excluded
    by the -1 sentinel rather than being folded into a fifth type, which is the
    whole reason the head is 4-way and not 5-way.
    """
    type_logits, diff_logit = model.forward_parts(**batch_inputs(batch, device))
    type_target = batch["type_label"].to(device)
    is_diff = batch["is_diff"].to(device)

    has_type = type_target >= 0
    if has_type.any():
        loss_type = torch.nn.functional.cross_entropy(
            type_logits[has_type], type_target[has_type], weight=type_weights
        )
    else:  # a batch of nothing but plain/novel negatives is possible, if rare
        loss_type = type_logits.sum() * 0.0
    loss_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        diff_logit, is_diff, pos_weight=pos_weight
    )
    return loss_type + diff_lambda * loss_diff


def domain_loss(model, batch, device, pos_weight=None, grl_lambda: float = 1.0):
    """Stage 17: BCE on `model.domain_logits` against `batch["is_real"]`.

    Unweighted by class-imbalance-of-domain by default (`pos_weight=None`):
    the real fraction is small (~7% in Stage 16's mix) but that imbalance is
    exactly what the domain head is *supposed* to have a hard time with --
    down-weighting it away would blunt the adversarial signal the encoder
    is meant to resist. Kept as a parameter, not hardcoded, in case a run
    needs it."""
    domain_logit = model.domain_logits(**batch_inputs(batch, device), grl_lambda=grl_lambda)
    is_real = batch["is_real"].to(device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        domain_logit, is_real, pos_weight=pos_weight
    )


def evaluate(model, loader, device):
    """(predictions [n], labels [n], event_ids [n]) over a loader."""
    model.eval()
    preds, golds, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(**batch_inputs(batch, device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            golds.extend(batch["label"].tolist())
            ids.extend(batch["event_id"])
    return np.array(preds), np.array(golds), ids


def macro_f1(pred, gold, n_classes) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(gold, pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def train_fold(args, entries, train_idx, test_idx, fold: int, device):
    cfg = FeatureConfig(flank=args.flank, max_junctions=args.max_junctions,
                        depth_norm=args.depth_norm, event_groups=tuple(args.event_groups or ()))
    aug = AugmentConfig.from_names(args.augment, prob=args.augment_prob,
                                   subsample_low=args.subsample_low)

    fold_entries = [entries[i] for i in train_idx]
    test_entries = [entries[i] for i in test_idx]

    # Model selection happens on an inner split carved out of the *training*
    # fold, never on the held-out fold being reported. Selecting the best epoch
    # on the reported fold -- which is what this did until 2026-08-17 -- makes
    # the fold's own noise part of the choice, so the reported sim-CV is a
    # best-of-N maximum rather than an estimate, and it is being used as the
    # steering signal for every experiment. The split is by gene, like the
    # outer one, and seeded per fold so the folds do not share an inner
    # partition. Cost: the model trains on ~15% less data per fold.
    if args.select_on == "inner":
        assign = split_genes([e[3] for e in fold_entries],
                             val_frac=args.inner_val_frac, test_frac=0.0,
                             seed=args.seed + 1000 + fold)
        train_entries = [e for e in fold_entries if assign[e[3]] == "train"]
        val_entries = [e for e in fold_entries if assign[e[3]] == "val"]
        assert_no_gene_leak(
            [dict(gene_id=e[3], split=assign[e[3]]) for e in fold_entries],
            gene_key="gene_id", split_key="split",
        )
        if not val_entries:
            raise ValueError(f"--inner-val-frac {args.inner_val_frac} left fold {fold} no inner val set")
    else:  # "held-out": the old, optimistic protocol, kept to reproduce old runs
        train_entries, val_entries = fold_entries, test_entries

    # scaler is fit on unaugmented training-fold records only, then reused for
    # the held-out fold -- fitting it on everything would leak fold statistics.
    # It is fit on the whole training fold, inner val included: that is still
    # training-fold data, so nothing about the reported fold enters it.
    fit_ds = JunctionSetDataset(
        fold_entries, cfg=cfg, feature_subset=args.feature_subset, cache=not args.no_cache, coverage_bins=args.coverage_bins
    )
    scaler = fit_ds.fit_scaler()

    train_ds = JunctionSetDataset(
        train_entries, cfg=cfg, augment=aug, scaler=scaler,
        feature_subset=args.feature_subset, cache=not args.no_cache, coverage_bins=args.coverage_bins, seed=args.seed,
    )
    test_ds = JunctionSetDataset(
        test_entries, cfg=cfg, scaler=scaler,
        feature_subset=args.feature_subset, cache=not args.no_cache, coverage_bins=args.coverage_bins, seed=args.seed,
    )
    # must come after test_ds: under "held-out" the val set *is* the test set
    val_ds = JunctionSetDataset(
        val_entries, cfg=cfg, scaler=scaler,
        feature_subset=args.feature_subset, cache=not args.no_cache, coverage_bins=args.coverage_bins, seed=args.seed,
    ) if args.select_on == "inner" else test_ds

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers,
    ) if args.select_on == "inner" else test_loader

    mcfg = ModelConfig(
        n_classes=len(CLASSES), d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
        two_head=args.two_head, n_types=len(TYPE_CLASSES),
        use_distance_bias=not args.no_distance_bias, use_relation_bias=not args.no_relation_bias,
        coverage_bins=args.coverage_bins, coverage_channels=args.coverage_channels,
        domain_adversarial=args.domain_lambda > 0,
        n_event_features=len(event_feature_names(cfg)),
    )
    model = JunctionSetTransformer(mcfg).to(device)
    if fold == 0:
        print(f"model: {model.n_parameters():,} trainable parameters", file=sys.stderr)

    weights = class_weights(train_ds.labels, CLASSES, device, boost=args.class_weight_boost)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # two-head: class-balance the type head over the four types, and balance the
    # differential head by the positive:negative ratio of the training fold
    type_weights = pos_weight = None
    if args.two_head:
        type_weights, pos_weight = aux_weights(train_ds, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"val_macro_f1": -1.0, "epoch": -1, "state": None}
    since_best = 0

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss, n_seen = 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            if args.two_head:
                loss = two_head_loss(model, batch, device, type_weights, pos_weight,
                                     args.diff_lambda)
            else:
                logits, _ = model(**batch_inputs(batch, device))
                loss = criterion(logits, batch["label"].to(device))
            if args.domain_lambda > 0:
                loss = loss + args.domain_lambda * domain_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total_loss += float(loss) * len(batch["label"])
            n_seen += len(batch["label"])
        sched.step()

        val_pred, val_gold, _ = evaluate(model, val_loader, device)
        f1 = macro_f1(val_pred, val_gold, len(CLASSES))
        if f1 > best["val_macro_f1"]:
            best = {
                "val_macro_f1": f1, "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            since_best = 0
        else:
            since_best += 1

        if args.verbose or epoch % 10 == 0:
            which = "inner-val" if args.select_on == "inner" else "held-out"
            print(
                f"  fold {fold} epoch {epoch:3d}  loss {total_loss / max(n_seen, 1):.4f}  "
                f"{which} macro-F1 {f1:.3f}  (best {best['val_macro_f1']:.3f} @ {best['epoch']})",
                file=sys.stderr,
            )
        if since_best >= args.patience:
            print(f"  fold {fold}: early stop at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)", file=sys.stderr)
            break

    # --refit: keep the honest epoch choice but give the final model the whole
    # training fold back, by retraining from scratch for exactly that many
    # epochs. Selection still never sees the reported fold; what changes is only
    # how much data the shipped weights saw. Without this, --select-on inner
    # confounds "honest selection" with "15% less training data".
    if args.select_on == "inner" and args.refit:
        n_epochs = best["epoch"] + 1
        print(f"  fold {fold}: refitting on the full training fold for {n_epochs} epochs",
              file=sys.stderr)
        full_ds = JunctionSetDataset(
            fold_entries, cfg=cfg, augment=aug, scaler=scaler,
            feature_subset=args.feature_subset, cache=not args.no_cache, coverage_bins=args.coverage_bins, seed=args.seed,
        )
        full_loader = DataLoader(
            full_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
            num_workers=args.num_workers, drop_last=False,
        )
        model = JunctionSetTransformer(mcfg).to(device)
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights(full_ds.labels, CLASSES, device, boost=args.class_weight_boost)
        )
        refit_type_weights, refit_pos_weight = (
            aux_weights(full_ds, device) if args.two_head else (None, None))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # same schedule shape as the selection run, so epoch n means the same thing
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        tail = []  # weights of the last --avg-last-k epochs, for tail averaging
        for epoch in range(n_epochs):
            full_ds.set_epoch(epoch)
            model.train()
            for batch in full_loader:
                opt.zero_grad(set_to_none=True)
                if args.two_head:
                    loss = two_head_loss(model, batch, device, refit_type_weights,
                                         refit_pos_weight, args.diff_lambda)
                else:
                    logits, _ = model(**batch_inputs(batch, device))
                    loss = criterion(logits, batch["label"].to(device))
                if args.domain_lambda > 0:
                    loss = loss + args.domain_lambda * domain_loss(model, batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
            sched.step()
            if args.avg_last_k > 1 and epoch >= n_epochs - args.avg_last_k:
                tail.append({k: v.detach().cpu().clone().float()
                             for k, v in model.state_dict().items()})
        if len(tail) > 1:
            # Tail averaging (SWA without the extra schedule): the refit run
            # stops at one arbitrary epoch, and 1.2 showed the transfer cost of
            # this protocol lives in exactly that arbitrariness. Averaging the
            # final weights over the last k epochs spends nothing extra and
            # removes the dependence on the last epoch's noise.
            avg = {k: sum(t[k] for t in tail) / len(tail) for k in tail[0]}
            best["state"] = {k: v.to(dtype=tail[0][k].dtype) for k, v in avg.items()}
            print(f"  fold {fold}: averaged the last {len(tail)} epochs' weights", file=sys.stderr)
        else:
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_entries = fold_entries  # the prior stored below is what this model saw

    # the held-out fold is scored exactly once, with the epoch the inner split
    # chose -- so this number is an estimate, not a maximum over epochs
    model.load_state_dict(best["state"])
    pred, gold, ids = evaluate(model, test_loader, device)
    best.update(macro_f1=macro_f1(pred, gold, len(CLASSES)), pred=pred, gold=gold, ids=ids)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(
            {
                "model_state": best["state"],
                "model_config": mcfg.to_dict(),
                "scaler": scaler.state_dict(),
                "classes": list(CLASSES),
                "feature_config": {"flank": cfg.flank, "max_junctions": cfg.max_junctions,
                                   "canonicalize": cfg.canonicalize,
                                   "depth_norm": cfg.depth_norm,
                                   "depth_target": cfg.depth_target,
                                   "event_groups": list(cfg.event_groups)},
                "feature_subset": args.feature_subset,
                "fold": fold,
                "held_out_macro_f1": best["macro_f1"],
                # which number chose the epoch -- under select_on="inner" the
                # two are independent, under "held-out" they are the same set
                "inner_val_macro_f1": best["val_macro_f1"],
                "select_on": args.select_on,
                "best_epoch": best["epoch"],
                # training class prior, so inference-time prior correction
                # (dsignal.predict.PriorCorrection) does not need the labels
                # TSV. Checkpoints written before this field exist -- callers
                # must tolerate its absence. Under --select-on inner this counts
                # the inner-train subset, i.e. exactly what the model saw.
                "train_class_counts": {c: int(sum(1 for e in train_entries if e[2] == c)) for c in CLASSES},
            },
            pjoin(args.out_dir, f"fold{fold}.pt"),
        )

    return best


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="per-fold checkpoints go here")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)

    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no-distance-bias", action="store_true",
                    help="ablation (plan Stage 8, B3): disable the signed log-distance attention bias")
    ap.add_argument("--coverage-bins", type=int, default=0,
                    help="Stage 12 / Track D: bins in the coverage branch (0 = branch off, "
                         "the pre-Stage-12 behaviour)")
    ap.add_argument("--coverage-channels", type=int, default=64,
                    help="channels in the coverage encoder's conv stack")
    ap.add_argument("--no-relation-bias", action="store_true",
                    help="ablation (plan Stage 8, B3): disable the pairwise-relation attention bias")

    ap.add_argument("--flank", type=int, default=50)
    ap.add_argument("--max-junctions", type=int, default=256)
    ap.add_argument("--event-groups", nargs="*", default=None, choices=list(EXTRA_EVENT_GROUPS),
                    help="Stage 18.5: opt-in event-feature groups appended to the CLS vector "
                    "(covshape: lag-50 coverage autocorrelation; delcov: deletion-coverage "
                    "summaries). Stored in the checkpoint's feature_config; off by default so "
                    "the vector stays the historical 32 columns")
    ap.add_argument(
        "--feature-subset", default="no-geometry",
        help="'all', 'geometry' or 'no-geometry', optionally followed by ',no-covshape'/"
        "',no-delcov' to zero an opted-in event group (Stage 18.5 ablation). "
        "See dsignal/baseline_gbm.py for the full note. Window geometry is "
        "excluded by default -- originally because the v1 window sampler made it "
        "a pure simulation artifact. Under window sampler v3 the coupling is "
        "partly real (geometry alone recovers 13/17 curated IR events on the "
        "benchmark). 'all' has now been scored under v3 and is a null with the "
        "wrong sign: sim-CV 0.867 -> 0.873 but benchmark dev 55/70 -> 52/70 "
        "collapsed (McNemar p = 0.549), the loss concentrated in RI (6/8 -> "
        "3/8). The default stays. 'geometry' is the geometry-only control.",
    )
    ap.add_argument(
        "--augment", nargs="*", default=[],
        help="any of: flip condition_swap subsample junction_dropout jitter, or 'all'. "
        "Default none -- the control run.",
    )
    ap.add_argument("--augment-prob", type=float, default=0.5)
    ap.add_argument(
        "--subsample-low", type=float, default=0.3,
        help="subsample augmentation retains p ~ U(this, 1) of the reads. The "
        "default 0.3 never reaches the depth the curated benchmark actually "
        "sits at -- dsignal/audit_noise.py measures the simulation as 5.2x "
        "deeper, i.e. real data lives near 0.19. Widening it to 0.15 to close "
        "that gap was TRIED AND HURT: 73.6%% benchmark collapsed vs 78.6%% for "
        "the 0.3 control, and sim-CV 0.862 vs 0.867. The default stays at 0.3.",
    )
    ap.add_argument(
        "--depth-norm", choices=("none", "scale", "drop"), default="none",
        help="how to treat absolute sequencing depth (dsignal/features.py: "
        "rescale_depth). 'scale' maps every record onto a reference spliced "
        "depth, making the features invariant to how deeply the locus was "
        "sequenced; 'drop' zeroes the depth-absolute columns instead. "
        "Default 'none' is the historical behaviour. MEASURED: 'scale' is exact "
        "(max feature change 0.0 under 0.2x/5x rescaling) and changes nothing on "
        "the benchmark (77.9%% vs 78.6%%, McNemar p=1.000, +2 of 35 even on the "
        "shallowest depth quartile); 'drop' costs both sim-CV and transfer, so "
        "the model does use magnitude productively. The case for 'scale' is "
        "design, not evidence -- the benchmark is one dataset at one depth and "
        "structurally cannot test what invariance protects against.",
    )

    ap.add_argument(
        "--select-on", choices=("inner", "held-out"), default="inner",
        help="where the best epoch and early stopping are decided. 'inner' "
        "(default since 2026-08-17) carves a gene-disjoint validation split out "
        "of each training fold; the held-out fold is then scored exactly once, "
        "with the chosen epoch. 'held-out' is the old protocol, which chose the "
        "epoch on the fold it then reported -- a best-of-N maximum, not an "
        "estimate. Every sim-CV number from before that date is on 'held-out' "
        "and reads about 0.02 macro-F1 high; use this flag to reproduce one.",
    )
    ap.add_argument("--inner-val-frac", type=float, default=0.15,
                    help="fraction of each training fold's genes held out for epoch selection")
    ap.add_argument(
        "--refit", action="store_true",
        help="with --select-on inner: after choosing the epoch, retrain from "
        "scratch on the *whole* training fold for that many epochs. Selection "
        "still never sees the reported fold; the model just gets the inner-val "
        "15%% back. Roughly doubles training time. Use it to tell 'honest "
        "selection' apart from 'less training data' -- the two moved together "
        "when the protocol changed and the transfer drop needed attributing.",
    )

    ap.add_argument(
        "--avg-last-k", type=int, default=1,
        help="with --refit: average the weights of the refit run's last k epochs "
        "instead of taking the final ones. 1 disables it. Stage 1.2 located the "
        "transfer cost of honest selection in the epoch choice itself, and this "
        "is the cheap way to stop betting on a single epoch -- it costs nothing "
        "beyond holding k state dicts in RAM.",
    )

    ap.add_argument(
        "--two-head", action="store_true",
        help="factorize the head into a 4-way structural type and a binary "
        "is-differential output, predicting NN when the latter says not "
        "differential (plan Stage 1.3). Separates 'what structure is here', "
        "which the simulation gets right and which is depth-robust, from 'did "
        "it change', which is where the domain shift lives -- the sim trains "
        "|dPSI| ~ U(0.3, 0.8) against negatives pinned at exactly 0 while "
        "81.6%% of real curated events sit below 0.30.",
    )
    ap.add_argument("--diff-lambda", type=float, default=1.0,
                    help="weight on the is-differential term of the two-head loss")
    ap.add_argument("--domain-lambda", type=float, default=0.0,
                    help="Stage 17: weight on the gradient-reversal domain-adversarial "
                    "loss term (dsignal/model.py's domain_head, against real-vs-simulated "
                    "shortcuts). 0.0 (default) disables it entirely -- no domain_head is "
                    "even constructed -- and training is byte-for-byte the pre-Stage-17 "
                    "model")

    ap.add_argument(
        "--class-weight-boost", nargs=2, metavar=("CLASS", "FACTOR"), action="append", default=None,
        help="multiply one class's balanced cross-entropy weight by FACTOR, e.g. "
        "'--class-weight-boost CSE 2.0'. Repeatable. The cheap way to shift a "
        "decision boundary (plans/signal-native-v2-ensemble.md Stage 2's "
        "ens_cse member, for the CSE-vs-A3/A5 confusion) without touching "
        "features or architecture.",
    )

    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="do not hold normalized records in RAM")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json-out", default=None, help="write the aggregate metrics here")
    ap.add_argument("--dump-oof", default=None,
                    help="write per-event out-of-fold predictions (ix, event_id, truth, "
                    "pred) to this TSV -- lets a caller re-score under a different "
                    "labeling convention (e.g. dsignal.score_benchmark.apply_trash_as_nn) "
                    "without retraining")
    return ap


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    boost = {}
    for c, factor in (args.class_weight_boost or []):
        if c not in CLASSES:
            raise ValueError(f"--class-weight-boost: {c!r} is not one of {CLASSES}")
        boost[c] = float(factor)
    args.class_weight_boost = boost

    entries = read_labels(args.labels, args.npz_dir)
    labels = [e[2] for e in entries]
    gene_ids = [e[3] for e in entries]
    print(f"loaded {len(entries)} events", file=sys.stderr)
    print("class counts:", {c: labels.count(c) for c in CLASSES})
    print(f"augmentations: {args.augment or ['none (control)']}, feature subset: {args.feature_subset}, "
          f"depth norm: {args.depth_norm}, subsample_low: {args.subsample_low}, "
          f"epoch selection: {args.select_on}"
          + (f" ({args.inner_val_frac:.0%} of each training fold)" if args.select_on == "inner" else ""))

    splits = grouped_folds(np.zeros(len(entries)), labels, gene_ids, args.folds)

    t0 = time.time()
    oof_pred = np.empty(len(entries), dtype=object)
    oof_gold = np.empty(len(entries), dtype=object)
    fold_f1, fold_val_f1 = [], []
    for fold, (train_idx, test_idx) in enumerate(splits):
        best = train_fold(args, entries, train_idx, test_idx, fold, device)
        fold_f1.append(best["macro_f1"])
        fold_val_f1.append(best["val_macro_f1"])
        for pos, i in enumerate(test_idx):
            oof_pred[i] = CLASSES[best["pred"][pos]]
            oof_gold[i] = CLASSES[best["gold"][pos]]
        sel = (f" (epoch chosen on inner val, macro-F1 {best['val_macro_f1']:.3f})"
               if args.select_on == "inner" else " (epoch chosen on this same fold)")
        print(f"fold {fold}: held-out macro-F1 {best['macro_f1']:.3f} "
              f"at epoch {best['epoch']}{sel}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\ntrained {args.folds} folds in {elapsed / 60:.1f} min "
          f"(per-fold macro-F1 {' '.join(f'{f:.3f}' for f in fold_f1)})", file=sys.stderr)

    title = f"junction-set transformer [{args.feature_subset}] " \
            f"[aug: {','.join(args.augment) or 'none'}] ({args.folds}-fold group CV, by gene)"
    result = report(oof_gold, oof_pred, list(CLASSES), title)
    print_vs_reference(result)
    collapsed = report_collapsed(oof_gold, oof_pred, "transformer, A3+A5 collapsed to A")

    if args.dump_oof:
        with open(args.dump_oof, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ix", "event_id", "truth", "pred"])
            for (ix, event_id, _label, _gene_id, _path), truth, pred in zip(entries, oof_gold, oof_pred):
                w.writerow([ix, event_id, truth, pred])
        print(f"wrote out-of-fold predictions to {args.dump_oof}", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "augment": args.augment, "feature_subset": args.feature_subset,
                    "depth_norm": args.depth_norm, "subsample_low": args.subsample_low,
                    "folds": args.folds, "seed": args.seed,
                    "select_on": args.select_on, "inner_val_frac": args.inner_val_frac,
                    "refit": args.refit, "avg_last_k": args.avg_last_k,
                    "two_head": args.two_head, "diff_lambda": args.diff_lambda,
                    "domain_lambda": args.domain_lambda,
                    "class_weight_boost": args.class_weight_boost,
                    "per_fold_macro_f1": fold_f1,
                    "per_fold_inner_val_macro_f1": fold_val_f1,
                    "elapsed_min": elapsed / 60,
                    "result": result, "collapsed": collapsed,
                },
                f,
                indent=2,
            )
        print(f"wrote metrics to {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
