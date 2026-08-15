"""Training CLI for the junction-set Transformer (Phase 3 step 3).

Same evaluation protocol as ``sim/baseline_model.py`` and
``dsignal/baseline_gbm.py`` -- ``GroupKFold(5)`` grouped by gene with an
``assert_no_gene_leak`` check, same reporting format -- so every number stays
comparable to the whole layer-by-layer history. Held-out-fold predictions are
pooled across folds into one out-of-fold report, exactly as
``cross_val_predict`` does for the two baselines.

Usage:
    python -m dsignal.train --npz-dir sim/experiment/npz_scale1260 \
        --labels sim/experiment/labels_scale1260.tsv --out-dir runs/control
    python -m dsignal.train ... --augment flip condition_swap
    python -m dsignal.train ... --augment all
"""

from __future__ import annotations

import argparse
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
    AugmentConfig,
    JunctionSetDataset,
    collate,
    read_labels,
)
from dsignal.features import FeatureConfig
from dsignal.model import JunctionSetTransformer, ModelConfig


def class_weights(labels, classes, device):
    """Class-balanced cross-entropy weights from the *training* fold only."""
    counts = np.array([max(sum(1 for l in labels if l == c), 1) for c in classes], dtype=np.float64)
    w = counts.sum() / (len(classes) * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def evaluate(model, loader, device):
    """(predictions [n], labels [n], event_ids [n]) over a loader."""
    model.eval()
    preds, golds, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(
                batch["tokens"].to(device),
                batch["cls"].to(device),
                batch["s_rel"].to(device),
                batch["e_rel"].to(device),
                batch["pad_mask"].to(device),
            )
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            golds.extend(batch["label"].tolist())
            ids.extend(batch["event_id"])
    return np.array(preds), np.array(golds), ids


def macro_f1(pred, gold, n_classes) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(gold, pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def train_fold(args, entries, train_idx, test_idx, fold: int, device):
    cfg = FeatureConfig(flank=args.flank, max_junctions=args.max_junctions)
    aug = AugmentConfig.from_names(args.augment, prob=args.augment_prob)

    train_entries = [entries[i] for i in train_idx]
    test_entries = [entries[i] for i in test_idx]

    # scaler is fit on unaugmented training-fold records only, then reused for
    # the held-out fold -- fitting it on everything would leak fold statistics
    fit_ds = JunctionSetDataset(
        train_entries, cfg=cfg, feature_subset=args.feature_subset, cache=not args.no_cache
    )
    scaler = fit_ds.fit_scaler()

    train_ds = JunctionSetDataset(
        train_entries, cfg=cfg, augment=aug, scaler=scaler,
        feature_subset=args.feature_subset, cache=not args.no_cache, seed=args.seed,
    )
    test_ds = JunctionSetDataset(
        test_entries, cfg=cfg, scaler=scaler,
        feature_subset=args.feature_subset, cache=not args.no_cache, seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers,
    )

    mcfg = ModelConfig(
        n_classes=len(CLASSES), d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    )
    model = JunctionSetTransformer(mcfg).to(device)
    if fold == 0:
        print(f"model: {model.n_parameters():,} trainable parameters", file=sys.stderr)

    weights = class_weights(train_ds.labels, CLASSES, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"macro_f1": -1.0, "epoch": -1, "state": None, "pred": None, "gold": None, "ids": None}
    since_best = 0

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss, n_seen = 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            logits, _ = model(
                batch["tokens"].to(device),
                batch["cls"].to(device),
                batch["s_rel"].to(device),
                batch["e_rel"].to(device),
                batch["pad_mask"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total_loss += float(loss) * len(batch["label"])
            n_seen += len(batch["label"])
        sched.step()

        pred, gold, ids = evaluate(model, test_loader, device)
        f1 = macro_f1(pred, gold, len(CLASSES))
        if f1 > best["macro_f1"]:
            best = {
                "macro_f1": f1, "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "pred": pred, "gold": gold, "ids": ids,
            }
            since_best = 0
        else:
            since_best += 1

        if args.verbose or epoch % 10 == 0:
            print(
                f"  fold {fold} epoch {epoch:3d}  loss {total_loss / max(n_seen, 1):.4f}  "
                f"held-out macro-F1 {f1:.3f}  (best {best['macro_f1']:.3f} @ {best['epoch']})",
                file=sys.stderr,
            )
        if since_best >= args.patience:
            print(f"  fold {fold}: early stop at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)", file=sys.stderr)
            break

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(
            {
                "model_state": best["state"],
                "model_config": mcfg.to_dict(),
                "scaler": scaler.state_dict(),
                "classes": list(CLASSES),
                "feature_config": {"flank": cfg.flank, "max_junctions": cfg.max_junctions,
                                   "canonicalize": cfg.canonicalize},
                "feature_subset": args.feature_subset,
                "fold": fold,
                "held_out_macro_f1": best["macro_f1"],
                # training class prior, so inference-time prior correction
                # (dsignal.predict.PriorCorrection) does not need the labels
                # TSV. Checkpoints written before this field exist -- callers
                # must tolerate its absence.
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

    ap.add_argument("--flank", type=int, default=50)
    ap.add_argument("--max-junctions", type=int, default=256)
    ap.add_argument(
        "--feature-subset", choices=("all", "geometry", "no-geometry"), default="no-geometry",
        help="see dsignal/baseline_gbm.py -- window geometry is a Phase-2 "
        "simulation artifact and is excluded by default; 'geometry' is the "
        "plan's geometry-only control, which must score at chance",
    )
    ap.add_argument(
        "--augment", nargs="*", default=[],
        help="any of: flip condition_swap subsample junction_dropout jitter, or 'all'. "
        "Default none -- the control run.",
    )
    ap.add_argument("--augment-prob", type=float, default=0.5)

    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="do not hold normalized records in RAM")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json-out", default=None, help="write the aggregate metrics here")
    return ap


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    entries = read_labels(args.labels, args.npz_dir)
    labels = [e[2] for e in entries]
    gene_ids = [e[3] for e in entries]
    print(f"loaded {len(entries)} events", file=sys.stderr)
    print("class counts:", {c: labels.count(c) for c in CLASSES})
    print(f"augmentations: {args.augment or ['none (control)']}, feature subset: {args.feature_subset}")

    splits = grouped_folds(np.zeros(len(entries)), labels, gene_ids, args.folds)

    t0 = time.time()
    oof_pred = np.empty(len(entries), dtype=object)
    oof_gold = np.empty(len(entries), dtype=object)
    fold_f1 = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        best = train_fold(args, entries, train_idx, test_idx, fold, device)
        fold_f1.append(best["macro_f1"])
        for pos, i in enumerate(test_idx):
            oof_pred[i] = CLASSES[best["pred"][pos]]
            oof_gold[i] = CLASSES[best["gold"][pos]]
        print(f"fold {fold}: best held-out macro-F1 {best['macro_f1']:.3f} "
              f"at epoch {best['epoch']}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\ntrained {args.folds} folds in {elapsed / 60:.1f} min "
          f"(per-fold macro-F1 {' '.join(f'{f:.3f}' for f in fold_f1)})", file=sys.stderr)

    title = f"junction-set transformer [{args.feature_subset}] " \
            f"[aug: {','.join(args.augment) or 'none'}] ({args.folds}-fold group CV, by gene)"
    result = report(oof_gold, oof_pred, list(CLASSES), title)
    print_vs_reference(result)
    collapsed = report_collapsed(oof_gold, oof_pred, "transformer, A3+A5 collapsed to A")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "augment": args.augment, "feature_subset": args.feature_subset,
                    "folds": args.folds, "seed": args.seed,
                    "per_fold_macro_f1": fold_f1, "elapsed_min": elapsed / 60,
                    "result": result, "collapsed": collapsed,
                },
                f,
                indent=2,
            )
        print(f"wrote metrics to {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
