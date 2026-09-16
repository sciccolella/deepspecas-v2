"""Stage 22b.3: self-supervised pretraining on unlabeled real windows.

Two pretext tasks, both defined on `dsignal.model.JunctionSetTransformer
.forward_pretrain` and `dsignal.dataset.SSLPretrainDataset`'s two
independently-augmented views of the same window (see both modules'
docstrings and `plans/signal-native-v2-stage22b-ssl-pretrain.md` §Design):

1. **Masked junction-feature reconstruction (MFM)** -- MSE between
   `recon_head`'s prediction at each view's masked junction positions and
   that view's own true (scaled) token features. Computed on both views and
   averaged.
2. **Augmentation contrastive (NT-Xent / SimCLR)** -- the two views' pooled
   CLS embeddings, projected through `proj_head` and L2-normalized, are a
   positive pair; every other window in the batch (and its own two views)
   supplies negatives.

`L = alpha * L_mfm + beta * L_contrastive`, both always computed (so a
variant is selected purely by its loss weight -- MFM-only is `--beta 0`,
contrastive-only is `--alpha 0`, both is the default) -- three pretraining
variants are an ablation of this one script, not three different code paths.

No sim data and no labels enter this stage at all (the labels.tsv `label`
column from `sim.sample_unlabeled_windows` is empty and never read), so
there is no real-vs-sim shortcut for the model to exploit, unlike Stage 15's
classifier-training failure mode.

A gene-disjoint slice of the SSL pool (`--val-frac`, via `sim.gene_split
.split_genes`) is held out for loss monitoring only -- convergence sanity,
never model selection and never the curated benchmark.

Output is a single checkpoint (no folds; SSL pretraining runs once) with
`model_state`/`model_config`/`feature_config`/`feature_subset`/`scaler`,
the same shape `dsignal.train`'s per-fold checkpoints use for those keys, so
`dsignal.train --init-encoder-from` can load it with the same
config-matching asserts `load_parent` already applies.

Usage::

    python -m dsignal.pretrain \\
        --npz-dir sim/experiment/real_ssl_pool/npz \\
        --labels sim/experiment/real_ssl_pool/labels.tsv \\
        --alpha 1.0 --beta 1.0 --mask-frac 0.3 --val-frac 0.1 \\
        --augment all --feature-subset no-geometry \\
        --epochs 30 --batch-size 64 --lr 3e-4 --seed 0 \\
        --out runs/ssl_both.pt --json-out runs/ssl_both.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dsignal.dataset import AugmentConfig, FeatureScaler, SSLPretrainDataset, collate_ssl, read_labels
from dsignal.features import FeatureConfig
from dsignal.model import JunctionSetTransformer, ModelConfig, batch_inputs
from sim.gene_split import assert_no_gene_leak, split_genes


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard SimCLR NT-Xent over a batch of paired views. ``z1``/``z2``
    ``[B, D]``, already L2-normalized. The matched ``(z1[i], z2[i])`` pair is
    the positive; every other one of the ``2B - 2`` embeddings in the batch
    is a negative, exactly as `plan Stage 22b`'s "Augmentation contrastive"
    design describes."""
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = (z @ z.T) / temperature  # [2B, 2B]
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, target)


def step_losses(model: JunctionSetTransformer, batch: dict, device, temperature: float):
    """``(l_mfm, l_contrastive)`` for one two-view batch. MFM pools both
    views' masked-position reconstructions into a single MSE; contrastive
    needs both views' pooled vectors, which is exactly why
    `forward_pretrain` returns the unprojected pooled CLS rather than
    computing NT-Xent itself."""
    v1, v2 = batch["view1"], batch["view2"]
    inputs1, inputs2 = batch_inputs(v1, device), batch_inputs(v2, device)
    mask1, mask2 = v1["mask"].to(device), v2["mask"].to(device)

    recon1, pooled1 = model.forward_pretrain(mask=mask1, **inputs1)
    recon2, pooled2 = model.forward_pretrain(mask=mask2, **inputs2)

    target1, target2 = inputs1["tokens"][mask1], inputs2["tokens"][mask2]
    recon_all = torch.cat([recon1, recon2], dim=0)
    target_all = torch.cat([target1, target2], dim=0)
    l_mfm = F.mse_loss(recon_all, target_all) if target_all.numel() else recon_all.new_zeros(())

    z1 = F.normalize(model.proj_head(pooled1), dim=-1)
    z2 = F.normalize(model.proj_head(pooled2), dim=-1)
    l_contrastive = nt_xent(z1, z2, temperature)

    return l_mfm, l_contrastive


def run_epoch(model, loader, device, alpha: float, beta: float, temperature: float,
              optimizer=None, clip: float = 1.0):
    """One pass over ``loader``; trains if ``optimizer`` is given, otherwise
    a no-grad eval pass. Returns mean (total, mfm, contrastive) losses,
    batch-count-weighted (every batch but the last is the same size, so this
    matches a sample-weighted mean closely enough for a monitoring metric)."""
    train = optimizer is not None
    model.train(train)
    totals = {"total": 0.0, "mfm": 0.0, "contrastive": 0.0}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            l_mfm, l_contrastive = step_losses(model, batch, device, temperature)
            loss = alpha * l_mfm + beta * l_contrastive
            if train:
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            totals["total"] += float(loss.detach())
            totals["mfm"] += float(l_mfm.detach())
            totals["contrastive"] += float(l_contrastive.detach())
            n += 1
    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1,
                     help="gene-disjoint slice of the pool held out for loss monitoring only "
                     "(sim.gene_split.split_genes); never used for model selection")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--alpha", type=float, default=1.0, help="weight on the MFM loss")
    ap.add_argument("--beta", type=float, default=1.0, help="weight on the contrastive loss")
    ap.add_argument("--mask-frac", type=float, default=0.3,
                     help="fraction of each view's junction tokens masked for MFM")
    ap.add_argument("--temperature", type=float, default=0.1, help="NT-Xent temperature")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")

    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--coverage-bins", type=int, default=0)
    ap.add_argument("--coverage-channels", type=int, default=64)

    ap.add_argument("--flank", type=int, default=50)
    ap.add_argument("--max-junctions", type=int, default=256)
    ap.add_argument("--feature-subset", default="no-geometry",
                     help="see dsignal.train's flag of the same name; the SSL run's value is "
                     "what --init-encoder-from's downstream classifier should also use, though "
                     "nothing enforces that match since the two are independent training runs")
    ap.add_argument("--augment", nargs="*", default=["all"],
                     help="any of: flip condition_swap subsample junction_dropout jitter, or "
                     "'all' (default). Unlike dsignal.train, the default here is 'all', not "
                     "none -- the contrastive task needs the two views to actually differ")
    ap.add_argument("--augment-prob", type=float, default=0.5)
    ap.add_argument("--subsample-low", type=float, default=0.3)

    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if args.alpha <= 0 and args.beta <= 0:
        ap.error("--alpha and --beta cannot both be <= 0 -- nothing to train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    entries = read_labels(args.labels, args.npz_dir)
    if not entries:
        raise ValueError(f"no entries found: {args.labels} / {args.npz_dir}")
    print(f"{len(entries)} unlabeled windows", file=sys.stderr)

    gene_ids = [e[3] for e in entries]
    assignment = split_genes(gene_ids, val_frac=args.val_frac, test_frac=0.0, seed=args.seed)
    train_entries = [e for e in entries if assignment[e[3]] == "train"]
    val_entries = [e for e in entries if assignment[e[3]] == "val"]
    assert_no_gene_leak(
        [{"gene_id": e[3], "split": assignment[e[3]]} for e in entries])
    print(f"train {len(train_entries)} / val (monitoring only) {len(val_entries)}, "
          f"{len(set(gene_ids))} genes", file=sys.stderr)

    cfg = FeatureConfig(flank=args.flank, max_junctions=args.max_junctions)
    aug = AugmentConfig.from_names(args.augment, prob=args.augment_prob, subsample_low=args.subsample_low)

    def make_ds(ents):
        return SSLPretrainDataset(
            ents, cfg=cfg, augment=aug, mask_frac=args.mask_frac,
            feature_subset=args.feature_subset, cache=not args.no_cache,
            seed=args.seed, coverage_bins=args.coverage_bins,
        )

    train_ds = make_ds(train_entries)
    # the scaler is fit on the training split's *unaugmented* raw features,
    # same convention as dsignal.train / JunctionSetDataset.fit_scaler
    scaler = train_ds.fit_scaler()
    val_ds = make_ds(val_entries)
    val_ds.scaler = scaler

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_ssl,
                          num_workers=args.num_workers, drop_last=shuffle)

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False) if val_entries else None

    from dsignal.features import N_EVENT_FEATURES, N_TOKEN_FEATURES

    mcfg = ModelConfig(
        n_token_features=N_TOKEN_FEATURES, n_event_features=N_EVENT_FEATURES,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        coverage_bins=args.coverage_bins, coverage_channels=args.coverage_channels,
    )
    model = JunctionSetTransformer(mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        train_losses = run_epoch(model, train_loader, device, args.alpha, args.beta,
                                 args.temperature, optimizer=optimizer, clip=args.clip)
        if val_loader is not None:
            val_ds.set_epoch(epoch)
            val_losses = run_epoch(model, val_loader, device, args.alpha, args.beta, args.temperature)
        else:
            val_losses = {"total": float("nan"), "mfm": float("nan"), "contrastive": float("nan")}
        history.append({"epoch": epoch, "train": train_losses, "val": val_losses})
        print(f"epoch {epoch:3d}  train total={train_losses['total']:.4f} "
              f"mfm={train_losses['mfm']:.4f} contrastive={train_losses['contrastive']:.4f}  "
              f"val total={val_losses['total']:.4f} mfm={val_losses['mfm']:.4f} "
              f"contrastive={val_losses['contrastive']:.4f}", file=sys.stderr)

    elapsed = time.time() - t0
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": mcfg.to_dict(),
            "scaler": scaler.state_dict(),
            "feature_config": {"flank": cfg.flank, "max_junctions": cfg.max_junctions,
                               "canonicalize": cfg.canonicalize,
                               "depth_norm": cfg.depth_norm,
                               "depth_target": cfg.depth_target,
                               "event_groups": list(cfg.event_groups)},
            "feature_subset": args.feature_subset,
            "alpha": args.alpha,
            "beta": args.beta,
            "mask_frac": args.mask_frac,
            "temperature": args.temperature,
            "augment": list(args.augment),
            "seed": args.seed,
            "epochs": args.epochs,
            "n_train": len(train_entries),
            "n_val": len(val_entries),
            "final_train_loss": history[-1]["train"] if history else None,
            "final_val_loss": history[-1]["val"] if history else None,
        },
        args.out,
    )
    print(f"wrote {args.out}", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "npz_dir": args.npz_dir, "labels": args.labels,
                "alpha": args.alpha, "beta": args.beta, "mask_frac": args.mask_frac,
                "n_train": len(train_entries), "n_val": len(val_entries),
                "elapsed_sec": elapsed, "history": history,
            }, f, indent=2)
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
