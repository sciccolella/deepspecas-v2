"""Signal-native inference: `region,bam1,bam2` CSV in, predictions out
(Phase 3 step 5).

Takes the same input format as ``dsEnsemble.py`` / ``deepSpecas.py predict``, so
the v1 image-ensemble path and the v2 signal-native path are a drop-in
comparison. Unlike v1 there is no image rendering step at all: a record is
extracted straight from the BAMs (``dsignal.extract``), turned into junction
tokens (``dsignal.features``) and classified. Wall-clock per event is reported
because that comparison is the point -- v1's ensemble spends ~17s/event on
rendering alone.

Several checkpoints can be given at once (e.g. all five GroupKFold folds), in
which case their softmax outputs are averaged -- the same ensembling idea as
``dsEnsemble.py``, over folds rather than over image representations.

``--tta-condition-swap`` additionally averages each fold over both condition
orientations. The swap is exactly label-invariant, so it is free test-time
augmentation rather than a guess. Measured on the dev half of the benchmark
(2026-08-17, runs/benchmark_events_tta_dev.tsv vs ..._innerval_refit_dev.tsv):
49/70 -> 51/70 collapsed, 3 gained / 1 lost, McNemar p = 0.625 -- the right
sign, not significant at n=70, and it doubles inference time. Opt-in.

Usage:
    python -m dsignal.predict --csv benchmark/A3.csv --net runs/control/fold*.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np
import torch

from dsignal.dataset import (
    apply_feature_subset,
    apply_feature_subset_meta,
    augment_condition_swap,
    collate,
    FeatureScaler,
)
from dsignal.extract import ReadFilters, extract_event, parse_region
from dsignal.features import (
    FeatureConfig,
    build_coverage_profile,
    build_features,
    normalize_record,
)
from dsignal.model import JunctionSetTransformer, ModelConfig, batch_inputs


class PriorCorrection:
    """Logit adjustment for a train/deploy class-prior mismatch.

    ``p'(y|x) ∝ p(y|x) · (p_target(y) / p_train(y))^tau``, which is exactly
    ``softmax(logits + tau·log(p_target/p_train))`` -- doing it on the
    probability vector is equivalent and lets the same object be applied to
    cached per-fold probabilities.

    **Must be applied per fold, before the folds are averaged.** Correcting the
    averaged probability is a different (and wrong) estimator: the average of
    corrected posteriors is not the correction of the averaged posterior.

    Motivation: the simulation design carries a large NN fraction (43% at the
    1,260-event pilot, 25% in the current 4,800-event design) while the curated
    benchmark contains no NN events at all, and false NN is the dominant real
    error -- 41 of 140 under the pilot, and still the largest single bucket at
    13 of 140 under the current v3-window model. ``tau`` scales the correction:
    0 is a no-op, 1 is the full adjustment.

    **Open caveat.** The strength that scores best on the benchmark is currently
    chosen by *reading* benchmark accuracy, which is test-set fitting. There is
    a legitimate version of the argument -- deployment inputs are pre-filtered
    rMATS/SUPPA candidates, so a low NN prior is defensible a priori -- but it
    has to be argued from the deployment setting and fixed before scoring.
    """

    def __init__(self, train_prior, target_prior, tau: float = 1.0):
        train = np.asarray(train_prior, dtype=np.float64)
        target = np.asarray(target_prior, dtype=np.float64)
        if train.shape != target.shape:
            raise ValueError(f"prior shape mismatch: train {train.shape} vs target {target.shape}")
        if (train <= 0).any():
            raise ValueError("training prior has a zero/negative entry -- a class the model never saw")
        if (target < 0).any():
            raise ValueError("target prior has a negative entry")
        self.train_prior = train / train.sum()
        self.target_prior = target / target.sum()
        self.tau = float(tau)
        #: weight applied to each class's probability, before renormalization
        self.weights = (self.target_prior / self.train_prior) ** self.tau

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Corrected, renormalized probabilities. Works on a [C] vector or a
        [F, C] per-fold matrix (each row corrected independently)."""
        out = np.asarray(probs, dtype=np.float64) * self.weights
        total = out.sum(axis=-1, keepdims=True)
        # a target prior that zeroes every class the model put mass on would
        # leave nothing to renormalize; fall back to the uncorrected vector
        # rather than emitting NaNs
        safe = np.where(total > 0, total, 1.0)
        return np.where(total > 0, out / safe, probs)

    def __repr__(self):
        return (f"PriorCorrection(tau={self.tau}, "
                f"train={np.round(self.train_prior, 3).tolist()}, "
                f"target={np.round(self.target_prior, 3).tolist()})")


def class_counts_from_labels(labels_path: str, classes) -> np.ndarray:
    """Training class counts read off a labels TSV (the checkpoints predating
    this feature do not carry them)."""
    counts = {c: 0 for c in classes}
    with open(labels_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["label"] in counts:
                counts[row["label"]] += 1
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise ValueError(f"{labels_path}: no rows for class(es) {missing}")
    return np.array([counts[c] for c in classes], dtype=np.float64)


def make_target_prior(classes, kind: str = "uniform", nn_weight: float | None = None) -> np.ndarray:
    """``uniform``: every class equally likely at deploy time -- domain-agnostic
    and deployable. ``benchmark``: NN excluded entirely, an **oracle** setting
    that presumes the truth set contains no negatives; useful only as an upper
    bound. ``nn-weight``: keep the classes uniform but scale NN by
    ``nn_weight``, which traces the curve between the two."""
    classes = list(classes)
    prior = np.ones(len(classes), dtype=np.float64)
    if kind == "uniform":
        return prior
    if kind == "benchmark":
        prior[classes.index("NN")] = 0.0
        return prior
    if kind == "nn-weight":
        if nn_weight is None:
            raise ValueError("kind 'nn-weight' needs nn_weight")
        prior[classes.index("NN")] = float(nn_weight)
        return prior
    raise ValueError(f"unknown target prior {kind!r}")


class Predictor:
    """One or more trained folds of **one run**, applied to extractor records.

    Feature config, scaler, class list and feature subset all come from the
    checkpoint, never from CLI defaults -- applying a different transform at
    inference than at training is the classic silent way to get garbage. All
    checkpoints given to one Predictor must therefore agree on all three:
    mixing runs that differ in ``--flank``/``--depth-norm``/feature subset
    belongs to ``dsignal.ensemble.MultiRunPredictor``, which holds one
    Predictor per run instead of forcing them into a single flat fold list.
    """

    def __init__(self, checkpoint_paths, device=None, filters: ReadFilters | None = None,
                 prior_correction: "PriorCorrection | list[PriorCorrection] | None" = None,
                 tta_condition_swap: bool = False):
        if not checkpoint_paths:
            raise ValueError("no checkpoints given")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.filters = filters or ReadFilters()
        #: average the two condition orientations at inference (see _views)
        self.tta_condition_swap = bool(tta_condition_swap)
        #: optional train/deploy prior adjustment, applied per fold (see
        #: PriorCorrection). Either one correction shared by every fold, or a
        #: sequence with one entry per fold (see build_prior_correction).
        self.prior_correction = prior_correction
        self.models, self.scalers = [], []
        self.classes = None
        self.feature_subset = None
        self.cfg = None
        #: Stage 12 / Track D. Read off the checkpoints' model config, never
        #: from a CLI default -- 0 means the branch is off, which is what every
        #: checkpoint before Stage 12 carries.
        self.coverage_bins = None
        #: per-checkpoint training class counts, in ``self.classes`` order,
        #: for checkpoints that carry ``train_class_counts``; None entries for
        #: those predating the field (see build_prior_correction).
        self.train_class_counts = []

        for path in checkpoint_paths:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            model = JunctionSetTransformer(ModelConfig(**ck["model_config"])).to(self.device)
            model.load_state_dict(ck["model_state"])
            model.eval()
            self.models.append(model)
            self.scalers.append(FeatureScaler.from_state_dict(ck["scaler"]))

            classes = tuple(ck["classes"])
            subset = ck["feature_subset"]
            fcfg = FeatureConfig(**ck["feature_config"])
            if self.classes is None:
                self.classes, self.feature_subset, self.cfg = classes, subset, fcfg
            elif (classes, subset, fcfg) != (self.classes, self.feature_subset, self.cfg):
                raise ValueError(
                    f"{path}: checkpoints disagree -- classes {classes} vs {self.classes}, "
                    f"feature subset {subset!r} vs {self.feature_subset!r}, "
                    f"feature config {fcfg} vs {self.cfg}. All checkpoints passed to one "
                    f"Predictor must be folds of the same run; combine different runs via "
                    f"dsignal.ensemble.MultiRunPredictor instead"
                )

            cov_bins = int(ck["model_config"].get("coverage_bins", 0))
            if self.coverage_bins is None:
                self.coverage_bins = cov_bins
            elif cov_bins != self.coverage_bins:
                raise ValueError(
                    f"{path}: checkpoints disagree on coverage_bins ({cov_bins} vs "
                    f"{self.coverage_bins}); folds of one run must agree"
                )

            counts = ck.get("train_class_counts")
            self.train_class_counts.append(
                np.array([counts[c] for c in self.classes], dtype=np.float64) if counts else None
            )

    def build_prior_correction(self, kind: str, nn_weight: float | None = None,
                                tau: float = 1.0, train_labels: str | None = None):
        """A PriorCorrection for this run's checkpoints.

        With ``train_labels`` given, one correction is built from that file's
        class counts and shared by every fold -- the original, still-default
        behaviour, bit-exact with every stored number that used it. Without
        it, one correction is built **per fold** from that checkpoint's own
        ``train_class_counts`` (H2): folds of a GroupKFold run see slightly
        different training data, so this is the more accurate estimator, and
        it is what lets a run be scored without also carrying its labels TSV
        around. Raises if any checkpoint predates the field.
        """
        target = make_target_prior(self.classes, kind, nn_weight=nn_weight)
        if train_labels is not None:
            train_counts = class_counts_from_labels(train_labels, self.classes)
            return PriorCorrection(train_counts, target, tau)
        missing = [i for i, c in enumerate(self.train_class_counts) if c is None]
        if missing:
            raise ValueError(
                f"checkpoint(s) at index {missing} carry no train_class_counts; "
                f"pass train_labels explicitly for this run"
            )
        return [PriorCorrection(c, target, tau) for c in self.train_class_counts]

    def _views(self, record: dict):
        """Feature views of one record: the record itself, plus -- when
        ``tta_condition_swap`` is on -- the same record with the two conditions
        exchanged. That transform is exactly label-invariant (the label is the
        event *type*, not the direction of the PSI change; see
        ``dsignal.dataset.augment_condition_swap``), so averaging the two
        orientations is free test-time augmentation, not a guess."""
        norm = normalize_record(record, self.cfg)
        norms = [norm]
        if self.tta_condition_swap:
            norms.append(augment_condition_swap(norm))
        out = []
        for n in norms:
            tokens_raw, cls_raw, meta = build_features(norm=n, cfg=self.cfg)
            # built from the same (possibly condition-swapped) view, so TTA
            # applies to the coverage branch exactly as it does to the tokens
            cov = build_coverage_profile(n, self.coverage_bins) if self.coverage_bins else None
            out.append((tokens_raw, cls_raw, apply_feature_subset_meta(meta, self.feature_subset), cov))
        return out

    def fold_probs(self, record: dict, need_weights: bool = False):
        """``([F, C] per-fold softmax, attention or None)`` -- uncorrected.

        Kept separate from `predict_record` so a prior sweep can extract and
        run each event once and then evaluate many corrections analytically:
        the correction is a function of these per-fold posteriors alone.

        Under TTA the views are averaged *within* each fold, so what this
        returns is still one posterior per fold and everything downstream --
        prior correction, fold averaging -- is unchanged.
        """
        views = self._views(record)
        per_fold = np.zeros((len(self.models), len(self.classes)), dtype=np.float64)
        attn_out = None

        for f, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            for tokens_raw, cls_raw, meta, cov in views:
                probs, attn = self._forward(model, scaler, tokens_raw, cls_raw, meta, need_weights, cov)
                per_fold[f] += probs / len(views)
                if need_weights and attn is not None and attn_out is None:
                    attn_out = attn

        return per_fold, attn_out

    def _forward(self, model, scaler, tokens_raw, cls_raw, meta, need_weights: bool, cov=None):
        """One (fold, view) forward pass -> (softmax [C], attention or None)."""
        tokens, cls = apply_feature_subset(tokens_raw, cls_raw, self.feature_subset, self.cfg)
        tokens, cls = scaler.transform(tokens, cls)
        batch = collate(
            [
                {
                    "tokens": torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.float32)),
                    "cls": torch.from_numpy(np.ascontiguousarray(cls, dtype=np.float32)),
                    "s_rel": torch.from_numpy(meta["s_rel"]),
                    "e_rel": torch.from_numpy(meta["e_rel"]),
                    "label": -1,
                    "event_id": "",
                    "ix": "",
                    **({"coverage": torch.from_numpy(cov)} if cov is not None else {}),
                }
            ]
        )
        with torch.no_grad():
            logits, attn = model(
                **batch_inputs(batch, self.device), need_weights=need_weights
            )
        return (logits.softmax(dim=1)[0].cpu().numpy(),
                attn[0].cpu().numpy() if (need_weights and attn is not None) else None)

    def combine(self, per_fold: np.ndarray):
        """Per-fold posteriors -> (label, averaged probability vector).

        The prior correction is applied to each fold *before* averaging -- see
        PriorCorrection. With no correction configured this is the plain fold
        ensemble, bit-for-bit what the model reported before the feature
        existed. ``self.prior_correction`` may be a single PriorCorrection
        (applied identically to every fold, the original behaviour) or a
        sequence with one PriorCorrection per fold (build_prior_correction's
        per-checkpoint estimator).
        """
        pc = self.prior_correction
        if isinstance(pc, PriorCorrection):
            per_fold = pc.apply(per_fold)
        elif pc is not None:
            if len(pc) != len(per_fold):
                raise ValueError(f"{len(pc)} prior corrections for {len(per_fold)} folds")
            per_fold = np.stack([c.apply(row) for c, row in zip(pc, per_fold)])
        probs = per_fold.mean(axis=0)
        return self.classes[int(probs.argmax())], probs

    def predict_record(self, record: dict, need_weights: bool = False):
        """(label, probability vector, attention or None) for one record."""
        per_fold, attn_out = self.fold_probs(record, need_weights)
        label, probs = self.combine(per_fold)
        return label, probs, attn_out

    def predict_region(self, region_str: str, bam1: str, bam2: str, need_weights: bool = False):
        """Extract from the BAMs and classify. Returns (label, probs, attn,
        timing dict) -- extraction and inference timed separately, since the
        v1 comparison is really about the rendering step v2 does not have."""
        t0 = time.perf_counter()
        record = extract_event(parse_region(region_str), bam1, bam2, filters=self.filters)
        t1 = time.perf_counter()
        label, probs, attn = self.predict_record(record, need_weights)
        t2 = time.perf_counter()
        return label, probs, attn, {"extract_s": t1 - t0, "infer_s": t2 - t1, "total_s": t2 - t0}


def read_prediction_csv(path: str):
    with open(path) as f:
        return [(r[0], r[1], r[2]) for r in csv.reader(f) if r]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="region,bam1,bam2 CSV, no header")
    ap.add_argument("--net", nargs="+", required=True, help="one or more fold checkpoints (.pt)")
    ap.add_argument("--out", default=None, help="write region,label,probs TSV here")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-anchor", type=int, default=5)
    ap.add_argument("--max-nh", type=int, default=10)
    ap.add_argument(
        "--tta-condition-swap", action="store_true",
        help="average the prediction over both condition orientations. The swap "
        "is exactly label-invariant (the label is the event type, not the "
        "direction of the change), so this is free test-time augmentation; it "
        "doubles inference time and leaves extraction untouched",
    )
    ap.add_argument("--min-mapq", type=int, default=0)
    args = ap.parse_args(argv)

    predictor = Predictor(
        args.net,
        device=args.device,
        filters=ReadFilters(min_anchor=args.min_anchor, max_nh=args.max_nh, min_mapq=args.min_mapq),
        tta_condition_swap=args.tta_condition_swap,
    )
    print(f"loaded {len(predictor.models)} fold checkpoint(s), classes {predictor.classes}, "
          f"feature subset {predictor.feature_subset!r}", file=sys.stderr)

    rows = read_prediction_csv(args.csv)
    out_rows, timings = [], []
    for region, bam1, bam2 in rows:
        label, probs, _, timing = predictor.predict_region(region, bam1, bam2)
        timings.append(timing)
        prob_str = " ".join(f"{c}={p:.3f}" for c, p in zip(predictor.classes, probs))
        print(f"{region}\t{label}\t{prob_str}\t{timing['total_s']:.3f}s")
        out_rows.append([region, label] + [f"{p:.6f}" for p in probs])

    if timings:
        tot = np.array([t["total_s"] for t in timings])
        ext = np.array([t["extract_s"] for t in timings])
        inf = np.array([t["infer_s"] for t in timings])
        print(
            f"\n{len(tot)} events: {tot.mean():.3f} s/event total "
            f"({ext.mean():.3f} extract + {inf.mean():.3f} inference), "
            f"{tot.sum():.1f} s wall clock",
            file=sys.stderr,
        )

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["region", "prediction"] + [f"p_{c}" for c in predictor.classes])
            w.writerows(out_rows)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
