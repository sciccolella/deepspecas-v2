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

from dsignal.dataset import apply_feature_subset, collate, FeatureScaler
from dsignal.extract import ReadFilters, extract_event, parse_region
from dsignal.features import FeatureConfig, build_features
from dsignal.model import JunctionSetTransformer, ModelConfig


class Predictor:
    """One or more trained folds, applied to extractor records.

    Feature config, scaler, class list and feature subset all come from the
    checkpoint, never from CLI defaults -- applying a different transform at
    inference than at training is the classic silent way to get garbage.
    """

    def __init__(self, checkpoint_paths, device=None, filters: ReadFilters | None = None):
        if not checkpoint_paths:
            raise ValueError("no checkpoints given")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.filters = filters or ReadFilters()
        self.models, self.scalers = [], []
        self.classes = None
        self.feature_subset = None
        self.cfg = None

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
            elif (classes, subset) != (self.classes, self.feature_subset):
                raise ValueError(
                    f"{path}: checkpoints disagree -- classes {classes} vs {self.classes}, "
                    f"feature subset {subset!r} vs {self.feature_subset!r}"
                )

    def predict_record(self, record: dict, need_weights: bool = False):
        """(label, probability vector, attention or None) for one record."""
        tokens_raw, cls_raw, meta = build_features(record, self.cfg)
        probs = np.zeros(len(self.classes), dtype=np.float64)
        attn_out = None

        for model, scaler in zip(self.models, self.scalers):
            tokens, cls = apply_feature_subset(tokens_raw, cls_raw, self.feature_subset)
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
                    }
                ]
            )
            with torch.no_grad():
                logits, attn = model(
                    batch["tokens"].to(self.device),
                    batch["cls"].to(self.device),
                    batch["s_rel"].to(self.device),
                    batch["e_rel"].to(self.device),
                    batch["pad_mask"].to(self.device),
                    need_weights=need_weights,
                )
            probs += logits.softmax(dim=1)[0].cpu().numpy()
            if need_weights and attn is not None and attn_out is None:
                attn_out = attn[0].cpu().numpy()

        probs /= len(self.models)
        return self.classes[int(probs.argmax())], probs, attn_out

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
    ap.add_argument("--min-mapq", type=int, default=0)
    args = ap.parse_args(argv)

    predictor = Predictor(
        args.net,
        device=args.device,
        filters=ReadFilters(min_anchor=args.min_anchor, max_nh=args.max_nh, min_mapq=args.min_mapq),
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
