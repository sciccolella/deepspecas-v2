"""Combine multiple trained runs into one ensemble (Stage 0.3,
plans/signal-native-v2-ensemble.md).

``dsignal.predict.Predictor`` already averages *folds of one run* -- what is
new here is averaging **runs**, each potentially differing in seed,
augmentation, feature subset, `--flank`/`--depth-norm`, or training data
entirely. An ``EnsembleSpec`` names the member runs (each a directory of
``fold*.pt`` checkpoints); ``MultiRunPredictor`` holds one ``Predictor`` per
member (so H1/H3's config checks apply *within* a run, never across runs) and
combines their fold-averaged, prior-corrected posteriors.

Extraction happens **once per event**, then is fanned out to every member's
``fold_probs`` -- extraction dominates wall clock (see ``predict.py``'s
docstring), so a member count never multiplies it.

Three combination rules, over member posteriors already averaged within their
own run:

* ``mean_prob`` (default) -- the plain arithmetic mean, the estimator
  ``dsignal.predict.Predictor`` already uses over folds.
* ``mean_logprob`` -- weighted geometric mean (softmax of the weighted
  average of ``log(prob)``).
* ``sum_logit`` -- included only so v1's rule (``dsEnsemble.py``: sum raw
  logits, softmax once) is on the record for comparison. ``Predictor``
  exposes softmax outputs, not raw logits, and ``log(prob)`` is a valid
  logit representative only up to a per-model additive constant that softmax
  discards -- so this is an approximation of v1's rule, not a reproduction of
  it, and is expected to inherit its known miscalibration-sensitivity
  (larger-logit-magnitude nets dominate) rather than fix it.

Prior correction is applied **per fold, before fold-averaging, inside each
member's own Predictor** (unchanged from ``predict.py``) -- the cross-run
combine above operates on posteriors that are already correct in that sense.

The posterior cache (``--cache-out``/``--cache-in``) stores each member's raw
per-fold posteriors (pre-correction, pre-average) keyed by ``(csv, region)``,
modeled on ``prior_sweep.cache_fold_probs``/``evaluate``: one extraction pass,
then every rule/weighting/prior setting evaluated analytically from the cache
-- what makes Stage 4's rule-and-weight sweep cost minutes, not hours.

Usage::

    python -m dsignal.ensemble score --spec runs/ensembles/e1.json --half dev \\
        --tta-condition-swap --target-prior nn-weight --nn-weight 0.1 \\
        --cache-out runs/ens_cache_dev.npz
    python -m dsignal.ensemble score --spec runs/ensembles/e1.json --half dev \\
        --cache-in runs/ens_cache_dev.npz --rule mean_logprob

Or via ``dsignal.score_benchmark --ensemble runs/ensembles/e1.json``, which
reuses ``score``/``report`` unchanged so the per-event TSV schema stays
identical to the single-run path.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from os.path import basename, join as pjoin, splitext

import numpy as np

from dsignal.extract import ReadFilters, extract_event, parse_region
from dsignal.predict import Predictor, read_prediction_csv


@dataclass
class Member:
    run_dir: str
    #: labels TSV the run was trained on, for a shared-across-folds prior
    #: (Predictor.build_prior_correction's train_labels path). None (the
    #: default) uses each checkpoint's own stored train_class_counts instead.
    labels: str | None = None
    weight: float = 1.0


@dataclass
class EnsembleSpec:
    members: list[Member] = field(default_factory=list)
    name: str = "ensemble"

    @classmethod
    def load(cls, path: str) -> "EnsembleSpec":
        with open(path) as f:
            data = json.load(f)
        members = [Member(**m) for m in data["members"]]
        return cls(members=members, name=data.get("name", splitext(basename(path))[0]))

    def save(self, path: str) -> None:
        data = {"name": self.name,
                "members": [{"run_dir": m.run_dir, "labels": m.labels, "weight": m.weight}
                            for m in self.members]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")


def _fold_paths(run_dir: str) -> list[str]:
    paths = sorted(glob.glob(pjoin(run_dir, "fold*.pt")))
    if not paths:
        raise ValueError(f"{run_dir}: no fold*.pt checkpoints found")
    return paths


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def combine(member_probs: np.ndarray, rule: str, weights) -> np.ndarray:
    """``[M, C]`` per-member posteriors, already fold-averaged and
    prior-corrected within their own run, -> combined ``[C]`` vector."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (member_probs.shape[0],):
        raise ValueError(f"{len(weights)} weights for {member_probs.shape[0]} members")
    w = w / w.sum()
    if rule == "mean_prob":
        return (member_probs * w[:, None]).sum(axis=0)
    logp = np.log(np.clip(member_probs, 1e-12, None))
    if rule == "mean_logprob":
        return _softmax((logp * w[:, None]).sum(axis=0))
    if rule == "sum_logit":
        # weighted, but summed rather than averaged -- see module docstring
        # for why this only approximates v1's actual sum-of-raw-logits rule.
        return _softmax((logp * w[:, None] * len(w)).sum(axis=0))
    raise ValueError(f"unknown combine rule {rule!r} (want mean_prob/mean_logprob/sum_logit)")


class MultiRunPredictor:
    """One ``Predictor`` per ensemble member. Each member is internally
    homogeneous (H1/H3's checks apply within it); members may differ freely
    in feature config, subset, or training data from each other."""

    def __init__(self, spec: EnsembleSpec, device=None, filters: ReadFilters | None = None,
                 tta_condition_swap: bool = False,
                 target_prior: str = "none", nn_weight: float | None = None, tau: float = 1.0,
                 rule: str = "mean_prob"):
        self.filters = filters or ReadFilters()
        #: default combine rule, used when predict_record/predict_region's
        #: own `rule` argument is left as None -- lets this stand in for
        #: `Predictor` in score_benchmark.score(), which calls
        #: predict_region(region, bam1, bam2) with no rule argument.
        self.rule = rule
        self.predictors: list[Predictor] = []
        self.weights: list[float] = []
        classes = None
        for m in spec.members:
            pred = Predictor(_fold_paths(m.run_dir), device=device, filters=self.filters,
                              tta_condition_swap=tta_condition_swap)
            if target_prior != "none":
                pred.prior_correction = pred.build_prior_correction(
                    target_prior, nn_weight=nn_weight, tau=tau, train_labels=m.labels)
            if classes is None:
                classes = pred.classes
            elif pred.classes != classes:
                raise ValueError(f"{m.run_dir}: classes {pred.classes} vs {classes} -- "
                                  f"ensemble members must share the same class list")
            self.predictors.append(pred)
            self.weights.append(m.weight)
        self.classes = classes
        self.spec = spec

    def member_probs(self, record: dict) -> np.ndarray:
        """``[M, C]`` -- each member's own fold-averaged, prior-corrected
        posterior for one record."""
        out = np.zeros((len(self.predictors), len(self.classes)), dtype=np.float64)
        for i, pred in enumerate(self.predictors):
            per_fold, _ = pred.fold_probs(record)
            _, probs = pred.combine(per_fold)
            out[i] = probs
        return out

    def predict_record(self, record: dict, rule: str | None = None):
        probs = combine(self.member_probs(record), rule or self.rule, self.weights)
        return self.classes[int(np.argmax(probs))], probs

    def predict_region(self, region_str: str, bam1: str, bam2: str, rule: str | None = None,
                        need_weights: bool = False):
        """Same 4-tuple shape as ``Predictor.predict_region`` (attention is
        always None here) so ``score_benchmark.score``/``report`` work
        unchanged over either predictor type."""
        t0 = time.perf_counter()
        record = extract_event(parse_region(region_str), bam1, bam2, filters=self.filters)
        t1 = time.perf_counter()
        label, probs = self.predict_record(record, rule)
        t2 = time.perf_counter()
        return label, probs, None, {"extract_s": t1 - t0, "infer_s": t2 - t1, "total_s": t2 - t0}


# --- posterior cache: one extraction pass, many rule/weight/prior settings ---

def cache_member_fold_probs(mrp: MultiRunPredictor, csv_paths, verbose: bool = False, keep=None):
    """``[(csv, region, truth, [M] list of raw [F, C] per-fold posteriors)]``,
    uncorrected -- mirrors ``prior_sweep.cache_fold_probs`` one level up."""
    from dsignal.score_benchmark import CSV_LABEL

    cached = []
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r}")
        truth = CSV_LABEL[stem]
        for region, bam1, bam2 in read_prediction_csv(path):
            if keep is not None and not keep(stem, region):
                continue
            record = extract_event(parse_region(region), bam1, bam2, filters=mrp.filters)
            member_fold = [pred.fold_probs(record)[0] for pred in mrp.predictors]
            cached.append((stem, region, truth, member_fold))
            if verbose:
                print(f"  cached {region}", file=sys.stderr)
    return cached


def evaluate_cache(mrp: MultiRunPredictor, cached, rule: str = "mean_prob"):
    """``(predictions, [N, C] combined probs)`` from a cache built by
    ``cache_member_fold_probs`` -- applies each member's *current*
    ``prior_correction`` (so re-running with a different ``--target-prior``
    setting is still one analytical pass, no re-extraction) and the given
    combine rule."""
    preds, all_probs = [], []
    for _stem, _region, _truth, member_fold in cached:
        member_probs = np.zeros((len(mrp.predictors), len(mrp.classes)), dtype=np.float64)
        for i, (pred, per_fold) in enumerate(zip(mrp.predictors, member_fold)):
            _, probs = pred.combine(per_fold)
            member_probs[i] = probs
        combined = combine(member_probs, rule, mrp.weights)
        preds.append(mrp.classes[int(np.argmax(combined))])
        all_probs.append(combined)
    return preds, np.array(all_probs)


def save_cache(path: str, cached, classes) -> None:
    stems = np.array([c[0] for c in cached])
    regions = np.array([c[1] for c in cached])
    truths = np.array([c[2] for c in cached])
    n_members = len(cached[0][3])
    arrays = {}
    for m in range(n_members):
        # per-member fold count can differ across members (e.g. a run with a
        # different --folds), so each member's [N, F_m, C] array is stored
        # under its own key rather than stacked into one tensor.
        arrays[f"member{m}"] = np.stack([c[3][m] for c in cached])
    np.savez(path, stems=stems, regions=regions, truths=truths,
              classes=np.array(classes), n_members=n_members, **arrays)


def load_cache(path: str):
    data = np.load(path, allow_pickle=False)
    n_members = int(data["n_members"])
    cached = []
    for i in range(len(data["stems"])):
        member_fold = [data[f"member{m}"][i] for m in range(n_members)]
        cached.append((str(data["stems"][i]), str(data["regions"][i]), str(data["truths"][i]), member_fold))
    return cached, list(data["classes"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score an ensemble spec against the curated benchmark")
    s.add_argument("--spec", required=True, help="EnsembleSpec JSON (runs/ensembles/*.json)")
    s.add_argument("--csv", nargs="+", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--min-anchor", type=int, default=5)
    s.add_argument("--max-nh", type=int, default=10)
    s.add_argument("--min-mapq", type=int, default=0)
    s.add_argument("--tta-condition-swap", action="store_true")
    s.add_argument("--target-prior", choices=("none", "uniform", "benchmark", "nn-weight"), default="none")
    s.add_argument("--nn-weight", type=float, default=None)
    s.add_argument("--tau", type=float, default=1.0)
    s.add_argument("--rule", choices=("mean_prob", "mean_logprob", "sum_logit"), default="mean_prob")
    s.add_argument("--half", choices=("both", "dev", "test"), default="both")
    s.add_argument("--split", default="benchmark/split.tsv")
    s.add_argument("--cache-out", default=None, help="write per-member per-fold posteriors here")
    s.add_argument("--cache-in", default=None, help="skip extraction, read the cache instead")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    args.func(args)


def cmd_score(args) -> None:
    from dsignal.score_benchmark import DEFAULT_CSVS, report

    spec = EnsembleSpec.load(args.spec)
    mrp = MultiRunPredictor(
        spec, device=args.device,
        filters=ReadFilters(min_anchor=args.min_anchor, max_nh=args.max_nh, min_mapq=args.min_mapq),
        tta_condition_swap=args.tta_condition_swap,
        target_prior=args.target_prior, nn_weight=args.nn_weight, tau=args.tau,
    )
    print(f"ensemble {spec.name!r}: {len(mrp.predictors)} members, "
          f"weights {mrp.weights}, classes {mrp.classes}, rule {args.rule!r}", file=sys.stderr)

    keep = None
    split = None
    if args.half != "both":
        from dsignal.benchmark_split import load_split

        split = load_split(args.split)
        keep = lambda stem, region: split.get((stem, region)) == args.half  # noqa: E731

    csv_paths = args.csv or DEFAULT_CSVS
    if args.cache_in:
        cached, cached_classes = load_cache(args.cache_in)
        if list(cached_classes) != list(mrp.classes):
            raise ValueError(f"{args.cache_in}: classes {cached_classes} vs spec classes {mrp.classes}")
        if keep is not None:
            cached = [c for c in cached if keep(c[0], c[1])]
        print(f"loaded {len(cached)} cached events from {args.cache_in}", file=sys.stderr)
    else:
        cached = cache_member_fold_probs(mrp, csv_paths, args.verbose, keep=keep)
        if args.cache_out:
            save_cache(args.cache_out, cached, mrp.classes)
            print(f"wrote cache to {args.cache_out}", file=sys.stderr)

    preds, probs = evaluate_cache(mrp, cached, args.rule)
    rows = []
    from dsignal.extract import parse_region as _parse_region

    from dsignal.score_benchmark import collapse
    for (stem, region, truth, _member_fold), pred, p in zip(cached, preds, probs):
        _chrom, start, end = _parse_region(region)
        rows.append({
            "csv": stem, "region": region, "width": end - start, "truth": truth, "pred": pred,
            "correct": int(pred == truth), "correct_collapsed": int(collapse(pred) == collapse(truth)),
            **{f"p_{c}": p[i] for i, c in enumerate(mrp.classes)},
            # per-member timing isn't captured through the cache path, and the
            # cache is the point of this module -- report()'s wall-clock
            # section is not meaningful here, unlike score_benchmark's.
            "extract_s": 0.0, "infer_s": 0.0, "total_s": 0.0,
        })

    report(rows, mrp, args.half)

    if args.out:
        import csv as _csv

        for r in rows:
            r["truth_c"], r["pred_c"] = collapse(r["truth"]), collapse(r["pred"])
        with open(args.out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote per-event results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
