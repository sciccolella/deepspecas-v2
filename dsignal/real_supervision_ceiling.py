"""Stage 11 Stage 1 (plans/signal-native-v2-trash-sim.md): a dev-only
real-supervision upper bound for NN recall on ``benchmark_new``.

Before spending days re-simulating a TRASH noise regime, this bounds how much
of the NN gap (Stage 10: 9-14% recall, p_NN AUC 0.639) is closable at all --
by training and evaluating with 5-fold grouped cross-validation *within* the
curated dev half itself, rather than zero-shot from simulation. It answers
"is NN intrinsically hard here" before "is the simulator missing the regime".

**This is a ceiling, not a model.** It is trained on the evaluation set's own
dev half. It must never be scored on test, never ensembled into a deployed
predictor, and never quoted as a deliverable -- only as a diagnostic bound
next to the Stage 10 zero-shot numbers.

Grouping is by ``benchmark_new/split.tsv``'s existing ``cluster`` column (50
kb single-linkage clusters -- see ``dsignal/benchmark_split.py``'s
docstring), not by gene: ``curation.tsv`` does not carry a gene column for
every row. This reuses ``dsignal/train.py`` unchanged by smuggling the
cluster id through ``event_id``'s ``gene_id_of`` convention (split on the
first ``";"``, ``sim/gene_split.py``) -- ``event_id`` here is
``f"{cluster};{region}"``, so ``train.py``'s "group by gene" grouping is
actually "group by cluster" with zero code changes there beyond the
``--dump-oof`` hook this stage added.

Usage::

    # 1. build the dev npz set + labels.tsv (local; benchmark_new's BAMs are
    #    checked into the repo, no pasteur needed)
    python -m dsignal.real_supervision_ceiling build \\
        --out-dir runs/real_supervision_ceiling

    # 2. train (5-fold, grouped by cluster) -- same base recipe as the
    #    zero-shot model, run separately so its stdout/stderr and any GPU
    #    failure are visible on their own
    python -m dsignal.train \\
        --npz-dir runs/real_supervision_ceiling/npz \\
        --labels runs/real_supervision_ceiling/labels.tsv \\
        --augment all --feature-subset no-geometry --refit --inner-val-frac 0.30 \\
        --dump-oof runs/real_supervision_ceiling/oof.tsv \\
        --out-dir runs/real_supervision_ceiling/folds \\
        --json-out runs/real_supervision_ceiling/train.json

    # 3. report under both truth conventions (Stage 0) + NN recall
    python -m dsignal.real_supervision_ceiling report \\
        --oof runs/real_supervision_ceiling/oof.tsv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from os.path import join as pjoin

from dsignal.baseline_gbm import report, report_collapsed
from dsignal.score_benchmark import CSV_LABEL, apply_trash_as_nn, collapse

#: truth label -> the benchmark_new CSV stem it lives in (inverse of CSV_LABEL).
_INV_CSV_LABEL = {v: k for k, v in CSV_LABEL.items()}

DEFAULT_SPLIT = "benchmark_new/split.tsv"
DEFAULT_CURATION = "benchmark_new/curation.tsv"
DEFAULT_BAM1 = "benchmark_new/c1.bam"
DEFAULT_BAM2 = "benchmark_new/c2.bam"


def _dev_rows(split_path: str):
    """[(csv_stem, region, truth, cluster)] for the dev half of split_path,
    sorted for a deterministic build order."""
    with open(split_path, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["half"] == "dev"]
    rows.sort(key=lambda r: (r["csv"], r["region"]))
    return [(r["csv"], r["region"], r["truth"], r["cluster"]) for r in rows]


def cmd_build(args) -> None:
    from os import makedirs

    rows = _dev_rows(args.split)
    print(f"{len(rows)} dev events from {args.split}", file=sys.stderr)

    makedirs(args.out_dir, exist_ok=True)
    extract_csv = pjoin(args.out_dir, "extract.csv")
    labels_path = pjoin(args.out_dir, "labels.tsv")
    npz_dir = pjoin(args.out_dir, "npz")
    makedirs(npz_dir, exist_ok=True)

    with open(extract_csv, "w", newline="") as fcsv, open(labels_path, "w", newline="") as flab:
        cw = csv.writer(fcsv)
        lw = csv.writer(flab, delimiter="\t")
        lw.writerow(["ix", "event_id", "label", "kind"])
        for ix, (stem, region, truth, cluster) in enumerate(rows):
            cw.writerow([region, args.bam1, args.bam2])
            lw.writerow([f"{ix:05d}", f"{cluster};{region}", truth, "real_dev"])

    print(f"wrote {extract_csv} and {labels_path}", file=sys.stderr)

    cmd = [
        sys.executable, "-m", "dsignal.extract",
        "--csv", extract_csv, "--out-dir", npz_dir,
        "--min-anchor", str(args.min_anchor), "--max-nh", str(args.max_nh),
        "--min-mapq", str(args.min_mapq),
    ]
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)

    print(
        f"\ndone. Next: python -m dsignal.train --npz-dir {npz_dir} --labels {labels_path} "
        f"--augment all --feature-subset no-geometry --refit --inner-val-frac 0.30 "
        f"--dump-oof {pjoin(args.out_dir, 'oof.tsv')} --out-dir {pjoin(args.out_dir, 'folds')} "
        f"--json-out {pjoin(args.out_dir, 'train.json')}",
        file=sys.stderr,
    )


def _oof_to_rows(oof_path: str):
    """oof.tsv (ix, event_id, truth, pred) -> score_benchmark-format rows
    (csv, region, truth, pred, correct, correct_collapsed), region and the
    benchmark_new CSV stem recovered from event_id ('cluster;region')."""
    rows = []
    with open(oof_path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            _cluster, region = r["event_id"].split(";", 1)
            truth, pred = r["truth"], r["pred"]
            rows.append({
                "csv": _INV_CSV_LABEL[truth],
                "region": region,
                "truth": truth,
                "pred": pred,
                "correct": int(pred == truth),
                "correct_collapsed": int(collapse(pred) == collapse(truth)),
            })
    return rows


def _nn_recall(rows) -> tuple[int, int]:
    nn_true = [r for r in rows if r["truth"] == "NN"]
    hit = sum(1 for r in nn_true if r["pred"] == "NN")
    return hit, len(nn_true)


def cmd_report(args) -> None:
    rows = _oof_to_rows(args.oof)
    print(f"=== real-supervision ceiling: {len(rows)} dev events, 5-fold grouped CV ===")

    print("\n--- convention A (curated truth, unchanged) ---")
    truth_a = [r["truth"] for r in rows]
    pred_a = [r["pred"] for r in rows]
    labels = sorted(CSV_LABEL.values())
    report(truth_a, pred_a, labels, "real-supervision ceiling")
    report_collapsed(truth_a, pred_a, "real-supervision ceiling, A3+A5 collapsed to A")
    hit, n = _nn_recall(rows)
    print(f"\nNN recall (A): {hit}/{n} = {hit / n:.1%}" if n else "\nNN recall (A): n=0")

    print("\n--- convention B (TRASH-as-NN, trash==NA dropped) ---")
    rows_b = apply_trash_as_nn(rows, args.curation)
    truth_b = [r["truth"] for r in rows_b]
    pred_b = [r["pred"] for r in rows_b]
    report(truth_b, pred_b, labels, "real-supervision ceiling, TRASH-as-NN")
    report_collapsed(truth_b, pred_b, "real-supervision ceiling, TRASH-as-NN, A3+A5 collapsed to A")
    hit_b, n_b = _nn_recall(rows_b)
    print(f"\nNN recall (B): {hit_b}/{n_b} = {hit_b / n_b:.1%}" if n_b else "\nNN recall (B): n=0")

    if args.out:
        fields = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote per-event (convention A) results to {args.out}", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="extract benchmark_new dev to npz + write labels.tsv")
    b.add_argument("--split", default=DEFAULT_SPLIT)
    b.add_argument("--bam1", default=DEFAULT_BAM1)
    b.add_argument("--bam2", default=DEFAULT_BAM2)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--min-anchor", type=int, default=5)
    b.add_argument("--max-nh", type=int, default=10)
    b.add_argument("--min-mapq", type=int, default=0)
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("report", help="score a train.py --dump-oof TSV under both conventions")
    r.add_argument("--oof", required=True)
    r.add_argument("--curation", default=DEFAULT_CURATION)
    r.add_argument("--out", default=None, help="write convention-A per-event TSV here")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
