"""v1 prediction CSVs -> score_benchmark-format per-event TSVs (Stage 0.1,
plans/signal-native-v2-ensemble.md).

Turns "beat the v1 ensemble" into one `benchmark_split compare` invocation,
the same test every other comparison on this branch already uses. v1's own
prediction files (`ix,region,pred[,A,CSE,NN,RI]`) already carry the `--a
--cse` collapse baked in -- v1 never distinguishes A3 from A5, so `pred` here
is literally `"A"` for every such event, never `"A3"`/`"A5"`. That makes the
raw 5-class `correct` column meaningless for v1 (it is 0 for every true A3/A5
event by construction) -- only `correct_collapsed` is the comparable number,
which is what every v1 figure in CLAUDE.md/plans quotes. Probabilities are
left blank: `benchmark_split.compare` only reads `csv`/`region`/`truth`/`pred`,
and v1_pretrained's probabilities are already collapsed to 4 classes so they
would not be honest under the 5-class column headers.

Usage:
    python -m dsignal.v1_reference \\
        --glob "runs/v1/pred_*.csv" --out runs/benchmark_events_v1_ensemble.tsv
    python -m dsignal.v1_reference \\
        --glob "runs/v1_pretrained/pred/*.csv" --out runs/benchmark_events_v1_squishstack.tsv
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from os.path import basename, splitext

from dsignal.dataset import CLASSES
from dsignal.extract import parse_region
from dsignal.score_benchmark import CSV_LABEL, collapse

FIELDS = (
    ["csv", "region", "width", "truth", "pred", "correct", "correct_collapsed"]
    + [f"p_{c}" for c in CLASSES]
    + ["extract_s", "infer_s", "total_s", "truth_c", "pred_c"]
)


def read_v1_pred_csv(path: str):
    """[(region, pred)] -- ``pred`` already carries v1's ``--a --cse`` collapse
    (A/CSE/NN/RI), read from either the bare (``ix,region,pred``) or the
    probability-carrying (``ix,region,pred,A,CSE,NN,RI``) variant."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append((row["region"], row["pred"]))
    return out


def build_rows(csv_paths):
    rows = []
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem.startswith("pred_"):
            stem = stem[len("pred_"):]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r} (known: {sorted(CSV_LABEL)})")
        truth = CSV_LABEL[stem]
        for region, pred in read_v1_pred_csv(path):
            chrom, start, end = parse_region(region)
            row = {
                "csv": stem,
                "region": region,
                "width": end - start,
                "truth": truth,
                "pred": pred,
                "correct": int(pred == truth),
                "correct_collapsed": int(collapse(pred) == collapse(truth)),
                **{f"p_{c}": "" for c in CLASSES},
                "extract_s": "",
                "infer_s": "",
                "total_s": "",
                "truth_c": collapse(truth),
                "pred_c": collapse(pred),
            }
            rows.append(row)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True, help="glob of v1 pred_*.csv files, one per truth class")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matched {args.glob!r}")
    rows = build_rows(paths)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_c = sum(r["correct_collapsed"] for r in rows)
    print(f"wrote {args.out}: {n} events, {n_c}/{n} = {n_c / n:.1%} collapsed (unsplit)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
