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

``--curation`` mode (Stage 10, plans/signal-native-v2-bnew-dev-results.md) is
for ``benchmark_new/``, where ``--glob`` cannot be used correctly: v1 was run
over the *candidate* class (``runs/v1_uncurated_new/pred_*.csv``'s stem), but
curation moved rows between classes (e.g. some A3-truth rows started as IR
candidates), so the candidate-CSV stem is no longer the truth-CSV stem. This
mode instead reads truth and ``v1_pred`` straight from
``benchmark_new/curation.tsv`` -- which already carries both, from the same
Stage 8 consensus review that produced the frozen benchmark -- and reconstructs
each row's *truth*-side stem (``A3``/``A5``/``ES``/``IR``/``NN``) via the
inverse of ``score_benchmark.CSV_LABEL``, matching how ``benchmark_new/*.csv``
were built. Ambiguous curation rows (``ambiguous == "1"``, never scored
anywhere) are dropped. Rows are keyed by ``(stem, region)``, never region
alone -- keying by region alone is what produced the retracted 62/69 figure
(see CLAUDE.md).

Usage:
    python -m dsignal.v1_reference --curation benchmark_new/curation.tsv \\
        --out runs/bnew_dev_v1_ensemble.tsv --split benchmark_new/split.tsv --half dev
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


def build_rows_from_curation(curation_path: str, keep=None):
    """``benchmark_new/curation.tsv`` -> score_benchmark-format rows, keyed by
    the *truth*-side stem (inverse of ``score_benchmark.CSV_LABEL``), not the
    candidate-class stem the row happened to be extracted under. Drops
    ``ambiguous == "1"`` rows (curated but never scored anywhere). ``keep``,
    if given, is ``(stem, region) -> bool`` (e.g. a split filter)."""
    inv_label = {v: k for k, v in CSV_LABEL.items()}  # CSE->ES, else identity
    rows = []
    with open(curation_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["ambiguous"] == "1":
                continue
            truth = row["truth"]
            if truth not in inv_label:
                raise ValueError(f"{curation_path}: unrecognized truth {truth!r} in {row!r}")
            stem = inv_label[truth]
            region = row["region"]
            if keep is not None and not keep(stem, region):
                continue
            pred = row["v1_pred"]
            chrom, start, end = parse_region(region)
            rows.append({
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
            })
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--glob", help="glob of v1 pred_*.csv files, one per truth class")
    src.add_argument("--curation", help="benchmark_new/curation.tsv-style curation ledger")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None, help="split TSV (benchmark_split format); "
                     "with --curation only, restricts to --half")
    ap.add_argument("--half", choices=("both", "dev", "test"), default="both")
    args = ap.parse_args(argv)

    if args.glob:
        paths = sorted(glob.glob(args.glob))
        if not paths:
            raise SystemExit(f"no files matched {args.glob!r}")
        rows = build_rows(paths)
    else:
        keep = None
        if args.half != "both":
            if not args.split:
                ap.error("--half needs --split")
            from dsignal.benchmark_split import load_split
            split = load_split(args.split)
            half = args.half
            keep = lambda stem, region: split.get((stem, region)) == half  # noqa: E731
        rows = build_rows_from_curation(args.curation, keep=keep)

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
