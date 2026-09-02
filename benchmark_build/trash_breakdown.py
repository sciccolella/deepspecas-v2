"""Clean-vs-TRASH accuracy breakdown for a set of already-scored per-event TSVs
(Stage 10, plans/signal-native-v2-bnew-dev-results.md).

Properly-scored replacement for Stage 9's `v2_pred`-derived 70.4%/24.6%
clean/TRASH figures (those came from `curation.tsv`'s `v2_pred` review-aid
column, not a prior-corrected `score_benchmark` run). This reads real
per-event predictions (`dsignal.score_benchmark --out ...` / `dsignal.v1_reference
--curation ... --out ...` TSVs) and joins them to `curation.tsv`'s `trash`
column on `(csv, region)` -- never region alone, per the branch's own
retraction of the 62/69 figure for exactly that mistake.

Usage:
    python -m benchmark_build.trash_breakdown \\
        --curation benchmark_new/curation.tsv \\
        --pred runs/bnew_dev_e4_union2.tsv:v2_e4_union2 \\
               runs/bnew_dev_v1_ensemble.tsv:v1_ensemble \\
        --out runs/bnew_dev_trash_breakdown.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from dsignal.score_benchmark import collapse

CSV_LABEL_INV = {"A3": "A3", "A5": "A5", "CSE": "ES", "RI": "IR", "NN": "NN"}


def load_trash(curation_path: str):
    """(csv_stem, region) -> "0"/"1"/"NA", from curation.tsv's truth+trash
    columns (dropping ambiguous rows)."""
    out = {}
    with open(curation_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["ambiguous"] == "1":
                continue
            stem = CSV_LABEL_INV[row["truth"]]
            out[(stem, row["region"])] = row["trash"]
    return out


def _acc(rows, collapsed):
    if not rows:
        return None
    f = collapse if collapsed else (lambda x: x)
    return float(np.mean([f(r["pred"]) == f(r["truth"]) for r in rows]))


def report(name, rows, trash):
    joined = [(r, trash.get((r["csv"], r["region"]))) for r in rows]
    n_nomatch = sum(1 for _, t in joined if t is None)
    clean = [r for r, t in joined if t == "0"]
    dirty = [r for r, t in joined if t == "1"]
    n_na = sum(1 for _, t in joined if t == "NA")
    print(f"\n=== {name} (n={len(rows)}, {n_nomatch} unmatched to curation.tsv, "
          f"{n_na} trash=NA dropped) ===")
    for label, sub in (("clean", clean), ("TRASH", dirty)):
        a5, ac = _acc(sub, False), _acc(sub, True)
        if a5 is None:
            print(f"  {label:5s} n=  0")
            continue
        print(f"  {label:5s} n={len(sub):3d}  5-class {a5:6.1%}  collapsed {ac:6.1%}")
    return {
        "name": name, "n_clean": len(clean), "n_trash": len(dirty),
        "acc_clean_5": _acc(clean, False), "acc_trash_5": _acc(dirty, False),
        "acc_clean_c": _acc(clean, True), "acc_trash_c": _acc(dirty, True),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curation", required=True)
    ap.add_argument("--pred", nargs="+", required=True,
                     help="path:name pairs of score_benchmark-format TSVs")
    ap.add_argument("--out", default=None, help="summary TSV")
    args = ap.parse_args(argv)

    trash = load_trash(args.curation)

    summaries = []
    for spec in args.pred:
        path, _, name = spec.partition(":")
        name = name or path
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        summaries.append(report(name, rows, trash))

    if args.out:
        fields = ["name", "n_clean", "n_trash", "acc_clean_5", "acc_trash_5",
                  "acc_clean_c", "acc_trash_c"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(summaries)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
