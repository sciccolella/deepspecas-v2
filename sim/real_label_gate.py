"""Stage 12 / A1: the pre-registered gate for annotation-derived real labels.

Runs `sim.real_label`'s labeller over windows whose truth was established by
manual curation -- `benchmark_new`'s 485 unambiguous rows and `benchmark/`'s
140 -- and asks whether annotation plus a BAM-derived presence filter can
recover what a curator decided by eye. If it can, real supervision needs no
curation and can be scaled to ~10^4 events; if it cannot, Track A stops.

Pass criteria, declared in the plan before this was ever run:

  1. non-NN agreement >= 0.80   (reference ceiling: the rMATS x SUPPA2
     consensus class is 93.3% correct conditional on the event being real)
  2. NN recall >= 0.50 on clean (`trash == 0`) windows   (for scale: zero-shot
     v2 gets 0.09-0.14, and the consensus pipeline gets 0 by construction)

Decisions are read off the **dev** half; test is reported alongside but is not
what the criteria are evaluated on, matching how every other tuned quantity on
this branch is handled.

Usage:
    python -m sim.real_label_gate --which bnew
    python -m sim.real_label_gate --which old
    python -m sim.real_label_gate --which bnew --sweep
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from dsignal.extract import extract_event, parse_region
from sim.real_label import (
    AMBIGUOUS,
    MIN_JUNC_READS,
    MIN_WINDOW_COV,
    NN,
    RI_MIN_INTRON_FRAC,
    CatalogueIndex,
    calibrate_offset,
    label_window,
)

CSV_LABEL = {"A3": "A3", "A5": "A5", "ES": "CSE", "IR": "RI", "NN": "NN"}
CLASSES = ("A3", "A5", "CSE", "NN", "RI")


def load_bnew(curation: str, split: Optional[str]) -> List[dict]:
    halves = {}
    if split:
        with open(split) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                halves[(row["csv"], row["region"])] = row["half"]
    # split.tsv is keyed by the *curated* stem (curation moved rows between
    # classes), not by `csv_source`, which is the pre-curation candidate class.
    # Keying on csv_source silently matches only the 304 rows where the two
    # agree -- and drops every NN row, since no NN candidate class exists.
    truth_stem = {"CSE": "ES", "RI": "IR"}
    rows = []
    with open(curation) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["ambiguous"] != "0":
                continue
            stem = truth_stem.get(row["truth"], row["truth"])
            rows.append(
                {
                    "region": row["region"],
                    "truth": row["truth"],
                    "trash": row["trash"],
                    "half": halves.get((stem, row["region"]), "?"),
                    "bam1": "benchmark_new/c1.bam",
                    "bam2": "benchmark_new/c2.bam",
                }
            )
    return rows


def load_old(csv_dir: str) -> List[dict]:
    rows = []
    for stem in ("A3", "A5", "ES", "IR"):
        with open(f"{csv_dir}/{stem}.csv") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                region, bam1, bam2 = line.split(",")[:3]
                rows.append(
                    {
                        "region": region,
                        "truth": CSV_LABEL[stem],
                        "trash": "NA",
                        "half": "?",
                        "bam1": bam1,
                        "bam2": bam2,
                    }
                )
    return rows


def _confusion(pairs, labels) -> Dict[str, Counter]:
    conf = {t: Counter() for t in labels}
    for truth, pred in pairs:
        conf.setdefault(truth, Counter())[pred] += 1
    return conf


def _print_confusion(conf, title: str) -> None:
    preds = sorted({p for c in conf.values() for p in c})
    print(f"\n{title}")
    print("  truth\\pred  " + "".join(f"{p:>7}" for p in preds) + f"{'n':>7}{'acc':>8}")
    for t in sorted(conf):
        row = conf[t]
        n = sum(row.values())
        acc = row.get(t, 0) / n if n else 0.0
        print(f"  {t:<11}" + "".join(f"{row.get(p, 0):>7}" for p in preds) + f"{n:>7}{acc:>8.3f}")


def report(rows: List[dict], results: List, title: str) -> dict:
    pairs = [(r["truth"], res.label) for r, res in zip(rows, results)]
    conf = _confusion(pairs, CLASSES)
    _print_confusion(conf, title)

    non_nn = [(t, p) for t, p in pairs if t != NN]
    agree = sum(1 for t, p in non_nn if t == p)
    nn_rows = [(r, res) for r, res in zip(rows, results) if r["truth"] == NN]
    nn_clean = [(r, res) for r, res in nn_rows if r["trash"] == "0"]
    nn_rec = sum(1 for _, res in nn_rows if res.label == NN) / max(1, len(nn_rows))
    nn_rec_clean = sum(1 for _, res in nn_clean if res.label == NN) / max(1, len(nn_clean))
    abstain = sum(1 for _, p in pairs if p == AMBIGUOUS)

    print(f"  non-NN agreement : {agree}/{len(non_nn)} = {agree / max(1, len(non_nn)):.3f}"
          f"   [criterion 1: >= 0.80]")
    if nn_rows:
        print(f"  NN recall (all)  : {nn_rec:.3f}  (n={len(nn_rows)})")
        print(f"  NN recall (clean): {nn_rec_clean:.3f}  (n={len(nn_clean)})"
              f"   [criterion 2: >= 0.50]")
    print(f"  abstained (AMBIG): {abstain}/{len(pairs)}")

    if any(r["trash"] in ("0", "1") for r in rows):
        for flag, name in (("0", "clean"), ("1", "TRASH")):
            sub = [(r["truth"], res.label) for r, res in zip(rows, results) if r["trash"] == flag]
            if not sub:
                continue
            hit = sum(1 for t, p in sub if t == p)
            print(f"  {name:<6} overall  : {hit}/{len(sub)} = {hit / len(sub):.3f}")
    return {
        "non_nn_agree": agree / max(1, len(non_nn)),
        "nn_recall_clean": nn_rec_clean,
        "n_non_nn": len(non_nn),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=("bnew", "old"), default="bnew")
    ap.add_argument("--curation", default="benchmark_new/curation.tsv")
    ap.add_argument("--split", default="benchmark_new/split.tsv")
    ap.add_argument("--old-dir", default="benchmark")
    ap.add_argument("--events-dir", default="sim/events")
    ap.add_argument("--min-reads", type=int, default=MIN_JUNC_READS)
    ap.add_argument("--ri-min-frac", type=float, default=RI_MIN_INTRON_FRAC)
    ap.add_argument("--min-window-cov", type=float, default=MIN_WINDOW_COV)
    ap.add_argument("--sweep", action="store_true", help="threshold sensitivity, diagnostic only")
    ap.add_argument("--out", help="per-window TSV")
    args = ap.parse_args(argv)

    rows = load_bnew(args.curation, args.split) if args.which == "bnew" else load_old(args.old_dir)
    print(f"loaded {len(rows)} curated windows ({args.which})", file=sys.stderr)

    index = CatalogueIndex.from_dir(args.events_dir)
    print("catalogue indexed", file=sys.stderr)

    records = []
    for i, r in enumerate(rows):
        records.append(extract_event(parse_region(r["region"]), r["bam1"], r["bam2"]))
        if (i + 1) % 100 == 0:
            print(f"  extracted {i + 1}/{len(rows)}", file=sys.stderr)

    offset, scores = calibrate_offset(records[:200], index)
    print(f"\ncoordinate offset calibration (annotation -> observed): {offset}")
    for off, hits in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"   offset {off}: {hits} matched catalogued junctions")

    def run(min_reads, ri_frac, min_cov):
        return [
            label_window(rec, index, offset=offset, min_reads=min_reads,
                         ri_min_frac=ri_frac, min_window_cov=min_cov)
            for rec in records
        ]

    results = run(args.min_reads, args.ri_min_frac, args.min_window_cov)
    print(f"\n=== defaults: min_reads={args.min_reads} ri_min_frac={args.ri_min_frac} "
          f"min_window_cov={args.min_window_cov} ===")
    report(rows, results, "all curated windows")
    by_half = defaultdict(list)
    for r, res in zip(rows, results):
        by_half[r["half"]].append((r, res))
    for half in ("dev", "test"):
        if by_half.get(half):
            hr, hres = zip(*by_half[half])
            report(list(hr), list(hres), f"half = {half}  [criteria are read off dev]")

    if args.sweep:
        print("\n=== threshold sensitivity (diagnostic, not the criterion) ===")
        print(f"  {'min_reads':>10}{'ri_frac':>9}{'non-NN agree':>14}{'NN rec clean':>14}")
        for mr in (2, 5, 10, 20):
            for rf in (0.05, 0.10, 0.20):
                res = run(mr, rf, args.min_window_cov)
                sub = [(r, x) for r, x in zip(rows, res) if r["half"] in ("dev", "?")]
                sr, sx = zip(*sub)
                pairs = [(r["truth"], x.label) for r, x in zip(sr, sx)]
                non_nn = [(t, p) for t, p in pairs if t != NN]
                nn_clean = [(r, x) for r, x in zip(sr, sx) if r["truth"] == NN and r["trash"] == "0"]
                a = sum(1 for t, p in non_nn if t == p) / max(1, len(non_nn))
                n = sum(1 for _, x in nn_clean if x.label == NN) / max(1, len(nn_clean))
                print(f"  {mr:>10}{rf:>9}{a:>14.3f}{n:>14.3f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["region", "truth", "pred", "resolved", "half", "trash",
                        "n_candidates", "n_present", "mean_cov"])
            for r, res in zip(rows, results):
                w.writerow([r["region"], r["truth"], res.label, res.resolved, r["half"],
                            r["trash"], res.n_candidates, res.n_present, f"{res.mean_cov:.2f}"])
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
