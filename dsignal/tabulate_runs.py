"""Tabulate several `dsignal/train.py --json-out` runs side by side.

Used for the augmentation ablation (Phase 3 step 4 / Verification item 5):
no-aug control first, then each augmentation alone, then all.

Usage:
    python -m dsignal.tabulate_runs runs/*.json
"""

from __future__ import annotations

import argparse
import json
from os.path import basename, splitext

CLASSES = ("A3", "A5", "CSE", "NN", "RI")

#: Standing reference row, so the table never has to be read against memory.
#: Provenance: sim/baseline_model.py's 24 crude features on npz_scale1260 --
#: 1,260 events, window sampler v1, 43% NN. It is the fixed yardstick the whole
#: layer-by-layer history was measured against, NOT a like-for-like control for
#: anything trained on npz_geomwin (4,800 events, v3 windows, 25% NN), which is
#: a different data distribution with a different majority-class floor.
REFERENCE_ROWS = [
    ("RF hand-features [npz_scale1260, v1 windows]", 0.488, 0.325,
     {"A3": 0.108, "A5": 0.160, "CSE": 0.132, "NN": 0.657, "RI": 0.568}),
]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    if not args.no_reference:
        rows.extend(REFERENCE_ROWS)

    for path in args.json:
        with open(path) as f:
            d = json.load(f)
        name = splitext(basename(path))[0]
        r = d["result"]
        rows.append((name, r["accuracy"], r["macro_f1"], r["per_class_f1"]))

    width = max(len(r[0]) for r in rows) + 2
    header = f"{'run':{width}s} {'acc':>6s} {'macroF1':>8s} " + " ".join(f"{c:>6s}" for c in CLASSES)
    print(header)
    print("-" * len(header))
    for name, acc, macro, per_class in rows:
        cells = " ".join(f"{per_class.get(c, float('nan')):6.3f}" for c in CLASSES)
        print(f"{name:{width}s} {acc:6.3f} {macro:8.3f} {cells}")

    print("\nA3/A5 F1 is the metric that matters -- that is the class pair the")
    print("topology features exist to fix (v1's standing failure mode).")


if __name__ == "__main__":
    main()
