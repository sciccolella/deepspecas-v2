"""Turn a `plot2s.py` render of the v2 simulation into v1 train/val CSVs.

Plan Stage 2, the representation crossover: v1's images + ResNet50 trained on
*v2's* simulated data and scored on the same 140-event benchmark. It is a
diagnostic, not an improvement -- it separates "v2's simulation is wrong" from
"signal-native is a fragile representation". If v1's representation transfers
well from this very same simulated data, the simulation is not the problem.

Two things this has to get right:

* **Split by event, not by image.** Every event renders once per crossfeed
  setting (4 images at the default `CROSSFEEDS`), and those four are near
  duplicates. Splitting rows at random puts an event's own images on both sides
  and the validation number becomes meaningless. Here all images of an event
  stay together, which -- since ``sim/select_events.py`` gives one gene at most
  one event -- also makes the split gene-disjoint, matching how
  ``dsignal.train`` groups its folds.
* **Keep the label mapping identical to v2's.** The render CSV carries the
  label in its own column, so A3/A5/CSE/NN/RI come straight from
  ``labels.geomwin.tsv``; no re-derivation, nothing to drift.

Usage:
    python -m sim.build_v1_trainset --render-csv runs/v1_crossover/all.csv \\
        --out-dir runs/v1_crossover --val-frac 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from os.path import join as pjoin

import numpy as np

#: `<chrom>_<start>-<end>.<bam1>.<bam2>.cf_<a>-<b>.png` -> the region part,
#: which identifies the event (one event per gene, so this groups by gene too)
_EVENT_RE = re.compile(r"^([^.]+)\.")


def event_of(image_name: str) -> str:
    m = _EVENT_RE.match(image_name)
    if not m:
        raise ValueError(f"cannot parse an event out of {image_name!r}")
    return m.group(1)


def read_render_csv(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, label = line.rsplit(",", 1)
            rows.append((name, label))
    return rows


def split_by_event(rows, val_frac: float, seed: int):
    by_event = defaultdict(list)
    for name, label in rows:
        by_event[event_of(name)].append((name, label))

    events = sorted(by_event)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(events))
    n_val = int(round(len(events) * val_frac))
    val_events = {events[i] for i in order[:n_val]}

    train, val = [], []
    for ev, items in by_event.items():
        (val if ev in val_events else train).extend(items)
    return train, val, len(events), len(val_events)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-csv", required=True, help="plot2s --outcsv output (image,label)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = read_render_csv(args.render_csv)
    train, val, n_events, n_val_events = split_by_event(rows, args.val_frac, args.seed)

    # the gate: no event may appear on both sides
    tr_ev = {event_of(n) for n, _ in train}
    va_ev = {event_of(n) for n, _ in val}
    overlap = tr_ev & va_ev
    assert not overlap, f"{len(overlap)} events span train and val, e.g. {sorted(overlap)[:3]}"

    for name, items in (("train", train), ("val", val)):
        path = pjoin(args.out_dir, f"{name}.csv")
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(items)
        counts = Counter(l for _, l in items)
        print(f"{name}: {len(items):,} images over {len({event_of(n) for n, _ in items}):,} events   "
              + "  ".join(f"{c}={counts[c]}" for c in sorted(counts)))

    print(f"\n{len(rows):,} images, {n_events:,} events, {n_val_events:,} held out for val "
          f"({args.val_frac:.0%}), split by event so an event's crossfeeds never straddle it")


if __name__ == "__main__":
    main()
