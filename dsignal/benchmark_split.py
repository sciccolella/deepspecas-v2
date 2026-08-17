"""Stratified dev/test split of the 140-event curated benchmark, and the paired
McNemar test between two scored runs restricted to one half.

Why this exists (plans/signal-native-v2-docs-and-model.md, Stage 0). Two things
have been tuned by reading benchmark accuracy on all 140 events: the NN prior
strength, and ``DPSI_BETA``/``PSI_LEVEL_BETA`` in ``sim/select_events.py``
(fitted by moments to the benchmark's own matched events). Both are recorded as
open honesty caveats in plans/signal-native-v2-resume-2.md. Splitting the
benchmark once, into a committed file that cannot drift, turns those into
dev-set choices: everything tuned is read off **dev**, and **test** is scored
once per milestone and reported separately.

Stated cost, up front: McNemar on 70 events detects a swing of roughly 8-10
events where 140 detected 5-6. Several past interventions (the whole
depth-invariance line, +2 of 35) would have been unmeasurable at that power.
The alternative is that no future number from this benchmark is clean.

**Grouping.** The split is cluster-disjoint, not provably gene-disjoint: the
benchmark CSVs carry regions only, and no usable annotation is checked into this
repo, so "same gene" is approximated by single-linkage clustering of the regions
at ``--link 50000`` bp on the same chromosome. That threshold is not arbitrary
-- the sorted inter-event gaps jump from 5,571 bp to 67,203 bp, so any threshold
in that decade produces the same 4 multi-event clusters (of sizes 2) and 132
singletons. Two events that share a gene but sit >50 kb apart would be missed;
with 140 events at 23 chromosomes that is a small residual risk, and it is
recorded rather than hidden.

One quirk the clustering exposes: region ``14:24115727-24116971`` appears in
**both** ``A3.csv`` and ``A5.csv``, i.e. the benchmark holds one region under two
truth labels. Rows are therefore keyed by ``(csv, region)``, never by region
alone.

**Stratification.** Strata are the four truth classes. Clusters are assigned
largest-first, each to whichever half is further below its per-class quota; ties
go to the half holding fewer events so far, and only then to a seeded RNG. That
second criterion is what lands the two odd class counts (CSE 65, RI 17) in
opposite halves and gives exactly 70/70 -- dev A3=14 A5=15 CSE=33 RI=8, test
A3=14 A5=15 CSE=32 RI=9.

Usage::

    python -m dsignal.benchmark_split make --out benchmark/split.tsv
    python -m dsignal.benchmark_split compare \\
        runs/benchmark_events_geomwin.tsv runs/benchmark_events_<new>.tsv --half dev
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from os.path import basename, splitext

from dsignal.predict import read_prediction_csv
from dsignal.prior_sweep import mcnemar
from dsignal.score_benchmark import CSV_LABEL, DEFAULT_CSVS, collapse

DEFAULT_SPLIT = "benchmark/split.tsv"


def _events(csv_paths):
    """[(csv_stem, region, chrom, start, end)] over the benchmark CSVs."""
    out = []
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r}")
        for region, _bam1, _bam2 in read_prediction_csv(path):
            chrom, rest = region.split(":")
            start, end = rest.split("-")
            out.append((stem, region, chrom, int(start), int(end)))
    return out


def cluster(events, link: int):
    """Single-linkage cluster indices: same chromosome and gap <= ``link`` bp.

    A stand-in for gene identity; see the module docstring for what it can miss.
    """
    parent = list(range(len(events)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_chrom = defaultdict(list)
    for i, ev in enumerate(events):
        by_chrom[ev[2]].append(i)
    for idxs in by_chrom.values():
        idxs.sort(key=lambda i: events[i][3])
        for a, b in zip(idxs, idxs[1:]):
            if events[b][3] - events[a][4] <= link:
                union(a, b)

    groups = defaultdict(list)
    for i in range(len(events)):
        groups[find(i)].append(i)
    return [sorted(v) for v in groups.values()]


def assign(events, clusters, seed: int):
    """Cluster -> "dev"/"test", stratified by truth class.

    Largest cluster first; each goes to the half further below its quota for the
    cluster's own class mix, ties broken by the seeded RNG. Quotas are half of
    each class count, so the two halves come out at 70/70 with per-class counts
    differing by at most one.
    """
    rng = random.Random(seed)
    counts = Counter(CSV_LABEL[e[0]] for e in events)
    quota = {"dev": {c: n / 2 for c, n in counts.items()},
             "test": {c: n / 2 for c, n in counts.items()}}
    have = {"dev": Counter(), "test": Counter()}
    order = sorted(clusters, key=lambda c: (-len(c), events[c[0]][2], events[c[0]][3]))

    out = {}
    for cl in order:
        mix = Counter(CSV_LABEL[events[i][0]] for i in cl)
        # deficit = how far below quota this half would still be after taking it
        def deficit(half):
            return sum(quota[half][c] - have[half][c] for c in mix.elements())
        d_dev, d_test = deficit("dev"), deficit("test")
        t_dev, t_test = sum(have["dev"].values()), sum(have["test"].values())
        if (d_dev, -t_dev) > (d_test, -t_test):
            half = "dev"
        elif (d_test, -t_test) > (d_dev, -t_dev):
            half = "test"
        else:
            half = rng.choice(("dev", "test"))
        have[half] += mix
        for i in cl:
            out[i] = half
    return out, have


def write_split(path, events, clusters, halves) -> None:
    cluster_of = {}
    for k, cl in enumerate(sorted(clusters, key=lambda c: (events[c[0]][2], events[c[0]][3]))):
        for i in cl:
            cluster_of[i] = k
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["csv", "region", "truth", "cluster", "half"])
        for i, (stem, region, chrom, start, end) in enumerate(events):
            w.writerow([stem, region, CSV_LABEL[stem], cluster_of[i], halves[i]])


def load_split(path=DEFAULT_SPLIT):
    """``(csv_stem, region) -> "dev"/"test"``."""
    with open(path, newline="") as f:
        return {(r["csv"], r["region"]): r["half"] for r in csv.DictReader(f, delimiter="\t")}


def filter_rows(rows, half, split=None):
    """Rows of a scored per-event table restricted to one half ("both" = all)."""
    if half == "both":
        return list(rows)
    if split is None:
        split = load_split()
    return [r for r in rows if split.get((r["csv"], r["region"])) == half]


def _read_events_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def cmd_make(args) -> None:
    events = _events(args.csv)
    clusters = cluster(events, args.link)
    halves, have = assign(events, clusters, args.seed)
    write_split(args.out, events, clusters, halves)

    sizes = Counter(len(c) for c in clusters)
    print(f"{len(events)} events, {len(clusters)} clusters at link={args.link} bp "
          f"(sizes {dict(sorted(sizes.items()))})")
    for half in ("dev", "test"):
        n = sum(1 for v in halves.values() if v == half)
        per = "  ".join(f"{c}={have[half][c]}" for c in ("A3", "A5", "CSE", "RI"))
        print(f"  {half:4s} n={n:3d}   {per}")
    # a cluster must not straddle the split
    by_cluster = defaultdict(set)
    for i, cl in enumerate(clusters):
        for j in cl:
            by_cluster[i].add(halves[j])
    assert all(len(v) == 1 for v in by_cluster.values()), "cluster straddles the split"
    print(f"wrote {args.out}", file=sys.stderr)


def cmd_compare(args) -> None:
    split = load_split(args.split)
    base = filter_rows(_read_events_tsv(args.base), args.half, split)
    new = filter_rows(_read_events_tsv(args.new), args.half, split)
    key = lambda r: (r["csv"], r["region"])
    base_by, new_by = {key(r): r for r in base}, {key(r): r for r in new}
    shared = [k for k in base_by if k in new_by]
    if len(shared) != len(base) or len(shared) != len(new):
        print(f"warning: {len(base)} base / {len(new)} new rows, {len(shared)} paired",
              file=sys.stderr)

    print(f"half={args.half}  n={len(shared)}   {args.base} -> {args.new}")
    for name, collapsed in (("5-class", False), ("collapsed", True)):
        f = collapse if collapsed else (lambda x: x)
        truth = [f(base_by[k]["truth"]) for k in shared]
        b_pred = [f(base_by[k]["pred"]) for k in shared]
        n_pred = [f(new_by[k]["pred"]) for k in shared]
        b_acc = sum(p == t for p, t in zip(b_pred, truth))
        n_acc = sum(p == t for p, t in zip(n_pred, truth))
        b, c, p = mcnemar(b_pred, n_pred, truth)
        print(f"  {name:9s} base {b_acc:3d}/{len(shared)} = {b_acc / len(shared):5.1%}   "
              f"new {n_acc:3d}/{len(shared)} = {n_acc / len(shared):5.1%}   "
              f"base-only {b:2d}  new-only {c:2d}  McNemar p = {p:.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="build the committed dev/test split")
    m.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    m.add_argument("--out", default=DEFAULT_SPLIT)
    m.add_argument("--link", type=int, default=50000,
                   help="single-linkage gap for the same-locus proxy (default: 50000)")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_make)

    c = sub.add_parser("compare", help="paired McNemar between two scored per-event TSVs")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--half", choices=("dev", "test", "both"), default="dev")
    c.add_argument("--split", default=DEFAULT_SPLIT)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
