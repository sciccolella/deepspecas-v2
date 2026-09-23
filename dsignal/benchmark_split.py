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
    # Stage 19 Gate B "pooled": mean p_* over seeds, argmax, same TSV format
    python -m dsignal.benchmark_split pool runs/bnew_dev_X_s{0,1,2}.tsv --out pooled.tsv
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
from dsignal.score_benchmark import CSV_LABEL, DEFAULT_CSVS, apply_trash_as_nn, collapse

DEFAULT_SPLIT = "benchmark/split.tsv"


def _load_trash(path):
    """``region -> "1"/"0"/"NA"``, from a Stage 9 ``curation.tsv``."""
    with open(path, newline="") as f:
        return {r["region"]: r["trash"] for r in csv.DictReader(f, delimiter="\t")}


def _events(csv_paths, trash=None):
    """[(csv_stem, region, chrom, start, end, trash)] over the benchmark CSVs.

    ``trash`` is ``None`` unless a ``--trash-tsv`` was given, in which case each
    event's value is ``"1"``/``"0"``/``"NA"`` -- ``"NA"`` is its own stratum, not
    folded into ``"0"``.
    """
    out = []
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r}")
        for region, _bam1, _bam2 in read_prediction_csv(path):
            chrom, rest = region.split(":")
            start, end = rest.split("-")
            t = trash.get(region, "NA") if trash is not None else None
            out.append((stem, region, chrom, int(start), int(end), t))
    return out


def _stratum(event):
    """The stratification key for one event: truth class, plus trash value when
    tracked (``event[5]`` is ``None`` when no ``--trash-tsv`` was given)."""
    stem, trash = event[0], event[5]
    cls = CSV_LABEL[stem]
    return cls if trash is None else (cls, trash)


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
    """Cluster -> "dev"/"test", stratified by truth class (and, when the events
    carry a trash value, by ``(class, trash)`` -- see ``_stratum``).

    Largest cluster first; each goes to the half further below its quota for the
    cluster's own stratum mix, ties broken by the seeded RNG. Quotas are half of
    each stratum count, so the two halves come out balanced with per-stratum
    counts differing by at most one.
    """
    rng = random.Random(seed)
    counts = Counter(_stratum(e) for e in events)
    quota = {"dev": {c: n / 2 for c, n in counts.items()},
             "test": {c: n / 2 for c, n in counts.items()}}
    have = {"dev": Counter(), "test": Counter()}
    order = sorted(clusters, key=lambda c: (-len(c), events[c[0]][2], events[c[0]][3]))

    out = {}
    for cl in order:
        mix = Counter(_stratum(events[i]) for i in cl)
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
        w.writerow(["csv", "region", "truth", "trash", "cluster", "half"])
        for i, (stem, region, chrom, start, end, trash) in enumerate(events):
            w.writerow([stem, region, CSV_LABEL[stem], "" if trash is None else trash,
                        cluster_of[i], halves[i]])


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
    trash = _load_trash(args.trash_tsv) if args.trash_tsv else None
    events = _events(args.csv, trash=trash)
    clusters = cluster(events, args.link)
    halves, have = assign(events, clusters, args.seed)
    write_split(args.out, events, clusters, halves)

    sizes = Counter(len(c) for c in clusters)
    print(f"{len(events)} events, {len(clusters)} clusters at link={args.link} bp "
          f"(sizes {dict(sorted(sizes.items()))})")
    strata = sorted({_stratum(e) for e in events})
    for half in ("dev", "test"):
        n = sum(1 for v in halves.values() if v == half)
        per = "  ".join(f"{s}={have[half][s]}" for s in strata)
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
    if args.trash_as_nn:
        base = apply_trash_as_nn(base, args.trash_as_nn)
        new = apply_trash_as_nn(new, args.trash_as_nn)
        print(f"convention B (TRASH-as-NN via {args.trash_as_nn})", file=sys.stderr)
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

    if args.by_class:
        reweight = _class_mix(args.reweight_mix) if args.reweight_mix else None
        _by_class(base_by, new_by, shared, reweight=reweight)


POOL_CLASSES = ("A3", "A5", "CSE", "NN", "RI")


def pool_rows(tables: list) -> list:
    """Seed-pooled rows: mean ``p_*`` over several scoring TSVs of the same
    events, then argmax -- a len(tables)-member mean-posterior ensemble (Stage
    19 Gate B's "pooled"). Every other column comes from the first table, with
    ``pred``/``correct``/``correct_collapsed``/``pred_c`` recomputed. The tables
    must cover the same ``(csv, region)`` keys with the same truth."""
    key = lambda r: (r["csv"], r["region"])
    first = tables[0]
    by = [{key(r): r for r in t} for t in tables]
    keys = [key(r) for r in first]
    for i, b in enumerate(by[1:], 1):
        if set(b) != set(keys):
            raise SystemExit(f"table {i} covers different events ({len(b)} vs {len(keys)})")
    out = []
    for r0 in first:
        k = key(r0)
        rows = [b[k] for b in by]
        if any(r["truth"] != r0["truth"] for r in rows):
            raise SystemExit(f"{k}: truth differs across tables")
        p = {c: sum(float(r[f"p_{c}"]) for r in rows) / len(rows) for c in POOL_CLASSES}
        pred = max(POOL_CLASSES, key=lambda c: (p[c], -POOL_CLASSES.index(c)))
        row = dict(r0)
        row.update({f"p_{c}": repr(p[c]) for c in POOL_CLASSES})
        row["pred"] = pred
        row["correct"] = str(int(pred == r0["truth"]))
        row["correct_collapsed"] = str(int(collapse(pred) == collapse(r0["truth"])))
        if "pred_c" in row:
            row["pred_c"] = collapse(pred)
        out.append(row)
    return out


def cmd_pool(args) -> None:
    tables = [_read_events_tsv(p) for p in args.tsv]
    rows = pool_rows(tables)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tables[0][0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    acc = sum(int(r["correct"]) for r in rows)
    accc = sum(int(r["correct_collapsed"]) for r in rows)
    print(f"pooled {len(tables)} tables over {n} events -> {args.out}: "
          f"5-class {acc}/{n} = {acc / n:.1%}, collapsed {accc}/{n} = {accc / n:.1%}")


def _class_mix(csv_paths):
    """truth-class -> proportion, from row counts of class-labelled CSVs (e.g.
    benchmark_new/{A3,A5,ES,IR,NN}.csv) via score_benchmark.CSV_LABEL."""
    counts = Counter()
    for path in csv_paths:
        stem = splitext(basename(path))[0]
        if stem not in CSV_LABEL:
            raise ValueError(f"{path}: no label mapping for {stem!r}")
        counts[CSV_LABEL[stem]] += sum(1 for _ in read_prediction_csv(path))
    total = sum(counts.values())
    return {c: n / total for c, n in counts.items()}


def _by_class(base_by, new_by, shared, reweight=None):
    """Per-truth-class recall (using collapsed correctness, since v1 never
    predicts A3/A5 -- see CLAUDE.md/the arena-b3 pre-registration) and a
    paired McNemar per class -- 5 tests, over the 5 truth strata A3/A5/CSE/NN/RI,
    not to be cherry-picked individually. Macro-recall is the unweighted mean
    over classes present, so it is not a function of any one class's sample
    size (Trap 3 in plans/signal-native-v2-arena-b3.md). ``reweight``, if
    given, is a ``truth-class -> proportion`` mapping (e.g. benchmark_new's
    class mix) used to print a sensitivity-reweighted accuracy alongside."""
    classes = sorted({base_by[k]["truth"] for k in shared})
    print(f"\nper-class recall (collapsed correctness, {len(classes)} truth strata):")
    b_recalls, n_recalls = {}, {}
    for cls in classes:
        ks = [k for k in shared if base_by[k]["truth"] == cls]
        truth = [collapse(base_by[k]["truth"]) for k in ks]
        b_pred = [collapse(base_by[k]["pred"]) for k in ks]
        n_pred = [collapse(new_by[k]["pred"]) for k in ks]
        b_acc = sum(p == t for p, t in zip(b_pred, truth))
        n_acc = sum(p == t for p, t in zip(n_pred, truth))
        b_recalls[cls] = b_acc / len(ks)
        n_recalls[cls] = n_acc / len(ks)
        b, c, p = mcnemar(b_pred, n_pred, truth)
        print(f"  {cls:4s} n={len(ks):4d}  base {b_acc:4d}/{len(ks):<4d} = {b_recalls[cls]:5.1%}   "
              f"new {n_acc:4d}/{len(ks):<4d} = {n_recalls[cls]:5.1%}   "
              f"base-only {b:2d}  new-only {c:2d}  McNemar p = {p:.4f}")

    b_macro = sum(b_recalls.values()) / len(classes)
    n_macro = sum(n_recalls.values()) / len(classes)
    print(f"  macro-recall (unweighted over {len(classes)} classes):  base {b_macro:5.1%}   new {n_macro:5.1%}")

    if reweight is not None:
        missing = [c for c in classes if c not in reweight]
        if missing:
            print(f"  warning: reweight mix has no entry for {missing}, skipping those classes",
                  file=sys.stderr)
        used = [c for c in classes if c in reweight]
        norm = sum(reweight[c] for c in used)
        b_rw = sum(b_recalls[c] * reweight[c] for c in used) / norm
        n_rw = sum(n_recalls[c] * reweight[c] for c in used) / norm
        print(f"  reweighted to given class mix ({', '.join(f'{c}={reweight[c]:.1%}' for c in used)}):"
              f"  base {b_rw:5.1%}   new {n_rw:5.1%}")


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
    m.add_argument("--trash-tsv", default=None,
                   help="Stage 9 curation.tsv -- stratify by (truth class, trash) "
                   "instead of truth class alone, so dev/test carry comparable "
                   "TRASH fractions per class. 'NA' trash values form their own stratum.")
    m.set_defaults(func=cmd_make)

    c = sub.add_parser("compare", help="paired McNemar between two scored per-event TSVs")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--half", choices=("dev", "test", "both"), default="dev")
    c.add_argument("--split", default=DEFAULT_SPLIT)
    c.add_argument("--trash-as-nn", default=None, metavar="curation.tsv",
                   help="Convention B (Stage 11 Stage 0): re-truth both sides via "
                   "dsignal.score_benchmark.apply_trash_as_nn before pairing, so "
                   "McNemar runs under TRASH-as-NN instead of the default convention A")
    c.add_argument("--by-class", action="store_true",
                   help="also print per-truth-class recall (collapsed correctness) "
                   "and a paired McNemar per class, plus the unweighted macro-recall")
    c.add_argument("--reweight-mix", nargs="+", default=None, metavar="CSV",
                   help="with --by-class: class-mix CSVs (e.g. benchmark_new/*.csv) whose "
                   "row counts give a truth-class distribution to reweight the arena's "
                   "per-class recalls by, as a sensitivity check (Trap 3)")
    c.set_defaults(func=cmd_compare)

    p = sub.add_parser("pool", help="mean-posterior pool of several scored per-event TSVs "
                       "(same events), written in the same format so `compare` takes it")
    p.add_argument("tsv", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pool)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
