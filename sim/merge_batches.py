"""Merge simulation batches into one training set.

Each batch is generated independently (its own ``select_events.py`` seed and its
own disjoint gene set), so both its ``labels.tsv`` row indices and its
``{ix:05d}.*.npz`` filenames restart at zero and would collide on merge. This
script re-keys a batch as ``<tag>_<ix>``, which ``dsignal.dataset.read_labels``
already handles unchanged: it splits an npz filename on the first ``.`` to get
the key, and matches it against the labels TSV's ``ix`` column.

Records are **hardlinked** where possible, so the combined directory costs
nothing on top of the per-batch ones.

Usage:
    python sim/merge_batches.py --batch b1=sim/experiment/batches/b1 \
        --batch b2=sim/experiment/batches/b2 \
        --out-npz sim/experiment/npz_scaleup \
        --out-labels sim/experiment/labels_scaleup.tsv

Each batch directory must contain ``npz/`` and ``labels.tsv``.

``--nn-frac`` additionally writes a labels file with the NN rows downsampled to
a target class fraction. Nothing else changes -- the same records, the same
folds -- so training on it is a clean design-vs-calibration ablation (step 4 of
plans/signal-native-v2-scaleup-7500.md). Downsampling is deterministic given
``--seed`` and drops whole *genes*' NN rows in shuffled order, never splitting a
gene across the kept/dropped boundary, so the gene-level CV grouping is
unaffected.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from os.path import basename, join as pjoin

import numpy as np

#: `weight` (Stage 20.2, per-row loss weight) is carried when a batch has it
#: and written empty otherwise -- dsignal.dataset.read_labels reads "" as 1.0.
FIELDS = ["ix", "event_id", "label", "kind", "weight"]


def _link(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.unlink(dst)
    try:
        os.link(src, dst)
    except OSError:  # different filesystem
        shutil.copy2(src, dst)


def merge(batches, out_npz: str, out_labels: str):
    os.makedirs(out_npz, exist_ok=True)
    rows = []
    for tag, path in batches:
        npz_dir = pjoin(path, "npz")
        with open(pjoin(path, "labels.tsv")) as fh:
            batch_rows = list(csv.DictReader(fh, delimiter="\t"))
        index = {}
        for name in os.listdir(npz_dir):
            if name.endswith(".npz"):
                index[name.split(".", 1)[0]] = name
        n = 0
        for row in batch_rows:
            name = index.get(row["ix"])
            if name is None:  # extraction dropped it; labels stay authoritative
                continue
            new_ix = f"{tag}_{row['ix']}"
            _link(pjoin(npz_dir, name), pjoin(out_npz, f"{new_ix}.{name.split('.', 1)[1]}"))
            rows.append({**{k: row.get(k, "") for k in FIELDS}, "ix": new_ix})
            n += 1
        print(f"  {tag}: {n}/{len(batch_rows)} records from {npz_dir}")

    with open(out_labels, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    counts = Counter(r["label"] for r in rows)
    total = len(rows)
    print(f"wrote {out_labels}: {total} records")
    for label, k in sorted(counts.items()):
        print(f"  {label}: {k} ({k / total:.1%})")
    return rows


def _gene_of(event_id: str) -> str:
    return event_id.split(";", 1)[0]


def _drop_genes_to(rows, keep_n: int, seed: int):
    """Keep roughly ``keep_n`` of ``rows``, dropping whole genes at a time so the
    gene-level CV grouping is never split."""
    by_gene = defaultdict(list)
    for r in rows:
        by_gene[_gene_of(r["event_id"])].append(r)
    genes = sorted(by_gene)
    np.random.default_rng(seed).shuffle(genes)
    kept, n = [], 0
    for g in genes:
        if n >= keep_n:
            break
        kept.extend(by_gene[g])
        n += len(by_gene[g])
    return kept


def _write_labels_subset(rows, out, out_path: str) -> None:
    """Write ``out`` (a subset of ``rows``) to ``out_path``, preserving ``rows``'
    original order, and print per-label counts."""
    order = {r["ix"]: i for i, r in enumerate(rows)}
    out = sorted(out, key=lambda r: order[r["ix"]])
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(out)
    counts = Counter(r["label"] for r in out)
    print(f"wrote {out_path}: {len(out)} records")
    for label, k in sorted(counts.items()):
        print(f"  {label}: {k} ({k / len(out):.1%})")


def resample_to_nn_frac(rows, target_frac: float, seed: int, out_path: str):
    """Write a labels file whose NN class is ``target_frac`` of the set.

    Below the set's natural NN fraction this drops NN rows and every positive
    is kept -- the clean version of the ablation. **Above** it, there are not
    enough NN rows, so positives have to be dropped instead; that also shrinks
    the training set, which is a confound and is printed as such rather than
    hidden.
    """
    nn = [r for r in rows if r["label"] == "NN"]
    rest = [r for r in rows if r["label"] != "NN"]
    natural = len(nn) / len(rows)
    if target_frac <= natural:
        keep_n = int(round(target_frac * len(rest) / (1.0 - target_frac)))
        kept_nn = _drop_genes_to(nn, keep_n, seed)
        kept_rest = rest
    else:
        keep_n = int(round(len(nn) * (1.0 - target_frac) / target_frac))
        kept_nn = nn
        kept_rest = _drop_genes_to(rest, keep_n, seed)
        print(f"  NOTE: target {target_frac:.0%} exceeds the set's natural "
              f"{natural:.0%}; {len(rest) - len(kept_rest)} positive rows dropped "
              f"to reach it, so this condition also has less data")

    _write_labels_subset(rows, kept_rest + kept_nn, out_path)


def subsample_events(rows, keep_n: int, seed: int, out_path: str) -> None:
    """Write a labels file with ~``keep_n`` records, dropping whole genes (via
    ``_drop_genes_to``) so nested subsamples at different ``keep_n`` share one
    gene-disjoint pool and ``GroupKFold`` grouping is never split."""
    kept = _drop_genes_to(rows, keep_n, seed)
    _write_labels_subset(rows, kept, out_path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", metavar="TAG=DIR",
                    help="repeatable; DIR must contain npz/ and labels.tsv")
    ap.add_argument("--from-labels", default=None,
                    help="skip merging and resample this existing labels file instead. "
                    "Used to derive a *positive-matched* NN condition: resampling the "
                    "43%%-NN file down to 25%% keeps its exact positive subset, so the "
                    "pair differs only in NN count and not in how much positive data "
                    "each was trained on")
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--nn-frac", type=float, action="append", default=None,
                    help="also write a labels file whose NN class is this fraction; repeatable")
    ap.add_argument("--nn-frac-out", default=None)
    ap.add_argument("--subsample-events", type=int, action="append", default=None,
                    help="also write a labels file with ~this many records, dropping whole "
                    "genes; repeatable. Nested across repeats (same --seed => the smaller "
                    "sets are gene-set prefixes of the larger ones) as long as the input row "
                    "set doesn't change between them. Output path is "
                    "<out-labels stem>.n<N>.tsv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.from_labels:
        with open(args.from_labels) as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        print(f"read {len(rows)} records from {args.from_labels}")
    else:
        if not args.batch or not args.out_npz:
            raise SystemExit("need --batch ... --out-npz, or --from-labels")
        batches = []
        for spec in args.batch:
            tag, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"--batch expects TAG=DIR, got {spec!r}")
            batches.append((tag, path))
        rows = merge(batches, args.out_npz, args.out_labels)

    for frac in args.nn_frac or []:
        out = args.nn_frac_out or args.out_labels.replace(".tsv", f".nn{int(frac * 100)}.tsv")
        resample_to_nn_frac(rows, frac, args.seed, out)

    for n in args.subsample_events or []:
        out = args.out_labels.replace(".tsv", f".n{n}.tsv")
        subsample_events(rows, n, args.seed, out)


if __name__ == "__main__":
    main()
