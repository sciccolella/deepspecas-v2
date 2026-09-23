"""Array-for-array identity of two extractor npz dirs (Stage 19 Gate A0).

Gate A0: P0 -- a batch regenerated with the tail off, indels off and ``@RG``
on -- must reproduce the old npz exactly: ``cov_*``, ``delcov_*``,
``junctions_*`` and the ``stats_*`` counts, record for record, plus a
byte-identical ``labels.tsv``. A mismatch is a pipeline bug to fix, not a
Gate A FAIL.

Records are paired by file name (``<ix>.<region>.npz``, so the window must
match too). ``--prefix-new`` / ``--prefix-old`` keep only that side's files carrying a
merge prefix such as ``b12_`` and strip it, so a batch can be checked against
its slice of a merged pool.

Usage::

    python -m sim.npz_diff OLD_DIR NEW_DIR [--labels OLD.tsv NEW.tsv]
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys

import numpy as np

ARRAY_KEYS = ("chrom", "start", "end", "strand", "cov_1", "cov_2", "delcov_1", "delcov_2",
              "junctions_1", "junctions_2")
STATS_KEYS = ("stats_1_json", "stats_2_json")


def _names(d: str, prefix: str) -> dict:
    out = {}
    for n in os.listdir(d):
        if n.endswith(".npz") and n.startswith(prefix):
            out[n[len(prefix):]] = os.path.join(d, n)
    return out


def diff_record(a_path: str, b_path: str) -> list:
    bad = []
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        for k in ARRAY_KEYS:
            if k not in a.files or k not in b.files:
                bad.append(f"{k}: missing")
            elif a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k]):
                bad.append(k)
        for k in STATS_KEYS:
            if json.loads(str(a[k])) != json.loads(str(b[k])):
                bad.append(k)
    return bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--prefix-old", default="")
    ap.add_argument("--prefix-new", default="")
    ap.add_argument("--labels", nargs=2, metavar=("OLD", "NEW"), default=None)
    ap.add_argument("--show", type=int, default=10, help="mismatching records to list")
    args = ap.parse_args(argv)

    old, new = _names(args.old, args.prefix_old), _names(args.new, args.prefix_new)
    only_old, only_new = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    shared = sorted(set(old) & set(new))
    mismatched, by_key = [], {}
    for n in shared:
        bad = diff_record(old[n], new[n])
        if bad:
            mismatched.append((n, bad))
            for k in bad:
                by_key[k] = by_key.get(k, 0) + 1
    ok = not (only_old or only_new or mismatched)
    print(f"{len(shared)} paired records; {len(only_old)} only in old, {len(only_new)} only in new; "
          f"{len(mismatched)} differ {by_key if by_key else ''}")
    for n, bad in mismatched[: args.show]:
        print(f"  {n}: {', '.join(bad)}")
    for n in (only_old[:3] + only_new[:3]):
        print(f"  unpaired: {n}")
    if args.labels:
        same = filecmp.cmp(args.labels[0], args.labels[1], shallow=False)
        print(f"labels {args.labels[0]} vs {args.labels[1]}: {'byte-identical' if same else 'DIFFER'}")
        ok = ok and same
    print("A0 identity: PASS" if ok else "A0 identity: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
