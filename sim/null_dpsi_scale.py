"""How large is a *null* dPSI in this simulation? (plan Stage 3.1)

``sim/select_events.py`` gives every ``nn_hard`` negative a real catalogued AS
event with ``psi1 == psi2`` **exactly**, while positives draw
``|dPSI| ~ U(0.3, 0.8)``. The model therefore learns a boundary somewhere near
0.15, and 81.6% of the curated benchmark's own matched events fall below 0.30 --
i.e. four out of five real positives land on the *negative* side of a boundary
the simulation taught. That is hypothesis 1 in
plans/signal-native-v2-docs-and-model.md, and the fix is to move positives and
negatives *together*: negatives should carry a small non-zero null effect, not a
mathematically exact zero.

The question this tool answers is where that null magnitude comes from.
**Not from the benchmark** -- `DPSI_BETA` is already fitted there and that is one
of the two open honesty caveats. It comes from the simulation's own replicate
structure: ``sim/abundance.py`` draws each of ``n_replicates`` per condition
from a negative binomial around the same targeted weights, so two conditions
holding *identical* targets still differ by pure sampling noise. That spread is
the technical floor below which "differential" is not a meaningful claim, and it
is exactly what a non-differential event should look like.

Measured per event from the count matrix (no alignment, no BAMs, no annotation
beyond the SUPPA catalogue the design already used):

    PSI_cond = sum(inclusion counts) / sum(inclusion + exclusion counts)

pooled over that condition's replicates, then ``|PSI_c1 - PSI_c2|`` over the
``nn_hard`` rows, whose two conditions share a target by construction.

Usage (on pasteur, where the batches live):
    python -m sim.null_dpsi_scale --batches sim/batches/b1 sim/batches/b2 \\
        --events-dir sim/events --design design.geomwin.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from os.path import join as pjoin

import numpy as np

from sim.catalogue import build_catalogue


def read_design(path: str, kinds=("nn_hard",)):
    with open(path) as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["kind"] in kinds]


def read_counts(path: str):
    """transcript_id -> {column: count}, plus the column names."""
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: empty count matrix")
    cols = [c for c in rows[0] if c != "transcript_id"]
    return {r["transcript_id"]: {c: float(r[c]) for c in cols} for r in rows}, cols


def psi_from_counts(counts, cols, incl, excl, cond_prefix: str):
    """Pooled PSI over one condition's replicates, or None if the event's
    transcripts carry no reads at all in that condition."""
    reps = [c for c in cols if c.startswith(cond_prefix)]
    i = sum(counts[t][c] for t in incl if t in counts for c in reps)
    e = sum(counts[t][c] for t in excl if t in counts for c in reps)
    return (i / (i + e)) if (i + e) > 0 else None


def measure(batch_dirs, events_dir: str, design_name: str, kinds):
    cat = {e.event_id: e for e in build_catalogue(events_dir)}
    print(f"catalogue: {len(cat):,} events", file=sys.stderr)

    out = []
    for d in batch_dirs:
        design = read_design(pjoin(d, design_name), kinds=kinds)
        counts, cols = read_counts(pjoin(d, "count_matrix.tsv"))
        missing = 0
        for row in design:
            ev = cat.get(row["event_id"])
            if ev is None:
                missing += 1
                continue
            p1 = psi_from_counts(counts, cols, ev.inclusion_transcripts,
                                 ev.exclusion_transcripts, "cond1")
            p2 = psi_from_counts(counts, cols, ev.inclusion_transcripts,
                                 ev.exclusion_transcripts, "cond2")
            if p1 is None or p2 is None:
                continue
            out.append(dict(batch=d, event_id=row["event_id"], kind=row["kind"],
                            label=row["label"], psi_c1=p1, psi_c2=p2,
                            abs_dpsi=abs(p1 - p2)))
        print(f"  {d}: {len(design)} rows, {missing} not in catalogue", file=sys.stderr)
    return out


def report(rows) -> None:
    d = np.array([r["abs_dpsi"] for r in rows])
    if not len(d):
        raise SystemExit("no measurable rows")
    qs = [50, 75, 90, 95, 99]
    print(f"\n=== null |dPSI| from replicate noise, n = {len(d):,} ===")
    print(f"  mean   {d.mean():.4f}     sd {d.std():.4f}")
    for q in qs:
        print(f"  p{q:<3d}   {np.percentile(d, q):.4f}")
    print(f"  max    {d.max():.4f}")
    print("\nRead this as the floor: a simulated 'non-differential' event already")
    print("moves this much from sampling alone. A negative drawn at |dPSI| well")
    print("inside this range is indistinguishable from one pinned at exactly 0,")
    print("so the injected null should sit *at* this scale, not below it.")

    # per class, since RI carries far fewer reads on its minor branch
    print("\nby class:")
    for lab in sorted({r["label"] for r in rows}):
        sub = np.array([r["abs_dpsi"] for r in rows if r["label"] == lab])
        print(f"  {lab:4s} n={len(sub):4d}  median {np.median(sub):.4f}  p90 {np.percentile(sub, 90):.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--events-dir", default="sim/events")
    ap.add_argument("--design", default="design.geomwin.tsv",
                    help="design file name inside each batch dir")
    ap.add_argument("--kinds", nargs="+", default=["nn_hard"],
                    help="design kinds to measure; nn_hard is the one whose two "
                    "conditions share a target exactly")
    ap.add_argument("--out", default=None, help="per-event TSV")
    args = ap.parse_args(argv)

    rows = measure(args.batches, args.events_dir, args.design, set(args.kinds))
    report(rows)
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
