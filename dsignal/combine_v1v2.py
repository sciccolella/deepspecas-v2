"""Stage 21.2 -- v1+v2 combination: mean of posteriors.

**Pre-registered rule (locked in before any scoring)**: simple average of v1's
and v2's probabilities in the collapsed 4-class space (A, CSE, NN, RI),
argmax. No free parameter, nothing to tune or second-guess after seeing
results. (Considered and rejected: an RI-rescue fallback keyed on v2's
RI-vs-A margin against a fixed tau -- carries a free parameter that would
need fixing a priori, a weaker guarantee than having none. Not pursued --
see plans/signal-native-v2-stage21-inference-levers.md.)

**Why the collapsed space.** The shipped v1 checkpoints
(``data/trained_nets/<mode>/trained.pth``) were trained with ``--a --cse``
collapsing (README.md's example ``dsEnsemble.py`` invocation), so v1's
released nets only ever produce 4-class posteriors (A, CSE, NN, RI) -- they
cannot separate A3/A5 or CE/SE. v2's own 5-class posteriors are collapsed to
match (``p_A = p_A3 + p_A5``) rather than retraining v1.

v1's per-event posteriors come from ``dsEnsemble.py --out`` (Stage 21.2 step
1, one TSV per truth CSV, e.g. one for each of ``A3.csv``/``A5.csv``/...).
v2's come from ``dsignal.score_benchmark --out`` (one TSV covering every
truth CSV already). Rows are joined on ``(csv, region)`` -- never region
alone, since the benchmark holds at least one region under two truth labels.

v2-only remains the preferred headline result regardless of outcome, per the
standing user decision recorded in
``plans/signal-native-v2-stage18-22-accuracy-plan.md``'s Context section --
this combination is reported alongside it, not proposed as a replacement.

Usage::

    python -m dsignal.combine_v1v2 \\
        --v1 runs/v1_bnew_A3.tsv runs/v1_bnew_A5.tsv runs/v1_bnew_ES.tsv \\
             runs/v1_bnew_IR.tsv runs/v1_bnew_NN.tsv \\
        --v2 runs/bnew_dev_e4_union2.tsv \\
        --out runs/combined_bnew_dev.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys

from dsignal.score_benchmark import collapse

#: the space this combination lives in -- v1's ceiling, not v2's native one.
CLASSES = ("A", "CSE", "NN", "RI")


def load_v1(paths) -> dict:
    """``(csv, region) -> {"A":.., "CSE":.., "NN":.., "RI":.., "truth":..}``,
    from one or more ``dsEnsemble.py --out`` TSVs (Stage 21.2 step 1)."""
    out = {}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (r["csv"], r["region"])
                if key in out:
                    raise ValueError(f"{path}: duplicate (csv, region) {key} across --v1 inputs")
                out[key] = {c: float(r[f"p_{c}"]) for c in CLASSES}
                out[key]["truth"] = r["truth"]
    return out


def load_v2(path: str) -> dict:
    """``(csv, region) -> {...}`` from a ``dsignal.score_benchmark --out`` TSV,
    collapsing v2's native 5-class posteriors (``p_A3``, ``p_A5``, ``p_CSE``,
    ``p_NN``, ``p_RI``) into the same 4-class space v1 lives in."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["csv"], r["region"])
            out[key] = {
                "A": float(r["p_A3"]) + float(r["p_A5"]),
                "CSE": float(r["p_CSE"]),
                "NN": float(r["p_NN"]),
                "RI": float(r["p_RI"]),
                "truth": r["truth"],
            }
    return out


def combine(p1: dict, p2: dict) -> dict:
    """Mean of posteriors, per class -- the entire rule."""
    return {c: (p1[c] + p2[c]) / 2 for c in CLASSES}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", nargs="+", required=True,
                    help="dsEnsemble.py --out TSV(s), one per truth CSV")
    ap.add_argument("--v2", required=True,
                    help="dsignal.score_benchmark --out TSV (5-class posteriors)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    v1 = load_v1(args.v1)
    v2 = load_v2(args.v2)

    shared = sorted(set(v1) & set(v2))
    only_v1, only_v2 = sorted(set(v1) - set(v2)), sorted(set(v2) - set(v1))
    if only_v1 or only_v2:
        print(f"warning: {len(only_v1)} event(s) only in --v1, {len(only_v2)} only in --v2 -- "
              f"combining the {len(shared)} events present in both", file=sys.stderr)
    if not shared:
        raise ValueError("no (csv, region) keys shared between --v1 and --v2 -- wrong inputs?")

    rows = []
    for key in shared:
        csv_stem, region = key
        t1, t2 = v1[key]["truth"], v2[key]["truth"]
        if t1 != t2:
            raise ValueError(f"{key}: truth mismatch, v1 says {t1!r}, v2 says {t2!r}")
        probs = combine(v1[key], v2[key])
        pred = max(probs, key=probs.get)
        rows.append(
            {
                "csv": csv_stem,
                "region": region,
                "truth": t1,
                **{f"p_{c}": probs[c] for c in CLASSES},
                "pred": pred,
                "correct": int(pred == t1),
                "correct_collapsed": int(pred == collapse(t1)),
            }
        )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} combined events to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
