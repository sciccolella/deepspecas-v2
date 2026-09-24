"""Per-window junction-tail statistics, real or simulated (Stage 19.1).

Stage 18.1 found the single largest real-vs-sim tell is the *sparse junction
tail*: real windows carry ~9 junctions with <= 3 reads each, simulation ~0.
This module is the committed version of the read-only analysis that motivated
Stage 19 (plans/signal-native-v2-stage19-junction-tail.md), and the measuring
stick for Gate A1. It reads extractor npz only -- annotation-free, so the
identical code path runs on real and simulated records.

Definitions (fixed; the Gate A1 ranges were computed with exactly these):

- A window's junction set is the union over both conditions, keyed by
  ``(start, end)`` (1-based intron bounds, jI convention). ``total`` is the
  read count summed over both conditions.
- **tail junction**: ``total <= TAIL_MAX`` (3).
- **tail/kb**: per window, ``n_tail / (width / 1000)``; reported as the
  *median over windows*.
- **tail fraction**: *pooled*, ``sum(n_tail) / sum(n_junctions)``.
- **one-condition fraction**: per window, the fraction of *all* its junctions
  seen in exactly one condition; reported as the *mean over windows* (windows
  with no junctions are skipped).
- **location** of a tail junction, against the window's 1-based (start, end]:
  ``in`` (both intron bounds inside), ``out1`` (exactly one), ``out2``
  (neither: the window sits inside the intron -- ``bam.fetch`` returns a read
  whose N-op merely spans the window). Inside fraction is pooled over tail
  junctions.
- **canonical**: STAR motif > 0 (``jM % 20`` in 1..6); pooled over tail
  junctions.
- **shared**: the tail junction's start or end equals the start/end of an
  *expressed* (non-tail) junction of the same window.

Usage::

    python -m sim.tail_stats \\
        --set aqr=sim/experiment/real_pos_aqr/npz+sim/experiment/real_nn_aqr/npz \\
        --set b12=sim/experiment/npz_union2@sim/experiment/labels_union2.tsv:b12_ \\
        --gate-a1 b12 --json-out runs/stage19/tail_stats.json

A ``--set`` value is ``NAME=DIR[@LABELS][:IXPREFIX][+DIR...]``. ``LABELS``
defaults to ``DIR/labels.tsv`` or ``DIR/../labels.tsv`` and is only needed for
``--by-class``; ``IXPREFIX`` keeps only records whose ``ix`` starts with it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from os.path import join as pjoin

import numpy as np

TAIL_MAX = 3
MAX_N_DEFAULT = 2000

# Gate A1, copied verbatim from the pre-registration (commit 0e11fe204). Never
# edit: see the plan doc's 19.0 Results for why the last two differ from the
# fit cohorts' own pos+nn range.
GATE_A1 = {
    "tail_per_kb_median": (1.16, 3.12),
    "tail_frac": (0.53, 0.69),
    "one_cond_frac": (0.46, 0.65),
    "inside_frac": (0.13, 0.25),
    "canonical_frac": (0.58, 0.70),
}

# Gate A1 as AMENDED 2026-09-24, before any pasteur spend (user decision; plan
# doc "Amendment 1"): [min, max] across aqr/prpf4/u2af2 pos+nn, importance-
# weighted to b12's window-width distribution (10 log-width quantile bins) --
# the fit cohorts the gate text names, compared at matched width. Frozen as
# numbers so the pilot cannot move them. GATE_A1 above is kept for the record.
GATE_A1_AMENDED = {
    "tail_per_kb_median": (1.778, 4.775),
    "tail_frac": (0.531, 0.698),
    "one_cond_frac": (0.456, 0.652),
    "inside_frac": (0.207, 0.303),
    "canonical_frac": (0.638, 0.823),
}


def window_junctions(record: dict) -> dict:
    """(start, end) -> [count_1, count_2, motif, min_anchor, strand]."""
    d: dict = {}
    for c, key in ((0, "junctions_1"), (1, "junctions_2")):
        for j in record[key]:
            k = (int(j["start"]), int(j["end"]))
            v = d.get(k)
            if v is None:
                v = d[k] = [0, 0, int(j["motif"]), int(j["min_anchor"]),
                            j["strand"].decode() if isinstance(j["strand"], bytes) else str(j["strand"])]
            v[c] += int(j["count"])
            v[3] = min(v[3], int(j["min_anchor"]))
            if v[2] <= 0 < int(j["motif"]):
                v[2] = int(j["motif"])
    return d


def location(start_1b: int, end_1b: int, win_start: int, win_end: int) -> str:
    """Intron bounds vs the window's 1-based (win_start, win_end]."""
    a_in = win_start < start_1b <= win_end
    b_in = win_start < end_1b <= win_end
    if a_in and b_in:
        return "in"
    if a_in or b_in:
        return "out1"
    return "out2"


def window_stats(record: dict) -> tuple[dict, list]:
    """(per-window row, per-tail-junction rows) for one extractor record."""
    s, e = int(record["start"]), int(record["end"])
    width_kb = max(e - s, 1) / 1000.0
    d = window_junctions(record)
    expressed = [k for k, v in d.items() if v[0] + v[1] > TAIL_MAX]
    starts = {k[0] for k in expressed}
    ends = {k[1] for k in expressed}
    n_one = sum(1 for v in d.values() if (v[0] == 0) != (v[1] == 0))
    tail = []
    for (a, b), v in d.items():
        total = v[0] + v[1]
        if total > TAIL_MAX:
            continue
        tail.append({
            "loc": location(a, b, s, e),
            "intron_len": b - a + 1,
            "motif": v[2],
            "canonical": v[2] > 0,
            "shared": a in starts or b in ends,
            "min_anchor": v[3],
            "total": total,
            "one_cond": (v[0] == 0) != (v[1] == 0),
            "same_strand": v[4] == str(record.get("strand", "?")),
        })
    cov1, cov2 = record["cov_1"], record["cov_2"]
    row = {
        "width_kb": width_kb,
        "n_junctions": len(d),
        "n_tail": len(tail),
        "n_one_cond": n_one,
        "tail_per_kb": len(tail) / width_kb,
        "jx_per_kb": len(d) / width_kb,
        "one_cond_frac": (n_one / len(d)) if d else float("nan"),
        "depth": float((cov1.mean() + cov2.mean()) / 2.0) if len(cov1) else 0.0,
        "inside_frac": float(np.mean([t["loc"] == "in" for t in tail])) if tail else float("nan"),
        "canonical_frac": float(np.mean([t["canonical"] for t in tail])) if tail else float("nan"),
    }
    return row, tail


def summarize(rows: list, tails: list) -> dict:
    """The A1 statistics plus the reported, ungated context."""
    if not rows:
        return {"n_windows": 0}
    nj = np.array([r["n_junctions"] for r in rows], dtype=float)
    nt = np.array([r["n_tail"] for r in rows], dtype=float)
    ocf = np.array([r["one_cond_frac"] for r in rows], dtype=float)
    out = {
        "n_windows": len(rows),
        "n_tail_junctions": len(tails),
        # --- gated (A1) ---
        "tail_per_kb_median": float(np.median([r["tail_per_kb"] for r in rows])),
        "tail_frac": float(nt.sum() / nj.sum()) if nj.sum() else float("nan"),
        "one_cond_frac": float(np.nanmean(ocf)) if np.isfinite(ocf).any() else float("nan"),
        "inside_frac": float(np.mean([t["loc"] == "in" for t in tails])) if tails else float("nan"),
        "canonical_frac": float(np.mean([t["canonical"] for t in tails])) if tails else float("nan"),
        # --- reported, not gated ---
        "jx_per_window_median": float(np.median(nj)),
        "tail_per_window_median": float(np.median(nt)),
        "tail_per_window_mean": float(nt.mean()),
        "jx_per_kb_median": float(np.median([r["jx_per_kb"] for r in rows])),
        "width_kb_median": float(np.median([r["width_kb"] for r in rows])),
        "depth_median": float(np.median([r["depth"] for r in rows])),
    }
    if tails:
        loc = Counter(t["loc"] for t in tails)
        out["loc_frac"] = {k: loc.get(k, 0) / len(tails) for k in ("in", "out1", "out2")}
        split = Counter(t["total"] for t in tails)
        out["count_split"] = {str(k): split.get(k, 0) / len(tails) for k in range(1, TAIL_MAX + 1)}
        out["one_cond_frac_of_tail"] = float(np.mean([t["one_cond"] for t in tails]))
        out["shared_frac"] = float(np.mean([t["shared"] for t in tails]))
        out["anchor_lt10_frac"] = float(np.mean([t["min_anchor"] < 10 for t in tails]))
        by_loc = {}
        for l in ("in", "out1", "out2"):
            ts = [t for t in tails if t["loc"] == l]
            if not ts:
                continue
            L = np.array([t["intron_len"] for t in ts], dtype=float)
            by_loc[l] = {
                "n": len(ts),
                "canonical": float(np.mean([t["canonical"] for t in ts])),
                "shared": float(np.mean([t["shared"] for t in ts])),
                "intron_len_q25_50_75": [float(x) for x in np.quantile(L, [0.25, 0.5, 0.75])],
                "anchor_lt10": float(np.mean([t["min_anchor"] < 10 for t in ts])),
            }
        out["by_loc"] = by_loc
    return out


# --------------------------------------------------------------------------
# loading


def _labels_for(npz_dir: str) -> str | None:
    for cand in (pjoin(npz_dir, "labels.tsv"), pjoin(os.path.dirname(npz_dir.rstrip("/")), "labels.tsv")):
        if os.path.exists(cand):
            return cand
    return None


def parse_set(spec: str) -> tuple[str, list]:
    """``NAME=DIR[@LABELS][:IXPREFIX][+DIR...]`` -> (name, [(dir, labels, prefix)])."""
    if "=" not in spec:
        raise ValueError(f"--set {spec!r}: expected NAME=DIR[...]")
    name, rest = spec.split("=", 1)
    parts = []
    for item in rest.split("+"):
        prefix = ""
        if ":" in item:
            item, prefix = item.rsplit(":", 1)
        labels = None
        if "@" in item:
            item, labels = item.split("@", 1)
        parts.append((item, labels or _labels_for(item), prefix))
    return name, parts


def list_records(parts: list) -> list:
    """[(path, label or None)] over a set's dirs, in sorted-path order."""
    out = []
    for npz_dir, labels, prefix in parts:
        label_of = {}
        if labels and os.path.exists(labels):
            with open(labels) as f:
                label_of = {r["ix"]: r["label"] for r in csv.DictReader(f, delimiter="\t")}
        for name in sorted(os.listdir(npz_dir)):
            if not name.endswith(".npz"):
                continue
            ix = name.split(".", 1)[0]
            if prefix and not ix.startswith(prefix):
                continue
            out.append((pjoin(npz_dir, name), label_of.get(ix)))
    return out


def load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in ("start", "end", "cov_1", "cov_2", "junctions_1", "junctions_2")}
        rec["strand"] = str(z["strand"]) if "strand" in z.files else "?"
    return rec


def measure(records: list, max_n: int | None, seed: int) -> tuple[list, list, list]:
    """(window rows, tail rows, labels) over a seeded subsample of <= max_n."""
    if max_n and len(records) > max_n:
        idx = np.random.default_rng(seed).choice(len(records), size=max_n, replace=False)
        records = [records[i] for i in sorted(idx)]
    rows, tails, labels = [], [], []
    for path, label in records:
        row, tail = window_stats(load_npz(path))
        row["label"] = label
        row["_tails"] = tail
        for t in tail:
            t["label"] = label
        rows.append(row)
        tails.extend(tail)
        labels.append(label)
    return rows, tails, labels


def gate_a1(summary: dict, ranges: dict = GATE_A1) -> dict:
    out = {}
    for k, (lo, hi) in ranges.items():
        v = summary.get(k, float("nan"))
        out[k] = {"value": v, "range": [lo, hi], "pass": bool(lo <= v <= hi)}
    out["PASS"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def weighted_summary(rows: list, tails_per_row: list, w: np.ndarray) -> dict:
    """The five A1 statistics with per-window weights (importance weights for
    width matching). Tail-level statistics weight each tail junction by its
    window's weight."""
    w = np.asarray(w, dtype=float)
    tk = np.array([r["tail_per_kb"] for r in rows])
    o = np.argsort(tk, kind="stable")
    c = np.cumsum(w[o]) / w.sum()
    nt = np.array([r["n_tail"] for r in rows], dtype=float)
    nj = np.array([r["n_junctions"] for r in rows], dtype=float)
    ocf = np.array([r["one_cond_frac"] for r in rows], dtype=float)
    m = np.isfinite(ocf)
    n_tail_w = sum(wi * len(ts) for wi, ts in zip(w, tails_per_row))
    return {
        "tail_per_kb_median": float(tk[o][np.searchsorted(c, 0.5)]),
        "tail_frac": float((w * nt).sum() / (w * nj).sum()),
        "one_cond_frac": float((w[m] * ocf[m]).sum() / w[m].sum()),
        "inside_frac": float(sum(wi * sum(t["loc"] == "in" for t in ts)
                                 for wi, ts in zip(w, tails_per_row)) / n_tail_w),
        "canonical_frac": float(sum(wi * sum(t["canonical"] for t in ts)
                                    for wi, ts in zip(w, tails_per_row)) / n_tail_w),
        "effective_n": float(w.sum() ** 2 / (w ** 2).sum()),
    }


def width_match_weights(rows: list, target_rows: list, n_bins: int = 10) -> np.ndarray:
    """Importance weights that give ``rows`` the log-width distribution of
    ``target_rows`` (quantile bins of the target)."""
    tw = np.log10([r["width_kb"] for r in target_rows])
    edges = np.quantile(tw, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p_t = np.histogram(tw, edges)[0] / len(tw)
    rw = np.log10([r["width_kb"] for r in rows])
    b = np.clip(np.searchsorted(edges, rw, side="right") - 1, 0, n_bins - 1)
    p_r = np.bincount(b, minlength=n_bins) / len(rows)
    return p_t[b] / np.maximum(p_r[b], 1e-12)


A3_FEATURES = ("tail_per_kb", "one_cond_frac", "inside_frac", "canonical_frac")


def a3_label_blindness(rows: list, seed: int = 0) -> dict:
    """Gate A3: 5-fold LightGBM on tail-only per-window features predicting the
    class; balanced accuracy must be <= 0.30 (chance 0.20). One gene per event
    within a batch, so a stratified split cannot leak a gene."""
    import lightgbm as lgb
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    rows = [r for r in rows if r.get("label")]
    X = np.array([[r[k] for k in A3_FEATURES] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows])
    classes = sorted(set(y))
    yi = np.array([classes.index(c) for c in y])
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yi):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                 random_state=seed, verbose=-1)
        clf.fit(X[tr], yi[tr])
        pred[te] = clf.predict(X[te])
    bacc = float(balanced_accuracy_score(yi, pred))
    return {"n": len(y), "classes": classes, "features": list(A3_FEATURES),
            "balanced_accuracy": bacc, "threshold": 0.30, "pass": bacc <= 0.30}


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, dest="sets",
                    help="NAME=DIR[@LABELS][:IXPREFIX][+DIR...]; repeatable")
    ap.add_argument("--max-n", type=int, default=MAX_N_DEFAULT,
                    help="seeded subsample per set (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-class", action="store_true", help="also summarize per label")
    ap.add_argument("--gate-a1", action="append", default=[],
                    help="set name(s) to check against the Gate A1 ranges")
    ap.add_argument("--a1-ranges", choices=("amended", "original"), default="amended",
                    help="amended (2026-09-24, width-matched; the live gate) or the original frozen ranges")
    ap.add_argument("--ranges-from", nargs="*", default=None,
                    help="also print [min, max] of each A1 statistic across these sets")
    ap.add_argument("--width-match", default=None, metavar="SET",
                    help="with --ranges-from: also print those sets' A1 statistics reweighted to "
                    "SET's window-width distribution (tail/kb and inside fraction are width-confounded)")
    ap.add_argument("--a3", action="append", default=[],
                    help="set name(s) to run the Gate A3 label-blindness check on (needs labels)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    result = {"definitions": {"tail_max": TAIL_MAX, "max_n": args.max_n, "seed": args.seed}, "sets": {}}
    for spec in args.sets:
        name, parts = parse_set(spec)
        records = list_records(parts)
        rows, tails, labels = measure(records, args.max_n, args.seed)
        entry = {"spec": spec, "n_records_available": len(records), "all": summarize(rows, tails)}
        if args.by_class:
            entry["by_class"] = {}
            for c in sorted({l for l in labels if l}):
                entry["by_class"][c] = summarize([r for r in rows if r["label"] == c],
                                                 [t for t in tails if t["label"] == c])
        result["sets"][name] = entry
        rows_by_set = result.setdefault("_rows", {})
        rows_by_set[name] = rows
        if name in args.a3:
            entry["a3"] = a3_label_blindness(rows, args.seed)
        s = entry["all"]
        print(f"{name:>12} n={s['n_windows']:>5}  tail/kb {_fmt(s.get('tail_per_kb_median'))}  "
              f"tail_frac {_fmt(s.get('tail_frac'))}  one_cond {_fmt(s.get('one_cond_frac'))}  "
              f"inside {_fmt(s.get('inside_frac'))}  canonical {_fmt(s.get('canonical_frac'))}  "
              f"| jx/win {_fmt(s.get('jx_per_window_median'))}  tail/win {_fmt(s.get('tail_per_window_median'))}  "
              f"depth {_fmt(s.get('depth_median'))}  loc {({k: round(v, 3) for k, v in s.get('loc_frac', {}).items()})}")
        if args.by_class:
            for c, cs in entry["by_class"].items():
                print(f"{'':>12}   {c:>4} n={cs['n_windows']:>5}  tail/kb {_fmt(cs.get('tail_per_kb_median'))}  "
                      f"tail_frac {_fmt(cs.get('tail_frac'))}  one_cond {_fmt(cs.get('one_cond_frac'))}  "
                      f"inside {_fmt(cs.get('inside_frac'))}  canonical {_fmt(cs.get('canonical_frac'))}")

    if args.ranges_from:
        rng_out = {}
        for k in GATE_A1:
            vals = [result["sets"][n]["all"][k] for n in args.ranges_from]
            rng_out[k] = [float(min(vals)), float(max(vals))]
        result["ranges_from"] = {"sets": args.ranges_from, "ranges": rng_out}
        print(f"\n[min, max] across {args.ranges_from}:")
        for k, (lo, hi) in rng_out.items():
            print(f"  {k:>20}: {lo:.3f} - {hi:.3f}   (frozen A1: {GATE_A1[k][0]} - {GATE_A1[k][1]})")

    if args.ranges_from and args.width_match:
        target = result["_rows"][args.width_match]
        wm = {}
        for n in args.ranges_from:
            rows = result["_rows"][n]
            wm[n] = weighted_summary(rows, [r["_tails"] for r in rows], width_match_weights(rows, target))
        wr = {k: [min(wm[n][k] for n in wm), max(wm[n][k] for n in wm)] for k in GATE_A1}
        result["ranges_from"]["width_matched_to"] = args.width_match
        result["ranges_from"]["width_matched"] = {"per_set": wm, "ranges": wr}
        print(f"\n[min, max] across {args.ranges_from}, width-matched to {args.width_match} "
              f"(effective n {', '.join('%.0f' % wm[n]['effective_n'] for n in wm)}):")
        for k, (lo, hi) in wr.items():
            print(f"  {k:>20}: {lo:.3f} - {hi:.3f}")

    a1_ranges = GATE_A1_AMENDED if args.a1_ranges == "amended" else GATE_A1
    for name in args.gate_a1:
        g = gate_a1(result["sets"][name]["all"], a1_ranges)
        result.setdefault("gate_a1", {})[name] = g
        print(f"\nGate A1 on {name} ({args.a1_ranges} ranges): {'PASS' if g['PASS'] else 'FAIL'}")
        for k, v in g.items():
            if isinstance(v, dict):
                print(f"  {k:>20}: {v['value']:.3f} in [{v['range'][0]}, {v['range'][1]}] -> "
                      f"{'ok' if v['pass'] else 'OUT'}")

    for name in args.a3:
        a3 = result["sets"][name]["a3"]
        print(f"\nGate A3 on {name}: balanced accuracy {a3['balanced_accuracy']:.3f} (n={a3['n']}, "
              f"threshold <= 0.30) -> {'PASS' if a3['pass'] else 'FAIL'}")

    result.pop("_rows", None)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
