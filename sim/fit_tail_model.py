"""Fit the Stage 19 junction-tail model and the indel rate -> sim/tail_model.json.

plans/signal-native-v2-stage19-junction-tail.md, step 19.1. Reads extractor
npz of the three gene-disjoint *training* cohorts only -- aqr, prpf4, u2af2,
pos + nn. It never reads ptbp1 (92 genes shared with benchmark_new dev),
bnew_dev, tra2 or any ``real_ssl_*`` random-window pool, and asserts so:
that keeps the tail model ptbp1-free, and so every model trained on the
re-simulated pool stays eligible for the benchmark_new test spend.

Deterministic: no subsampling, no timestamps; re-running on the same npz
writes a byte-identical JSON.

The model (and why it is not the plan's per-kb rate)
----------------------------------------------------
The plan specified "an empirical quantile table of tail junctions per kb".
The fit data contradict a per-kb process: the per-window tail count is flat in
window width (median 8 / 8 / 9 tail junctions across width terciles
0.1-1.8 / 1.8-6.7 / 6.7-328 kb, Spearman 0.10), while tail/kb falls 20x
across the same terciles, and the location mix moves with width (narrow
windows: mostly both-ends-outside; wide windows: mostly inside). That is the
signature of *long spurious junctions ("arcs") intersecting the window*: an
arc of intron length L intersects a window of width W at W + L - 1 genomic
offsets, and with E[L] ~ 100 kb >> W the count barely depends on W.

A first version fit exactly that -- one underlying arc-length density
recovered by inverse-intersection weighting 1 / (W_i + L_j), with the location
mix left to geometry -- and failed its own check: it put 0.69 of the arcs in
wide (6.7-328 kb) windows *inside* the window against 0.38 observed, because
short spurious junctions are not spread uniformly along the genome but sit near
expressed exons (78% of inside tail junctions share a site with an expressed
junction). So the generator (sim/build_tail_transcripts.py) is empirical and
conditional on window width instead -- label-blind, since width is the only
input:

1. ``width_bins``: 6 quantile bins of log10 window width over the fit windows.
   Per bin: the per-window tail-count pmf ``n_pmf``, the location mix
   ``loc_pmf`` (inside / one end out / both out), and per location a log10
   intron-length pmf ``len_pmf`` on ``log10_len_edges``. The generator draws
   N, then per arc a location, a length feasible for that location and width,
   and a uniform placement consistent with both. ``fit_check`` below confirms
   the location mix is reproduced per width tercile.
2. (kept from the plan) everything below is global, per location class.
3. ``p_canonical``, ``p_shared``, ``p_same_strand``: per location class.
4. ``count_split_obs``: observed 1/2/3-read split of tail junctions, and
   ``fragments_pmf``: the fragment-count pmf that reproduces it after
   polyester's PCR duplication (same ``pcr_rate``/``pcr_lambda`` as every batch)
   -- a triangular deconvolution, since duplicates only ever add reads.
5. ``anchor_pmf``: per-read anchor of single-read tail junctions (5..50; real
   reads are 100 bp, like the simulation's).

Indels: ``indel.del_base_ratio`` = pooled sum(delcov) / sum(cov). A deletion
event of length k contributes k deleted reference bases, so the per-base event
rate is ratio / E[k] with k ~ Geometric(0.6) capped at 5. Insertions are set
to the same event rate -- an assumption, since coverage arrays do not see
insertions.

Usage::

    python -m sim.fit_tail_model --out sim/tail_model.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

from sim.tail_stats import TAIL_MAX, load_npz, location, window_junctions

FIT_COHORTS = ("aqr", "prpf4", "u2af2")
FORBIDDEN = ("ptbp1", "real_dev", "bnew", "tra2", "real_ssl", "benchmark")

# log10 intron-length bins. 21 bp is STAR's alignIntronMin (a shorter gap is a
# deletion); the top edge is past STAR's default max intron (2^16 * 9 = 589,824).
LOG_L_LO, LOG_L_HI, N_L_BINS = np.log10(21.0), 6.0, 48
N_W_BINS = 6  # window-width quantile bins of the fit cohorts
ANCHOR_MIN, ANCHOR_MAX = 5, 50  # extractor min_anchor; half of a 100 bp read
INDEL_P, INDEL_MAX = 0.6, 5
PCR_RATE, PCR_LAMBDA = 0.1, 1.0  # sim/run_batch.sh defaults, every batch
MAX_FRAGMENTS = 3


def _default_dirs(root: str) -> list:
    return [os.path.join(root, f"real_{kind}_{c}", "npz") for c in FIT_COHORTS for kind in ("pos", "nn")]


def indel_len_pmf(p: float = INDEL_P, kmax: int = INDEL_MAX) -> np.ndarray:
    """P(k), k = 1..kmax, Geometric(p) truncated and renormalized."""
    k = np.arange(1, kmax + 1)
    w = p * (1 - p) ** (k - 1)
    return w / w.sum()


def pcr_copies_pmf(pcr_rate: float, lam: float, kmax: int = 12) -> np.ndarray:
    """P(copies = c), c = 0..kmax, for one fragment under the polyester fork's
    ``generate_fragments``: floor(n * pcr_rate) fragments are picked and each
    gains 1 + Poisson(lam) extra copies."""
    from math import exp, factorial

    pmf = np.zeros(kmax + 1)
    pmf[1] = 1.0 - pcr_rate
    for extra in range(1, kmax):
        n = extra - 1  # extra = 1 + Poisson
        pmf[1 + extra] += pcr_rate * exp(-lam) * lam ** n / factorial(n)
    return pmf / pmf.sum()


def deconvolve_fragments(split_obs: np.ndarray, pcr_rate: float, lam: float,
                         tmax: int = MAX_FRAGMENTS) -> np.ndarray:
    """Fragment-count pmf q(T), T = 1..tmax, whose read-count distribution after
    PCR duplication, restricted to 1..TAIL_MAX, matches ``split_obs``. Reads
    only ever increase, so the system is lower-triangular; clipped at 0."""
    c = pcr_copies_pmf(pcr_rate, lam)
    kmax = TAIL_MAX
    # M[k, T] = P(reads = k | T fragments), k, T = 1..kmax
    conv = [np.array([1.0])]
    for _ in range(tmax):
        conv.append(np.convolve(conv[-1], c))
    M = np.zeros((kmax, tmax))
    for T in range(1, tmax + 1):
        for k in range(1, kmax + 1):
            M[k - 1, T - 1] = conv[T][k] if k < len(conv[T]) else 0.0
    q = np.zeros(tmax)
    for k in range(kmax):
        resid = split_obs[k] - M[k, :k] @ q[:k]
        q[k] = max(resid / M[k, k], 0.0) if k < tmax else 0.0
    return q / q.sum()


def _bin_of(L: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, np.log10(L), side="right") - 1, 0, len(edges) - 2)


def fit(dirs: list) -> dict:
    for d in dirs:
        low = d.lower()
        if any(tok in low for tok in FORBIDDEN):
            raise SystemExit(f"refusing to fit on {d}: only aqr/prpf4/u2af2 pos+nn are allowed")

    edges = np.linspace(LOG_L_LO, LOG_L_HI, N_L_BINS + 1)
    n_tail_per_window, widths = [], []
    arcs = []  # (W, L, loc, canonical, shared, same_strand, total, min_anchor)
    sum_del = sum_cov = 0
    n_files = 0
    for d in dirs:
        for name in sorted(os.listdir(d)):
            if not name.endswith(".npz"):
                continue
            n_files += 1
            rec = load_npz(os.path.join(d, name))
            s, e = int(rec["start"]), int(rec["end"])
            with np.load(os.path.join(d, name)) as z:
                sum_del += int(z["delcov_1"].sum()) + int(z["delcov_2"].sum())
                sum_cov += int(z["cov_1"].sum()) + int(z["cov_2"].sum())
            jx = window_junctions(rec)
            expressed = [k for k, v in jx.items() if v[0] + v[1] > TAIL_MAX]
            starts, ends = {k[0] for k in expressed}, {k[1] for k in expressed}
            n = 0
            for (a, b), v in jx.items():
                total = v[0] + v[1]
                if total > TAIL_MAX:
                    continue
                n += 1
                arcs.append((e - s, b - a + 1, location(a, b, s, e), v[2] > 0,
                             a in starts or b in ends, v[4] == rec["strand"], total, v[3]))
            n_tail_per_window.append(n)
            widths.append(e - s)

    W = np.array([x[0] for x in arcs], dtype=float)
    L = np.array([x[1] for x in arcs], dtype=float)
    loc = np.array([x[2] for x in arcs])
    canon = np.array([x[3] for x in arcs])
    shared = np.array([x[4] for x in arcs])
    same = np.array([x[5] for x in arcs])
    total = np.array([x[6] for x in arcs])
    anchor = np.array([x[7] for x in arcs])

    # width-binned empirical tables: per-window count, location mix, and a
    # per-location log10 intron-length pmf, all conditional on the window's width
    widths_a = np.array(widths, dtype=float)
    w_edges = np.quantile(np.log10(widths_a), np.linspace(0, 1, N_W_BINS + 1))
    w_edges[0], w_edges[-1] = -np.inf, np.inf
    wb_window = np.clip(np.searchsorted(w_edges, np.log10(widths_a), side="right") - 1, 0, N_W_BINS - 1)
    wb_arc = np.clip(np.searchsorted(w_edges, np.log10(W), side="right") - 1, 0, N_W_BINS - 1)
    lb = _bin_of(L, edges)
    n_tail_a = np.array(n_tail_per_window)
    width_bins = []
    for i in range(N_W_BINS):
        mw, ma = wb_window == i, wb_arc == i
        cnt = np.bincount(n_tail_a[mw], minlength=1).astype(float)
        entry = {
            "log10_width_lo": None if not np.isfinite(w_edges[i]) else float(w_edges[i]),
            "log10_width_hi": None if not np.isfinite(w_edges[i + 1]) else float(w_edges[i + 1]),
            "n_windows": int(mw.sum()),
            "n_pmf": (cnt / cnt.sum()).tolist(),
            "loc_pmf": {l: float(np.mean(loc[ma] == l)) for l in ("in", "out1", "out2")},
            "len_pmf": {},
        }
        for l in ("in", "out1", "out2"):
            h = np.bincount(lb[ma & (loc == l)], minlength=N_L_BINS).astype(float)
            entry["len_pmf"][l] = (h / h.sum()).tolist() if h.sum() else [0.0] * N_L_BINS
        width_bins.append(entry)

    counts = np.bincount(n_tail_a, minlength=1).astype(float)
    split_obs = np.array([np.mean(total == k) for k in range(1, TAIL_MAX + 1)])
    frag_q = deconvolve_fragments(split_obs, PCR_RATE, PCR_LAMBDA)
    single = anchor[(total == 1)]
    single = single[(single >= ANCHOR_MIN) & (single <= ANCHOR_MAX)]
    anc = np.bincount(single - ANCHOR_MIN, minlength=ANCHOR_MAX - ANCHOR_MIN + 1).astype(float)

    by_loc = {}
    for l in ("in", "out1", "out2"):
        m = loc == l
        by_loc[l] = {
            "n": int(m.sum()),
            "p_canonical": float(canon[m].mean()),
            "p_shared": float(shared[m].mean()),
            "p_same_strand_given_canonical": float(same[m & canon].mean()),
        }

    len_pmf = indel_len_pmf()
    mean_len = float((np.arange(1, INDEL_MAX + 1) * len_pmf).sum())
    ratio = sum_del / sum_cov

    model = {
        "stage": "19.1",
        "provenance": {
            "fit_dirs": dirs,
            "n_windows": len(n_tail_per_window),
            "n_npz": n_files,
            "n_tail_junctions": int(len(arcs)),
            "tail_max_reads": TAIL_MAX,
            "note": "aqr/prpf4/u2af2 pos+nn only; never ptbp1, bnew_dev, tra2, real_ssl",
        },
        "n_per_window_pmf": (counts / counts.sum()).tolist(),
        "log10_len_edges": edges.tolist(),
        "width_bins": width_bins,
        "by_loc": by_loc,
        "count_split_obs": split_obs.tolist(),
        "fragments_pmf": frag_q.tolist(),
        "pcr": {"rate": PCR_RATE, "lambda": PCR_LAMBDA},
        "anchor": {"min": ANCHOR_MIN, "max": ANCHOR_MAX, "pmf": (anc / anc.sum()).tolist()},
        "indel": {
            "del_base_ratio": ratio,
            "sum_delcov": int(sum_del),
            "sum_cov": int(sum_cov),
            "len_geometric_p": INDEL_P,
            "len_max": INDEL_MAX,
            "mean_len": mean_len,
            "del_rate_per_base": ratio / mean_len,
            "ins_rate_per_base": ratio / mean_len,
            "note": "insertion rate set equal to the deletion rate (assumption: coverage arrays do not see insertions)",
        },
    }
    model["fit_check"] = fit_check(model, widths_a, n_tail_per_window, loc, W)
    return model


def fit_check(model: dict, widths: np.ndarray, n_tail: list, loc_obs: np.ndarray, W_obs: np.ndarray) -> dict:
    """Does the arc geometry alone reproduce the observed location mix, overall
    and per width tercile? Pure geometry (bins + uniform placement), seeded."""
    from sim.build_tail_transcripts import TailModel, sample_arc_geometry

    tm = TailModel(model)
    rng = np.random.default_rng(0)
    q = np.quantile(widths, [0, 1 / 3, 2 / 3, 1])
    out = {}
    for i in range(3):
        m = (widths >= q[i]) & (widths <= q[i + 1])
        ws = widths[m]
        sim = Counter()
        for w in ws[:2000]:
            for _ in range(20):
                a, b = sample_arc_geometry(tm, rng, 0, int(w))
                sim[location(a, b, 0, int(w))] += 1
        mo = (W_obs >= q[i]) & (W_obs <= q[i + 1])
        obs = Counter(loc_obs[mo].tolist())
        n_s, n_o = sum(sim.values()), sum(obs.values())
        out[f"width_{q[i] / 1000:.1f}-{q[i + 1] / 1000:.1f}kb"] = {
            "observed": {k: obs.get(k, 0) / n_o for k in ("in", "out1", "out2")},
            "model": {k: sim.get(k, 0) / n_s for k in ("in", "out1", "out2")},
            "tail_per_window_median": float(np.median(np.array(n_tail)[m])),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="sim/experiment", help="dir holding real_{pos,nn}_{aqr,prpf4,u2af2}/npz")
    ap.add_argument("--out", default="sim/tail_model.json")
    args = ap.parse_args(argv)
    model = fit(_default_dirs(args.root))
    with open(args.out, "w") as f:
        json.dump(model, f, indent=1, sort_keys=True)
        f.write("\n")
    p = model["provenance"]
    print(f"fit on {p['n_windows']} windows, {p['n_tail_junctions']} tail junctions -> {args.out}", file=sys.stderr)
    print(f"  split obs {np.round(model['count_split_obs'], 3).tolist()} -> fragments "
          f"{np.round(model['fragments_pmf'], 3).tolist()}", file=sys.stderr)
    for l, v in model["by_loc"].items():
        print(f"  {l:>4}: n={v['n']} canonical {v['p_canonical']:.3f} shared {v['p_shared']:.3f} "
              f"same-strand|canon {v['p_same_strand_given_canonical']:.3f}", file=sys.stderr)
    ind = model["indel"]
    print(f"  indel: delcov/cov {ind['del_base_ratio']:.3e}, mean len {ind['mean_len']:.3f} -> "
          f"rate {ind['del_rate_per_base']:.3e}/base (ins = del)", file=sys.stderr)
    for k, v in model["fit_check"].items():
        print(f"  {k}: observed {({a: round(b, 3) for a, b in v['observed'].items()})} "
              f"model {({a: round(b, 3) for a, b in v['model'].items()})}", file=sys.stderr)


if __name__ == "__main__":
    main()
