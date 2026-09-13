"""Verification checks for dsignal/features.py (Phase 3 step 1 Verification).

Four independent checks, per plans/signal-native-v2-phase3-model.md:

1. **Topology** -- a hand-built A3 record (two junctions sharing a donor,
   differing in acceptor) must give `n_sharing_donor == 2`,
   `n_sharing_acceptor == 1`, and donor-group count fractions summing to 1
   within the group, per condition.
2. **Canonicalization is an involution** -- mirroring a normalized record
   twice returns the original coordinates, coverage, and motif codes.
3. **Flip is an exact A3<->A5 map** -- the feature matrix of a flipped A3
   record equals that of a naturally-built A5 record with mirrored
   coordinates, column for column.
4. **Coverage context off-by-one** -- on real records from the 1260-event sim
   set, the token's intron-interior coverage must equal a manual
   `cov[start - 1 - rs : end - rs]` slice mean, and the donor/acceptor flank
   means must equal the manual slices immediately outside it.

Checks 5-6 (coverage signal, coverage profile symmetries) were added by
Stage 12; check 7 (Stage 18.5: the opt-in covshape/delcov event groups keep
the base 32 columns bit-identical and respect mirror / condition-swap /
depth-scale symmetries) by Stage 18.

Usage:
    python -m dsignal.verify_features [--npz-dir sim/experiment/npz_geomwin]
"""

from __future__ import annotations

import argparse
import glob
import sys
from os.path import join as pjoin

import numpy as np

from dsignal.extract import JUNCTION_DTYPE, load_record
from dsignal.features import (
    EVENT_FEATURE_NAMES,
    TOKEN_FEATURE_NAMES,
    FeatureConfig,
    build_features,
    mirror,
    normalize_record,
)


def _junctions(rows) -> np.ndarray:
    """rows = [(start, end, count, motif, min_anchor), ...] -> junction array."""
    arr = np.zeros(len(rows), dtype=JUNCTION_DTYPE)
    strand_of = {1: b"+", 2: b"-", 3: b"+", 4: b"-", 5: b"+", 6: b"-"}
    for i, (start, end, count, motif, anchor) in enumerate(rows):
        arr[i] = (start, end, count, float(count), motif, strand_of.get(motif, b"?"), anchor)
    return arr


def _record(start, end, j1, j2, strand="+", cov=None):
    n = end - start
    cov = np.full(n, 100, dtype=np.uint32) if cov is None else cov
    return {
        "chrom": "1",
        "start": start,
        "end": end,
        "strand": strand,
        "cov_1": cov,
        "delcov_1": np.zeros(n, dtype=np.uint32),
        "junctions_1": _junctions(j1),
        "cov_2": cov.copy(),
        "delcov_2": np.zeros(n, dtype=np.uint32),
        "junctions_2": _junctions(j2),
        "stats_1": {},
        "stats_2": {},
    }


def _col(tokens, name):
    return tokens[:, TOKEN_FEATURE_NAMES.index(name)]


def check_topology(verbose: bool) -> int:
    """A3: two junctions share a donor at 1200, acceptors differ (1800/1850).
    A third, unrelated junction elsewhere must not join either group."""
    rec = _record(
        1000,
        3000,
        j1=[(1200, 1800, 80, 1, 8), (1200, 1850, 20, 1, 8), (2200, 2400, 50, 1, 8)],
        j2=[(1200, 1800, 30, 1, 8), (1200, 1850, 70, 1, 8), (2200, 2400, 50, 1, 8)],
    )
    tokens, event, meta = build_features(rec)
    failed = 0

    donors = _col(tokens, "n_sharing_donor")
    acceptors = _col(tokens, "n_sharing_acceptor")
    # rows are sorted by (start, end): [1200-1800, 1200-1850, 2200-2400]
    if not np.allclose(donors, [2, 2, 1]):
        print(f"FAIL n_sharing_donor: {donors} != [2, 2, 1]")
        failed += 1
    if not np.allclose(acceptors, [1, 1, 1]):
        print(f"FAIL n_sharing_acceptor: {acceptors} != [1, 1, 1]")
        failed += 1

    for cond in (1, 2):
        frac = _col(tokens, f"donor_group_frac_{cond}")
        if not np.isclose(frac[0] + frac[1], 1.0):
            print(f"FAIL donor group fractions (cond {cond}) sum to {frac[0] + frac[1]}, not 1")
            failed += 1
        if not np.isclose(frac[2], 1.0):
            print(f"FAIL lone junction's donor-group fraction (cond {cond}) is {frac[2]}, not 1")
            failed += 1

    # the A3 signature: the donor group's split shifts between conditions
    delta = _col(tokens, "donor_group_frac_delta")
    if not (np.isclose(delta[0], 0.8 - 0.3) and np.isclose(delta[1], 0.2 - 0.7)):
        print(f"FAIL donor_group_frac_delta {delta[:2]} != [0.5, -0.5]")
        failed += 1

    if verbose and not failed:
        print("  topology: n_sharing_donor/acceptor, group fractions, and delta all as expected")
    print(f"check 1 (topology): {'OK' if not failed else f'{failed} FAILURES'}")
    return failed


def check_mirror_involution(verbose: bool) -> int:
    rec = _record(
        1000,
        3000,
        j1=[(1200, 1800, 80, 1, 8), (1200, 1850, 20, 3, 6), (2200, 2400, 50, 5, 9)],
        j2=[(1200, 1800, 30, 1, 8)],
        strand="-",
        cov=np.arange(2000, dtype=np.uint32),
    )
    failed = 0
    for canon in (True, False):
        norm = normalize_record(rec, FeatureConfig(canonicalize=canon))
        twice = mirror(mirror(norm))
        for field in ("s_rel", "e_rel", "count_1", "count_2", "motif", "cov_1", "cov_2"):
            a, b = getattr(norm, field), getattr(twice, field)
            if not np.array_equal(a, b):
                print(f"FAIL mirror involution (canonicalize={canon}) on {field}")
                failed += 1
        if twice.flipped != norm.flipped:
            print(f"FAIL mirror involution (canonicalize={canon}) on the flipped flag")
            failed += 1
    if verbose and not failed:
        print("  mirror(mirror(x)) == x on coordinates, counts, motifs and coverage")
    print(f"check 2 (mirror involution): {'OK' if not failed else f'{failed} FAILURES'}")
    return failed


def check_flip_maps_a3_to_a5(verbose: bool) -> int:
    """Flipping an A3 record (shared donor) must yield exactly the feature
    matrix of the mirror-image A5 record (shared acceptor) built naturally.

    Mirroring within a window of length n maps the window-relative inclusive
    interval [s, e] to [n-1-e, n-1-s]; in 1-based genomic terms with window
    start rs, position p maps to 2*rs + n + 1 - p.
    """
    rs, re = 1000, 3000
    n = re - rs
    cov = np.arange(n, dtype=np.uint32)

    def mirror_pos(p):
        return 2 * rs + n + 1 - p

    a3_j1 = [(1200, 1800, 80, 1, 8), (1200, 1850, 20, 1, 8), (2200, 2400, 50, 3, 9)]
    a3_j2 = [(1200, 1800, 30, 1, 8), (1200, 1850, 70, 1, 8), (2200, 2400, 50, 3, 9)]

    def mirrored(rows):
        # a mirrored junction's donor is the old acceptor, and its motif is the
        # reverse-complement partner (1<->2, 3<->4, 5<->6)
        flip_motif = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 0: 0}
        return sorted(
            (mirror_pos(e), mirror_pos(s), c, flip_motif[m], a) for (s, e, c, m, a) in rows
        )

    a3 = _record(rs, re, a3_j1, a3_j2, strand="+", cov=cov)
    a5 = _record(rs, re, mirrored(a3_j1), mirrored(a3_j2), strand="+", cov=cov[::-1].copy())

    cfg = FeatureConfig()
    a3_flipped = mirror(normalize_record(a3, cfg))
    tok_flipped, ev_flipped, _ = build_features(a3, cfg, norm=a3_flipped)
    tok_a5, ev_a5, _ = build_features(a5, cfg)

    failed = 0
    if tok_flipped.shape != tok_a5.shape:
        print(f"FAIL flipped-A3 token shape {tok_flipped.shape} != natural-A5 {tok_a5.shape}")
        return 1
    for c, name in enumerate(TOKEN_FEATURE_NAMES):
        if not np.allclose(tok_flipped[:, c], tok_a5[:, c], atol=1e-5):
            print(
                f"FAIL flipped A3 != natural A5 on token column {name!r}: "
                f"{tok_flipped[:, c]} vs {tok_a5[:, c]}"
            )
            failed += 1

    # the discriminative check: the flipped record's shared structure is an
    # acceptor group, where the original's was a donor group
    if not np.allclose(_col(tok_flipped, "n_sharing_acceptor"), _col(tok_a5, "n_sharing_acceptor")):
        failed += 1
    tok_a3, _, _ = build_features(a3, cfg)
    if _col(tok_a3, "n_sharing_donor").max() != 2 or _col(tok_flipped, "n_sharing_donor").max() != 1:
        print("FAIL flip did not turn a shared donor into a shared acceptor")
        failed += 1
    if not np.allclose(ev_flipped, ev_a5, atol=1e-5):
        bad = [
            name
            for i, name in enumerate(EVENT_FEATURE_NAMES)
            if not np.isclose(ev_flipped[i], ev_a5[i], atol=1e-5)
        ]
        print(f"FAIL flipped-A3 event vector != natural-A5 on: {bad}")
        failed += 1

    if verbose and not failed:
        print("  flip(A3) == A5 on all token columns and the event vector")
    print(f"check 3 (flip is an exact A3<->A5 map): {'OK' if not failed else f'{failed} FAILURES'}")
    return failed


def check_coverage_offsets(npz_dir: str, n_records: int, verbose: bool) -> int:
    """Real-data spot check: token coverage-context columns vs manual slices.

    Uses canonicalize=False so token row i still corresponds to the natural
    sorted (start, end) order and coordinates are directly comparable.
    """
    paths = sorted(glob.glob(pjoin(npz_dir, "*.npz")))[:n_records]
    if not paths:
        print(f"check 4 SKIPPED: no .npz under {npz_dir}")
        return 0

    cfg = FeatureConfig(canonicalize=False)
    failed = 0
    n_checked = 0
    for path in paths:
        rec = load_record(path)
        tokens, _, meta = build_features(rec, cfg)
        rs = rec["start"]
        keys = sorted(
            set(zip(rec["junctions_1"]["start"].tolist(), rec["junctions_1"]["end"].tolist()))
            | set(zip(rec["junctions_2"]["start"].tolist(), rec["junctions_2"]["end"].tolist()))
        )
        if len(keys) != tokens.shape[0]:
            print(f"FAIL {path}: {tokens.shape[0]} tokens for {len(keys)} union junctions")
            failed += 1
            continue
        cov = np.asarray(rec["cov_1"], dtype=np.float64)

        def slice_mean(lo, hi):
            """Mean over the clipped half-open [lo, hi). Clipping must happen
            *before* indexing: junctions can lie wholly outside the window
            (a read overlapping it can splice out), and a raw negative index
            would silently wrap to the far end of the array."""
            lo_c, hi_c = max(int(lo), 0), min(int(hi), len(cov))
            if hi_c <= lo_c:
                return 0.0
            return float(cov[lo_c:hi_c].mean())

        for i, (js, je) in enumerate(keys):
            lo, hi = js - 1 - rs, je - rs  # interior slice, 0-based half-open
            for label, want, got_name in (
                ("interior cov", slice_mean(lo, hi), "log1p_interior_cov_1"),
                ("donor flank", slice_mean(lo - cfg.flank, lo), "log1p_flank_donor_cov_1"),
                ("acceptor flank", slice_mean(hi, hi + cfg.flank), "log1p_flank_acc_cov_1"),
            ):
                got = float(np.expm1(_col(tokens, got_name)[i]))
                if not np.isclose(got, want, rtol=1e-4, atol=1e-3):
                    print(f"FAIL {path} junction {js}-{je}: {label} {got} != {want}")
                    failed += 1
            n_checked += 1

    if verbose:
        print(f"  checked {n_checked} junctions across {len(paths)} real records")
    print(f"check 4 (coverage-context offsets): {'OK' if not failed else f'{failed} FAILURES'}")
    return failed


def check_coverage_signal(npz_dir: str, n_records: int, verbose: bool) -> int:
    """Sanity, not correctness: across real records, spliced-out introns
    should on average carry *less* coverage than their flanking exons, i.e.
    the median interior:flank ratio must be well below 1. If this ever comes
    out around or above 1, the interior slice is pointing at exons."""
    paths = sorted(glob.glob(pjoin(npz_dir, "*.npz")))[:n_records]
    if not paths:
        return 0
    ratios = []
    for path in paths:
        rec = load_record(path)
        tokens, _, _ = build_features(rec)
        if tokens.shape[0]:
            r = _col(tokens, "interior_flank_ratio_1")
            ratios.extend(r[r > 0].tolist())
    if not ratios:
        print("check 5 SKIPPED: no junctions with flank coverage")
        return 0
    median = float(np.median(ratios))
    ok = median < 0.75
    print(
        f"check 5 (interior:flank ratio sanity): {'OK' if ok else 'FAILURE'} "
        f"-- median {median:.3f} over {len(ratios)} junctions (want < 0.75)"
    )
    return 0 if ok else 1


def check_coverage_profile(npz_dir: str, n_records: int, verbose: bool) -> int:
    """The coverage branch's input must respect the same symmetries the rest
    of the representation does (Stage 12 / Track D):

      * mirroring a record reverses its profile exactly -- otherwise the flip
        augmentation stops being label-preserving for the branch;
      * swapping the two conditions swaps the two rows exactly -- otherwise
        condition-swap TTA averages two views that are not the same event.

    Both are checked as exact equalities, not tolerances: the binning uses
    fractional edges precisely so that they hold bit-for-bit.
    """
    from dsignal.dataset import augment_condition_swap
    from dsignal.features import build_coverage_profile

    paths = sorted(glob.glob(pjoin(npz_dir, "*.npz")))[:n_records]
    if not paths:
        print("check 6 SKIPPED: no records")
        return 0
    worst_mirror = worst_swap = 0.0
    for path in paths:
        norm = normalize_record(load_record(path))
        prof = build_coverage_profile(norm)
        worst_mirror = max(
            worst_mirror,
            float(np.abs(build_coverage_profile(mirror(norm)) - prof[:, ::-1]).max()),
        )
        worst_swap = max(
            worst_swap,
            float(np.abs(build_coverage_profile(augment_condition_swap(norm)) - prof[::-1]).max()),
        )
    ok = worst_mirror == 0.0 and worst_swap == 0.0
    print(
        f"check 6 (coverage profile symmetries): {'OK' if ok else 'FAILURE'} "
        f"-- mirror err {worst_mirror:g}, condition-swap err {worst_swap:g} "
        f"over {len(paths)} records (want exactly 0)"
    )
    return 0 if ok else 1


def check_extra_event_groups(npz_dir: str, n_records: int, verbose: bool) -> int:
    """Check 7 (Stage 18.5): the opt-in `covshape`/`delcov` event groups
    respect the representation's symmetries, and opting in leaves the 32 base
    columns untouched:

      * base 32 columns identical with and without the groups;
      * mirror leaves every extra column unchanged (autocorrelation and
        deletion summaries are reversal-invariant);
      * condition swap exchanges the `_1`/`_2` columns exactly;
      * depth rescaling ("scale") leaves autocorrelation, deletion rate and
        nonzero fraction unchanged (log1p_delcov_mean is depth-absolute and
        is on the DEPTH_ABSOLUTE list instead);
      * a record with no deletions gives zero delcov columns.
    """
    from dsignal.dataset import augment_condition_swap
    from dsignal.features import EXTRA_EVENT_GROUPS, event_feature_names, rescale_depth

    cfg = FeatureConfig(event_groups=("covshape", "delcov"))
    names = event_feature_names(cfg)
    n_base = len(EVENT_FEATURE_NAMES)
    extra = names[n_base:]
    assert extra == EXTRA_EVENT_GROUPS["covshape"] + EXTRA_EVENT_GROUPS["delcov"], extra
    paths = sorted(glob.glob(pjoin(npz_dir, "*.npz")))[:n_records]
    if not paths:
        print("check 7 SKIPPED: no records")
        return 0
    failed = 0
    worst = {"base": 0.0, "mirror": 0.0, "swap": 0.0, "scale": 0.0}
    n_with_del = 0
    swap_pairs = [(names.index(n), names.index(n.replace("_1", "_2"))) for n in extra if n.endswith("_1")]
    scale_inv = [names.index(n) for n in extra if not n.startswith("log1p_")]
    for path in paths:
        rec = load_record(path)
        norm = normalize_record(rec, cfg)
        _, ev, _ = build_features(cfg=cfg, norm=norm)
        _, ev0, _ = build_features(cfg=FeatureConfig(), norm=normalize_record(rec, FeatureConfig()))
        worst["base"] = max(worst["base"], float(np.abs(ev[:n_base] - ev0).max()))
        _, ev_m, _ = build_features(cfg=cfg, norm=mirror(norm))
        worst["mirror"] = max(worst["mirror"], float(np.abs(ev_m[n_base:] - ev[n_base:]).max()))
        _, ev_s, _ = build_features(cfg=cfg, norm=augment_condition_swap(norm))
        for a, b in swap_pairs:
            worst["swap"] = max(worst["swap"], abs(float(ev_s[a] - ev[b])), abs(float(ev_s[b] - ev[a])))
        scaled = rescale_depth(norm, FeatureConfig(depth_norm="scale", event_groups=cfg.event_groups))
        _, ev_k, _ = build_features(cfg=cfg, norm=scaled)
        worst["scale"] = max(worst["scale"], float(np.abs(ev_k[scale_inv] - ev[scale_inv]).max()))
        if rec["delcov_1"].sum() + rec["delcov_2"].sum() > 0:
            n_with_del += 1
        elif np.abs(ev[[names.index(n) for n in EXTRA_EVENT_GROUPS["delcov"]]]).max() != 0:
            print(f"FAIL {path}: no deletions but nonzero delcov columns")
            failed += 1
    tol = 1e-5
    for k, v in worst.items():
        if v > tol:
            print(f"FAIL check 7 {k}: max deviation {v:g} > {tol}")
            failed += 1
    if verbose:
        print(f"  extra columns {extra}; {n_with_del}/{len(paths)} records carry deletions; "
              f"max deviations {worst}")
    print(f"check 7 (Stage 18.5 covshape/delcov groups: base columns untouched, mirror-invariant, "
          f"condition-swap exact, scale-invariant): {'OK' if not failed else f'{failed} FAILURES'}")
    return failed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", default="sim/experiment/npz_geomwin")
    ap.add_argument("--n-records", type=int, default=25)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    failed = 0
    failed += check_topology(args.verbose)
    failed += check_mirror_involution(args.verbose)
    failed += check_flip_maps_a3_to_a5(args.verbose)
    failed += check_coverage_offsets(args.npz_dir, args.n_records, args.verbose)
    failed += check_coverage_signal(args.npz_dir, args.n_records, args.verbose)
    failed += check_coverage_profile(args.npz_dir, args.n_records, args.verbose)
    failed += check_extra_event_groups(args.npz_dir, args.n_records, args.verbose)

    print(f"\n{'ALL CHECKS PASSED' if not failed else f'{failed} CHECK FAILURES'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
