"""Shared junction-topology + coverage-context feature builder (Phase 3 step 1).

Consumed by *both* ``dsignal/baseline_gbm.py`` and ``dsignal/dataset.py`` so the
LightGBM baseline and the junction-set Transformer are compared on identical
information rather than on incidentally different feature sets.

Two products per extractor record (``dsignal.extract.load_record``):

* a **per-junction token matrix** ``[J, F]``, rows = the union of ``(start, end)``
  across both conditions, carrying counts, motif, *topology* (who shares this
  junction's donor / acceptor, and what fraction of the donor- resp.
  acceptor-group's reads this junction holds, per condition) and *coverage
  context* (flank vs intron-interior coverage, the RI signal);
* an **event-level vector** ``[E]`` -- window geometry, coverage moments,
  pooled token summaries -- used as the Transformer's CLS token and, extended
  with token aggregates (:func:`build_gbm_row`), as the LightGBM feature row.

Coordinate conventions, the easy place to be silently wrong:

* junction ``start``/``end`` are STAR ``jI``: **1-based, inclusive** of the
  first/last *intronic* base;
* ``cov`` is 0-based relative to the record's ``start``, which
  ``extract_condition`` handed to ``pysam.fetch`` as a 0-based coordinate.

So the window-relative interior of a junction is
``cov[start - 1 - rs : end - rs]`` and its length is ``end - start + 1``.
Junctions may legitimately fall partly or wholly *outside* the window (a read
overlapping the window can splice out of it), so every coverage lookup is
clipped and flagged rather than assumed in-bounds.

**Strand canonicalization**: when the record's GTF-free strand vote is ``-``,
every coordinate is mirrored within the window (and motif codes are mapped to
their reverse-complement partners) before features are built, so "donor" always
means the transcription-direction donor. That makes A3/A5 well-defined
independent of strand and makes the left-right-flip augmentation an exact
A3<->A5 map. Strand ``?`` records are left unflipped and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Motif codes are STAR's jM (after the extractor's `% 20` annotated-bit mask):
# 1=GT/AG 2=CT/AC 3=GC/AG 4=CT/GC 5=AT/AC 6=GT/AT, 0=non-canonical, -1=unknown.
# Mirroring a record swaps each motif with its reverse-complement partner, so
# that in canonicalized space "1" always means the canonical GT/AG donor site.
_MOTIF_FLIP = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 0: 0, -1: -1}

# one-hot levels: the six canonical codes plus a catch-all for 0 / -1
_MOTIF_LEVELS = (1, 2, 3, 4, 5, 6)
N_MOTIF_LEVELS = len(_MOTIF_LEVELS) + 1

DEFAULT_FLANK = 50


@dataclass
class FeatureConfig:
    flank: int = DEFAULT_FLANK
    #: keep at most this many junctions (highest total count first); the
    #: discarded tail is summarized into the event vector, never dropped
    #: silently. None = keep everything.
    max_junctions: int | None = None
    #: canonicalize minus-strand records by mirroring within the window
    canonicalize: bool = True
    #: how to treat absolute sequencing depth -- see `rescale_depth`.
    #: "none" (the historical behaviour), "scale", or "drop".
    depth_norm: str = "none"
    #: reference spliced depth that "scale" maps every record onto. Arbitrary
    #: but shared by train and inference; chosen to sit in the numeric range
    #: the unnormalized features already occupied, so nothing else retunes.
    depth_target: float = 100.0


def _token_feature_names(cfg: FeatureConfig) -> list[str]:
    names = [
        "log10_intron_len",
        "log1p_count_1",
        "log1p_count_2",
        "log1p_nhw_1",
        "log1p_nhw_2",
        "log2_count_ratio",
        "count_frac_1",
        "count_frac_2",
        "count_frac_delta",
        "log1p_min_anchor",
    ]
    names += [f"motif_{m}" for m in _MOTIF_LEVELS] + ["motif_other"]
    names += [
        "n_sharing_donor",
        "n_sharing_acceptor",
        "donor_group_frac_1",
        "donor_group_frac_2",
        "donor_group_frac_delta",
        "acc_group_frac_1",
        "acc_group_frac_2",
        "acc_group_frac_delta",
    ]
    for c in (1, 2):
        names += [
            f"log1p_flank_donor_cov_{c}",
            f"log1p_flank_acc_cov_{c}",
            f"log1p_interior_cov_{c}",
            f"interior_flank_ratio_{c}",
            f"interior_nonzero_frac_{c}",
        ]
    names += [
        "is_top",
        "shares_donor_with_top",
        "shares_acceptor_with_top",
        "nested_in_top",
        "contains_top",
        "overlaps_top",
        "disjoint_from_top",
    ]
    names += ["rel_start", "rel_end", "rel_mid", "in_window", "present_1", "present_2"]
    return names


def _event_feature_names() -> list[str]:
    names = ["log10_window_width"]
    for c in (1, 2):
        names += [
            f"log1p_cov_mean_{c}",
            f"log1p_cov_median_{c}",
            f"log1p_cov_max_{c}",
            f"zero_cov_frac_{c}",
            f"cov_cv_{c}",
        ]
    names += [
        "log1p_n_junctions",
        "log1p_n_junctions_1",
        "log1p_n_junctions_2",
        "strand_is_minus",
        "strand_is_unknown",
        "max_n_sharing_donor",
        "mean_n_sharing_donor",
        "max_n_sharing_acceptor",
        "mean_n_sharing_acceptor",
        "frac_junctions_sharing_donor",
        "frac_junctions_sharing_acceptor",
        "max_interior_flank_ratio_1",
        "max_interior_flank_ratio_2",
        "max_abs_donor_group_frac_delta",
        "max_abs_acc_group_frac_delta",
        "max_abs_log2_count_ratio",
        "max_interior_nonzero_frac_1",
        "max_interior_nonzero_frac_2",
        "log1p_n_junctions_dropped",
        "dropped_count_frac",
        "frac_junctions_out_of_window",
    ]
    return names


TOKEN_FEATURE_NAMES = _token_feature_names(FeatureConfig())
EVENT_FEATURE_NAMES = _event_feature_names()
N_TOKEN_FEATURES = len(TOKEN_FEATURE_NAMES)
N_EVENT_FEATURES = len(EVENT_FEATURE_NAMES)

#: Features that scale linearly with sequencing depth *before* their log1p --
#: multiply every read in the window by k and these move by log1p(k), while
#: every other feature here is a ratio, a fraction, a count of things, or
#: geometry, and does not move at all.
#:
#: An event is the same event at 5x and at 300x, so absolute depth is not a
#: property of the class and these columns are where a depth-calibration error
#: in the simulation reaches the model. See `rescale_depth`.
DEPTH_ABSOLUTE_TOKEN_FEATURES = (
    "log1p_count_1", "log1p_count_2", "log1p_nhw_1", "log1p_nhw_2",
    "log1p_flank_donor_cov_1", "log1p_flank_acc_cov_1", "log1p_interior_cov_1",
    "log1p_flank_donor_cov_2", "log1p_flank_acc_cov_2", "log1p_interior_cov_2",
)
DEPTH_ABSOLUTE_EVENT_FEATURES = (
    "log1p_cov_mean_1", "log1p_cov_median_1", "log1p_cov_max_1",
    "log1p_cov_mean_2", "log1p_cov_median_2", "log1p_cov_max_2",
)

# One feature is depth-dependent without being depth-*absolute*, and zeroing
# the list above does not fix it: `log2_count_ratio` is
# log2((count_1 + 1) / (count_2 + 1)), and the pseudocount makes it a function
# of scale. Measured over 40 records at 0.2x and 5x depth it moves by up to
# 2.17 (and `max_abs_log2_count_ratio` by 2.115) with every declared column
# zeroed.
#
# This matters more than its size suggests: it is precisely the
# between-condition ratio -- the differential quantity that *should* be the
# whole signal -- and at low depth the pseudocount compresses it toward zero.
# The curated benchmark sits at ~1/5 of the simulation's depth, so real events
# arrive with their apparent differential systematically damped, on top of the
# genuine shortfall the audit measured. "scale" addresses it by mapping every
# record onto the reference depth before the pseudocount applies; "drop" does
# not, which is why "drop" is only an ablation.
#
# **Measured, and it is a null.** The mechanism above is real and the fix is
# exact (max feature change 0.0 under 0.2x/5x rescaling), but on the benchmark
# "scale" changes nothing: 77.9% vs 78.6% collapsed, McNemar p = 1.000, and
# +2 of 35 even on the shallowest depth quartile where this predicted the gain.
# "drop" costs both sim-CV and transfer, so the model does use depth magnitude
# productively. `depth_norm` therefore defaults to "none". The argument for
# adopting "scale" is design, not evidence -- the benchmark is one dataset at
# one depth and structurally cannot test what invariance protects against.


def depth_absolute_masks() -> tuple:
    """(token_mask, event_mask) over TOKEN_/EVENT_FEATURE_NAMES."""
    tok = np.array([n in DEPTH_ABSOLUTE_TOKEN_FEATURES for n in TOKEN_FEATURE_NAMES])
    ev = np.array([n in DEPTH_ABSOLUTE_EVENT_FEATURES for n in EVENT_FEATURE_NAMES])
    return tok, ev


# --------------------------------------------------------------------------
# normalization / canonicalization
# --------------------------------------------------------------------------


@dataclass
class NormalizedRecord:
    """Window-relative, optionally mirrored view of an extractor record.

    ``s_rel``/``e_rel`` are 0-based, **inclusive** window-relative indices of
    the first/last intronic base (so ``cov[s_rel:e_rel + 1]`` is the interior);
    they may be negative or >= ``n`` for junctions reaching outside the window.
    """

    n: int
    cov_1: np.ndarray
    cov_2: np.ndarray
    s_rel: np.ndarray
    e_rel: np.ndarray
    count_1: np.ndarray
    count_2: np.ndarray
    nhw_1: np.ndarray
    nhw_2: np.ndarray
    motif: np.ndarray
    min_anchor: np.ndarray
    strand: str
    flipped: bool


def _union_junctions(j1: np.ndarray, j2: np.ndarray):
    """Union of (start, end) keys across conditions, with per-condition counts
    (zero where a junction is present in only one condition)."""
    keys: dict = {}
    for arr, slot in ((j1, 0), (j2, 1)):
        for row in arr:
            key = (int(row["start"]), int(row["end"]))
            rec = keys.setdefault(
                key,
                {"count": [0.0, 0.0], "nhw": [0.0, 0.0], "motif": -1, "min_anchor": None},
            )
            rec["count"][slot] += float(row["count"])
            rec["nhw"][slot] += float(row["nh_weighted_count"])
            motif = int(row["motif"])
            if rec["motif"] == -1 and motif != -1:
                rec["motif"] = motif
            anchor = int(row["min_anchor"])
            rec["min_anchor"] = anchor if rec["min_anchor"] is None else min(rec["min_anchor"], anchor)
    return keys


def normalize_record(record: dict, cfg: FeatureConfig | None = None, force_flip: bool | None = None) -> NormalizedRecord:
    """Window-relative view of a record, mirrored if the strand vote says so.

    ``force_flip`` overrides the strand-driven decision (used by the
    left-right-flip augmentation, which mirrors an *already* canonicalized
    record and swaps the A3/A5 label).
    """
    cfg = cfg or FeatureConfig()
    rs = int(record["start"])
    cov_1 = np.asarray(record["cov_1"], dtype=np.float64)
    cov_2 = np.asarray(record["cov_2"], dtype=np.float64)
    n = int(record["end"]) - rs
    if n <= 0:  # degenerate window; fall back to whatever coverage we got
        n = max(len(cov_1), len(cov_2))

    keys = _union_junctions(record["junctions_1"], record["junctions_2"])
    if keys:
        items = sorted(keys.items())
        starts = np.array([k[0] for k, _ in items], dtype=np.int64)
        ends = np.array([k[1] for k, _ in items], dtype=np.int64)
        count_1 = np.array([v["count"][0] for _, v in items], dtype=np.float64)
        count_2 = np.array([v["count"][1] for _, v in items], dtype=np.float64)
        nhw_1 = np.array([v["nhw"][0] for _, v in items], dtype=np.float64)
        nhw_2 = np.array([v["nhw"][1] for _, v in items], dtype=np.float64)
        motif = np.array([v["motif"] for _, v in items], dtype=np.int64)
        min_anchor = np.array([v["min_anchor"] for _, v in items], dtype=np.float64)
    else:
        starts = ends = np.zeros(0, dtype=np.int64)
        count_1 = count_2 = nhw_1 = nhw_2 = min_anchor = np.zeros(0, dtype=np.float64)
        motif = np.zeros(0, dtype=np.int64)

    # jI is 1-based inclusive of the first/last intronic base; cov is 0-based
    # relative to `rs`.
    s_rel = starts - 1 - rs
    e_rel = ends - 1 - rs

    strand = str(record.get("strand", "?"))
    if force_flip is None:
        flip = cfg.canonicalize and strand == "-"
    else:
        flip = bool(force_flip)

    if flip:
        s_rel, e_rel = (n - 1 - e_rel), (n - 1 - s_rel)
        cov_1 = cov_1[::-1].copy()
        cov_2 = cov_2[::-1].copy()
        motif = np.array([_MOTIF_FLIP.get(int(m), int(m)) for m in motif], dtype=np.int64)
        # lexicographic (start, end) -- sorting on start alone leaves
        # shared-donor/shared-acceptor ties in an arbitrary order, which makes
        # mirroring non-involutive (verify_features check 2 catches exactly this)
        order = np.lexsort((e_rel, s_rel))
        s_rel, e_rel = s_rel[order], e_rel[order]
        count_1, count_2 = count_1[order], count_2[order]
        nhw_1, nhw_2 = nhw_1[order], nhw_2[order]
        motif, min_anchor = motif[order], min_anchor[order]

    return NormalizedRecord(
        n=n,
        cov_1=cov_1,
        cov_2=cov_2,
        s_rel=s_rel,
        e_rel=e_rel,
        count_1=count_1,
        count_2=count_2,
        nhw_1=nhw_1,
        nhw_2=nhw_2,
        motif=motif,
        min_anchor=min_anchor,
        strand=strand,
        flipped=flip,
    )


def depth_scale(norm: NormalizedRecord) -> float:
    """The record's local spliced depth: the pooled count of its **top**
    junction.

    A max, not a sum, so it does not move with how many junctions a window
    happens to contain, and it measures *spliced* depth specifically -- pooled
    mean coverage would have worked too but is inflated exactly where an intron
    is retained, which would deflate an RI window's junction counts and bleed
    the class into the scale it is normalized by.

    Pooled across both conditions on purpose: normalizing each condition by its
    own depth would divide out the between-condition difference, which is the
    entire signal. Falls back to pooled mean coverage for a junctionless window,
    and returns 0 when there is nothing at all.
    """
    if len(norm.count_1):
        s = float(np.max(norm.count_1 + norm.count_2))
        if s > 0:
            return s
    if norm.n:
        s = float((norm.cov_1.mean() + norm.cov_2.mean()) / 2.0)
        if s > 0:
            return s
    return 0.0


def rescale_depth(norm: NormalizedRecord, cfg: FeatureConfig) -> NormalizedRecord:
    """Map a record onto a reference depth, so that features built from it are
    invariant to how deeply the locus was sequenced.

    Every read count and every coverage value is multiplied by the same factor
    ``depth_target / depth_scale(norm)``. One shared factor is what makes this
    exactly invariant: scale a window's reads by k and the factor divides by k,
    so every feature -- absolute or relative, within-condition or between --
    comes out unchanged. The differential signal survives untouched, because
    both conditions are scaled identically.

    The argument for it is a *representation* argument, not a robustness trick.
    Depth is not a property of an event: the same A3 event is an A3 event at 5x
    and at 300x. Training on a simulation whose depth is miscalibrated
    (measured: 5.2x the curated benchmark, `dsignal/audit_noise.py`) puts every
    depth-absolute feature in the wrong place at inference, and no amount of
    subsample augmentation fixes a feature that should not carry scale in the
    first place.

    **That argument has been tested and it did not pay.** On the benchmark this
    is a null (77.9% vs 78.6% collapsed, McNemar p = 1.000; +2 of 35 on the
    shallowest depth quartile). It is kept, and `depth_norm` defaults to
    "none" -- the benchmark is one dataset at one depth, so it structurally
    cannot test what invariance protects against, and the null is therefore
    weak evidence rather than a refutation. Adopt it on design grounds if at
    all, not by pointing at a number.

    A no-op when ``cfg.depth_norm != "scale"``, or when the record is empty.
    """
    if cfg.depth_norm != "scale":
        return norm
    s = depth_scale(norm)
    if s <= 0:
        return norm
    k = cfg.depth_target / s
    return NormalizedRecord(
        n=norm.n,
        cov_1=norm.cov_1 * k,
        cov_2=norm.cov_2 * k,
        s_rel=norm.s_rel,
        e_rel=norm.e_rel,
        count_1=norm.count_1 * k,
        count_2=norm.count_2 * k,
        nhw_1=norm.nhw_1 * k,
        nhw_2=norm.nhw_2 * k,
        motif=norm.motif,
        min_anchor=norm.min_anchor,
        strand=norm.strand,
        flipped=norm.flipped,
    )


def mirror(norm: NormalizedRecord) -> NormalizedRecord:
    """Mirror an already-normalized record within its window (the flip
    augmentation). An involution: ``mirror(mirror(x)) == x``."""
    n = norm.n
    s_rel = n - 1 - norm.e_rel
    e_rel = n - 1 - norm.s_rel
    order = np.lexsort((e_rel, s_rel))  # see normalize_record on why not argsort(s_rel)
    return NormalizedRecord(
        n=n,
        cov_1=norm.cov_1[::-1].copy(),
        cov_2=norm.cov_2[::-1].copy(),
        s_rel=s_rel[order],
        e_rel=e_rel[order],
        count_1=norm.count_1[order],
        count_2=norm.count_2[order],
        nhw_1=norm.nhw_1[order],
        nhw_2=norm.nhw_2[order],
        motif=np.array([_MOTIF_FLIP.get(int(m), int(m)) for m in norm.motif], dtype=np.int64)[order],
        min_anchor=norm.min_anchor[order],
        strand=norm.strand,
        flipped=not norm.flipped,
    )


# --------------------------------------------------------------------------
# coverage context
# --------------------------------------------------------------------------


def _mean_slice(cov: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """(mean, nonzero fraction) over the clipped window-relative slice
    ``[lo, hi)``. Returns (0, 0) when the slice lies entirely outside."""
    lo_c, hi_c = max(lo, 0), min(hi, len(cov))
    if hi_c <= lo_c:
        return 0.0, 0.0
    seg = cov[lo_c:hi_c]
    return float(seg.mean()), float((seg > 0).mean())


def _coverage_context(cov: np.ndarray, s_rel: np.ndarray, e_rel: np.ndarray, flank: int):
    """Per-junction (flank-outside-donor, flank-outside-acceptor, interior
    mean, interior:flank ratio, interior nonzero fraction)."""
    j = len(s_rel)
    out = np.zeros((j, 5), dtype=np.float64)
    for i in range(j):
        s, e = int(s_rel[i]), int(e_rel[i])
        fd, _ = _mean_slice(cov, s - flank, s)
        fa, _ = _mean_slice(cov, e + 1, e + 1 + flank)
        interior, nz = _mean_slice(cov, s, e + 1)
        flank_mean = (fd + fa) / 2.0
        ratio = interior / flank_mean if flank_mean > 0 else 0.0
        out[i] = (fd, fa, interior, ratio, nz)
    return out


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------


def _group_fractions(key: np.ndarray, count: np.ndarray):
    """Per-junction share of the summed count of every junction with the same
    ``key`` (donor position or acceptor position), plus the group size."""
    j = len(key)
    frac = np.zeros(j, dtype=np.float64)
    size = np.zeros(j, dtype=np.float64)
    totals: dict = {}
    sizes: dict = {}
    for i in range(j):
        k = int(key[i])
        totals[k] = totals.get(k, 0.0) + float(count[i])
        sizes[k] = sizes.get(k, 0) + 1
    for i in range(j):
        k = int(key[i])
        total = totals[k]
        frac[i] = float(count[i]) / total if total > 0 else 0.0
        size[i] = sizes[k]
    return frac, size


def build_features(
    record: dict | None = None,
    cfg: FeatureConfig | None = None,
    norm: NormalizedRecord | None = None,
):
    """``(tokens [J, N_TOKEN_FEATURES], event_vec [N_EVENT_FEATURES], meta)``.

    Pass either a raw ``record`` or an already-normalized ``norm`` (the
    augmentation path in ``dsignal.dataset`` mutates the normalized record and
    rebuilds features from it, so everything here is derived from ``norm``
    alone -- ``record`` is only a convenience entry point).

    ``meta`` carries the kept junctions' window-relative coordinates, the
    discrete pairwise-relation inputs the Transformer's attention bias needs
    (``s_rel``/``e_rel``), so the model does not have to re-derive them.
    """
    cfg = cfg or FeatureConfig()
    if norm is None:
        if record is None:
            raise ValueError("build_features needs either `record` or `norm`")
        norm = normalize_record(record, cfg)

    # Applied here rather than in `normalize_record` so it lands *after* the
    # augmentations in dsignal.dataset, which mutate the normalized record.
    # The two are complementary, not redundant: rescaling removes the
    # systematic depth *scale*, while `augment_subsample` thins counts
    # binomially, so the sampling noise a shallow locus really carries survives
    # the rescale and is still worth training against.
    norm = rescale_depth(norm, cfg)

    n = norm.n
    s_rel, e_rel = norm.s_rel, norm.e_rel
    total_j = len(s_rel)

    # --- cap: keep the highest-count junctions, summarize the dropped tail ---
    count_tot = norm.count_1 + norm.count_2
    n_dropped, dropped_count_frac = 0, 0.0
    if cfg.max_junctions is not None and total_j > cfg.max_junctions:
        keep = np.argsort(-count_tot, kind="stable")[: cfg.max_junctions]
        keep = np.sort(keep)
        n_dropped = total_j - len(keep)
        grand = count_tot.sum()
        dropped_count_frac = float((grand - count_tot[keep].sum()) / grand) if grand > 0 else 0.0
        s_rel, e_rel = s_rel[keep], e_rel[keep]
        count_1, count_2 = norm.count_1[keep], norm.count_2[keep]
        nhw_1, nhw_2 = norm.nhw_1[keep], norm.nhw_2[keep]
        motif, min_anchor = norm.motif[keep], norm.min_anchor[keep]
    else:
        count_1, count_2 = norm.count_1, norm.count_2
        nhw_1, nhw_2 = norm.nhw_1, norm.nhw_2
        motif, min_anchor = norm.motif, norm.min_anchor

    j = len(s_rel)
    tokens = np.zeros((j, N_TOKEN_FEATURES), dtype=np.float32)

    if j:
        intron_len = np.maximum(e_rel - s_rel + 1, 1)
        tot_1, tot_2 = count_1.sum(), count_2.sum()

        donor_frac_1, donor_size = _group_fractions(s_rel, count_1)
        donor_frac_2, _ = _group_fractions(s_rel, count_2)
        acc_frac_1, acc_size = _group_fractions(e_rel, count_1)
        acc_frac_2, _ = _group_fractions(e_rel, count_2)

        cov_ctx_1 = _coverage_context(norm.cov_1, s_rel, e_rel, cfg.flank)
        cov_ctx_2 = _coverage_context(norm.cov_2, s_rel, e_rel, cfg.flank)

        top = int(np.argmax(count_1 + count_2))
        top_s, top_e = int(s_rel[top]), int(e_rel[top])

        cols: list[np.ndarray] = [
            np.log10(intron_len.astype(np.float64)),
            np.log1p(count_1),
            np.log1p(count_2),
            np.log1p(nhw_1),
            np.log1p(nhw_2),
            np.log2((count_1 + 1.0) / (count_2 + 1.0)),
            count_1 / tot_1 if tot_1 > 0 else np.zeros(j),
            count_2 / tot_2 if tot_2 > 0 else np.zeros(j),
            (count_1 / tot_1 if tot_1 > 0 else np.zeros(j))
            - (count_2 / tot_2 if tot_2 > 0 else np.zeros(j)),
            np.log1p(min_anchor),
        ]
        for m in _MOTIF_LEVELS:
            cols.append((motif == m).astype(np.float64))
        cols.append((~np.isin(motif, _MOTIF_LEVELS)).astype(np.float64))
        cols += [
            donor_size,
            acc_size,
            donor_frac_1,
            donor_frac_2,
            donor_frac_1 - donor_frac_2,
            acc_frac_1,
            acc_frac_2,
            acc_frac_1 - acc_frac_2,
        ]
        for ctx in (cov_ctx_1, cov_ctx_2):
            cols += [
                np.log1p(ctx[:, 0]),
                np.log1p(ctx[:, 1]),
                np.log1p(ctx[:, 2]),
                ctx[:, 3],
                ctx[:, 4],
            ]
        idx = np.arange(j)
        nested = (s_rel >= top_s) & (e_rel <= top_e) & (idx != top)
        contains = (s_rel <= top_s) & (e_rel >= top_e) & (idx != top)
        overlaps = (s_rel <= top_e) & (e_rel >= top_s) & ~nested & ~contains & (idx != top)
        disjoint = (s_rel > top_e) | (e_rel < top_s)
        cols += [
            (idx == top).astype(np.float64),
            ((s_rel == top_s) & (idx != top)).astype(np.float64),
            ((e_rel == top_e) & (idx != top)).astype(np.float64),
            nested.astype(np.float64),
            contains.astype(np.float64),
            overlaps.astype(np.float64),
            disjoint.astype(np.float64),
        ]
        span = float(max(n, 1))
        cols += [
            s_rel / span,
            e_rel / span,
            (s_rel + e_rel) / (2.0 * span),
            ((s_rel >= 0) & (e_rel < n)).astype(np.float64),
            (count_1 > 0).astype(np.float64),
            (count_2 > 0).astype(np.float64),
        ]
        tokens = np.stack(cols, axis=1).astype(np.float32)

    event = _build_event_vector(norm, tokens, n_dropped, dropped_count_frac, n)

    if cfg.depth_norm == "drop":
        # The blunt ablation: zero the depth-absolute columns instead of
        # rescaling them. Strictly less informative than "scale" (it discards
        # within-window magnitude too, not just the global scale), and it
        # exists to separate "absolute depth was actively misleading" from
        # "absolute depth was merely miscalibrated".
        tok_mask, ev_mask = depth_absolute_masks()
        if tokens.size:
            tokens[:, tok_mask] = 0.0
        event = np.asarray(event, dtype=np.float32)
        event[ev_mask] = 0.0

    meta = {
        "s_rel": s_rel.astype(np.int64),
        "e_rel": e_rel.astype(np.int64),
        "n": n,
        "flipped": norm.flipped,
        "n_junctions_total": total_j,
        "n_dropped": n_dropped,
    }
    return tokens, event, meta


def _col(name: str) -> int:
    return TOKEN_FEATURE_NAMES.index(name)


def _build_event_vector(norm, tokens, n_dropped, dropped_count_frac, n):
    def cov_stats(cov):
        if len(cov) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        mean = float(cov.mean())
        return [
            float(np.log1p(mean)),
            float(np.log1p(np.median(cov))),
            float(np.log1p(cov.max())),
            float((cov == 0).mean()),
            float(cov.std() / mean) if mean > 0 else 0.0,
        ]

    j = tokens.shape[0]

    def tmax(name):
        return float(tokens[:, _col(name)].max()) if j else 0.0

    def tmean(name):
        return float(tokens[:, _col(name)].mean()) if j else 0.0

    def tabsmax(name):
        return float(np.abs(tokens[:, _col(name)]).max()) if j else 0.0

    vals = [float(np.log10(max(n, 1)))]
    vals += cov_stats(norm.cov_1)
    vals += cov_stats(norm.cov_2)
    vals += [
        float(np.log1p(norm.s_rel.shape[0])),
        float(np.log1p(int((norm.count_1 > 0).sum()))),
        float(np.log1p(int((norm.count_2 > 0).sum()))),
        1.0 if norm.strand == "-" else 0.0,
        1.0 if norm.strand == "?" else 0.0,
        tmax("n_sharing_donor"),
        tmean("n_sharing_donor"),
        tmax("n_sharing_acceptor"),
        tmean("n_sharing_acceptor"),
        float((tokens[:, _col("n_sharing_donor")] > 1).mean()) if j else 0.0,
        float((tokens[:, _col("n_sharing_acceptor")] > 1).mean()) if j else 0.0,
        tmax("interior_flank_ratio_1"),
        tmax("interior_flank_ratio_2"),
        tabsmax("donor_group_frac_delta"),
        tabsmax("acc_group_frac_delta"),
        tabsmax("log2_count_ratio"),
        tmax("interior_nonzero_frac_1"),
        tmax("interior_nonzero_frac_2"),
        float(np.log1p(n_dropped)),
        float(dropped_count_frac),
        float(1.0 - tokens[:, _col("in_window")].mean()) if j else 0.0,
    ]
    assert len(vals) == N_EVENT_FEATURES, (len(vals), N_EVENT_FEATURES)
    return np.asarray(vals, dtype=np.float32)


# --------------------------------------------------------------------------
# LightGBM row: event vector + token aggregates
# --------------------------------------------------------------------------

_AGG_STATS = ("max", "mean", "wmean")


def gbm_feature_names() -> list[str]:
    names = list(EVENT_FEATURE_NAMES)
    for stat in _AGG_STATS:
        names += [f"{stat}__{n}" for n in TOKEN_FEATURE_NAMES]
    return names


def build_gbm_row(tokens: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Event vector concatenated with per-column max / mean / count-weighted
    mean over the token matrix -- the LightGBM view of the same information the
    Transformer gets token-by-token."""
    j = tokens.shape[0]
    if j:
        w = tokens[:, _col("count_frac_1")] + tokens[:, _col("count_frac_2")]
        w_sum = w.sum()
        wmean = (tokens * w[:, None]).sum(axis=0) / w_sum if w_sum > 0 else tokens.mean(axis=0)
        aggs = [tokens.max(axis=0), tokens.mean(axis=0), wmean]
    else:
        aggs = [np.zeros(N_TOKEN_FEATURES, dtype=np.float32)] * len(_AGG_STATS)
    return np.concatenate([event] + [a.astype(np.float32) for a in aggs])


def build_all(record: dict, cfg: FeatureConfig | None = None):
    """Convenience: ``(tokens, event, gbm_row, meta)``."""
    tokens, event, meta = build_features(record, cfg)
    return tokens, event, build_gbm_row(tokens, event), meta


# --------------------------------------------------------------------------
# coverage profile (Stage 12 / Track D)
# --------------------------------------------------------------------------

#: bins across the window for the coverage branch. 128 is a compromise: fine
#: enough that a retained intron's edge lands in its own bin for a typical
#: few-kb window, coarse enough that the branch cannot memorize base-level
#: noise that never transfers from simulation to real data.
DEFAULT_COVERAGE_BINS = 128


def build_coverage_profile(
    norm: NormalizedRecord, n_bins: int = DEFAULT_COVERAGE_BINS
) -> np.ndarray:
    """``[2, n_bins]`` depth-normalized log coverage, one row per condition.

    The hand-derived token/event features summarize coverage at junction
    anchors and as window-level moments; neither carries the *shape* of
    coverage along the window -- and the C1 analysis (Stage 12) found the
    intron/exon coverage profile to be exactly where simulation and real data
    diverge for CSE and NN. This is that shape, at a resolution the model can
    use and the simulator cannot trivially fake.

    Depth-normalized by ``depth_scale`` (the pooled top-junction count) and
    pooled across conditions, so a window's absolute depth divides out while
    the *between-condition* difference -- the entire signal -- does not.

    **Exactly reversal-symmetric**: bin edges are fractional (``b * n /
    n_bins``) and each bin is the exact integral of the piecewise-constant
    coverage over its interval, so the profile of a mirrored record is the
    column-reversed profile of the original, up to float rounding. Integer
    edges would not be: rounding breaks the symmetry at odd window lengths,
    and the flip augmentation would then quietly stop being an involution.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    out = np.zeros((2, n_bins), dtype=np.float32)
    n = int(norm.n)
    if n == 0:
        return out

    scale = depth_scale(norm)
    if scale <= 0:
        scale = 1.0

    edges = np.arange(n_bins + 1, dtype=np.float64) * (n / n_bins)
    for row, cov in enumerate((norm.cov_1, norm.cov_2)):
        c = np.asarray(cov, dtype=np.float64)
        # cumulative integral of the step function, evaluated at fractional x:
        # full bases up to floor(x) plus the partial base at the boundary
        csum = np.concatenate(([0.0], np.cumsum(c)))
        lo = np.floor(edges).astype(np.int64)
        lo = np.clip(lo, 0, n)
        frac = edges - lo
        partial = np.where(lo < n, c[np.clip(lo, 0, n - 1)] * frac, 0.0)
        integral = csum[lo] + partial
        per_bin = np.diff(integral) / (n / n_bins)
        out[row] = np.log1p(np.maximum(per_bin, 0.0) / scale)
    return out
