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
