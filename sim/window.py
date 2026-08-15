"""Randomized extraction windows around an event span (plan Phase 2 step 5,
recalibrated by plans/signal-native-v2-window-geometry-fix.md).

Every row selected by `sim/select_events.py` has its own true span (the
SUPPA event span for positives/hard negatives, a sub-interval of the gene
span for plain/novel negatives). Using that span as the extraction window
verbatim would let a model learn "junction near byte 0/N" instead of
splicing signal, so the window is randomized around it.

**The first version of this module drew the flank as a log-uniform multiple
of the event span**, which made window *width* a function of the event span
-- and event span tracks the label (RI introns are short, CSE/NN spans are
not). Window geometry alone then reached 46.3% accuracy on the 1260-event
set, a shortcut that cannot transfer to the real benchmark, whose windows
come from a curator. Width is now drawn from a **span-independent**
log-normal fitted to the pooled curated benchmark
(`benchmark/{A3,A5,ES,IR}.csv`, n=140, all four CSVs pooled so the draw
carries no label information), truncated to that benchmark's observed
width range. The event is then placed uniformly at random inside the drawn
window, which keeps the off-center property the original sampler had.

The one residual span->width coupling is irreducible: a window must contain
its event, so a drawn width below the event span is clamped up to it. That
is counted (`sample_window_detail` returns a `clamped` flag) and reported by
`sim/select_events.py`, since the clamp rate is the honest measure of how
much span information is left in the width.
"""

from __future__ import annotations

import numpy as np

#: Pooled curated-benchmark window widths (1-based inclusive, n=140 over
#: benchmark/{A3,A5,ES,IR}.csv): log10 mean 3.4448, sd 0.7024, observed
#: range [2, 181766]. Measured, not assumed -- see the module docstring.
BENCH_LOG10_MEAN = 3.4448
BENCH_LOG10_SD = 0.7024
BENCH_MIN_WIDTH = 2
BENCH_MAX_WIDTH = 181766

#: Flank kept on each side when a drawn width has to be clamped up to the event
#: span, drawn log-uniform in this **absolute** bp range (never a multiple of
#: the span -- that is the coupling being removed). Without it a clamped window
#: would coincide exactly with the event boundaries, which is the original
#: "window == event" leak this module exists to remove, and every clamped
#: window would be exactly centered on its event.
MIN_FLANK_LO = 50
MIN_FLANK_HI = 500


def _draw_width(rng: np.random.Generator, log10_mean: float, log10_sd: float,
                min_width: int, max_width: int, max_tries: int = 64) -> int:
    """Log-normal width truncated to [min_width, max_width] by rejection, so
    the retained distribution is the true truncation rather than a clipped one
    with mass piled on the bounds. Falls back to clipping in the (practically
    unreachable) event that every draw is rejected."""
    for _ in range(max_tries):
        w = int(round(10.0 ** float(rng.normal(log10_mean, log10_sd))))
        if min_width <= w <= max_width:
            return w
    return int(np.clip(w, min_width, max_width))


def sample_window_detail(
    event_start: int,
    event_end: int,
    chrom_len: int,
    rng: np.random.Generator,
    log10_mean: float = BENCH_LOG10_MEAN,
    log10_sd: float = BENCH_LOG10_SD,
    min_width: int = BENCH_MIN_WIDTH,
    max_width: int = BENCH_MAX_WIDTH,
    min_flank_lo: int = MIN_FLANK_LO,
    min_flank_hi: int = MIN_FLANK_HI,
) -> tuple:
    """Returns (win_start, win_end, clamped), coordinates 1-based inclusive
    like the region_start/region_end convention already used in design.tsv.

    A window must contain its event, so a draw narrower than the event span
    (plus a log-uniform absolute flank of ``MIN_FLANK_LO``-``MIN_FLANK_HI`` bp
    on each side) is clamped up to it. ``clamped`` marks
    those rows: they are the residual, irreducible span->width coupling, and
    `sim/select_events.py` reports the rate. (Redrawing from the benchmark
    distribution *truncated below the span* was tried instead and rejected: it
    removes the deterministic width-given-span mapping on those rows but
    inflates the pooled width distribution well past the benchmark it is
    supposed to match -- log10 mean 3.96 vs 3.78 for clamping, against the
    benchmark's 3.44 -- while measuring the same at the leak test.)
    """
    span = max(event_end - event_start + 1, 1)
    width = _draw_width(rng, log10_mean, log10_sd, min_width, max_width)

    min_flank = int(round(np.exp(rng.uniform(np.log(min_flank_lo), np.log(min_flank_hi)))))
    floor_width = span + 2 * min_flank
    clamped = width < floor_width
    if clamped:
        width = floor_width

    # place the event uniformly at random inside the window
    slack = width - span
    left_flank = int(round(float(rng.uniform(0.0, 1.0)) * slack))

    win_start = event_start - left_flank
    win_end = win_start + width - 1

    # keep the window on the chromosome by *sliding* it, not shrinking it, so
    # the drawn width survives; only a window wider than the chromosome (never
    # the case for GRCh38 vs a 182kb ceiling) would end up truncated.
    if win_start < 1:
        win_start, win_end = 1, width
    if win_end > chrom_len:
        win_end = chrom_len
        win_start = max(win_end - width + 1, 1)

    if win_end <= win_start:
        win_start, win_end = max(min(event_start, chrom_len - 1), 1), min(event_end, chrom_len)
        if win_end <= win_start:
            win_end = win_start + 1

    return win_start, win_end, clamped


def sample_window(
    event_start: int,
    event_end: int,
    chrom_len: int,
    rng: np.random.Generator,
    **kw,
) -> tuple:
    """(win_start, win_end); see `sample_window_detail` for the clamp flag."""
    win_start, win_end = sample_window_detail(event_start, event_end, chrom_len, rng, **kw)[:2]
    return win_start, win_end
