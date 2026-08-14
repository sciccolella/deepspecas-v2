"""Randomized extraction windows around an event span (plan Phase 2 step 5).

Every row selected by `sim/select_events.py` so far uses its own true
span (the SUPPA event span for positives/hard negatives, the full gene
span for plain negatives) as the extraction window verbatim. That means
the labelled event's boundaries always coincide with the window's
boundaries -- a model could learn "junction near byte 0/N" instead of
splicing signal. This module derives a jittered, asymmetric window
instead: flank width log-uniform in [0.1x, 10x] of the event span, split
unevenly left/right so the event is not reliably centered, clipped to
chromosome bounds.
"""

from __future__ import annotations

import numpy as np


def sample_window(
    event_start: int,
    event_end: int,
    chrom_len: int,
    rng: np.random.Generator,
    flank_lo: float = 0.1,
    flank_hi: float = 10.0,
) -> tuple:
    """Returns (win_start, win_end), both 1-based inclusive like the
    region_start/region_end convention already used in design.tsv.

    Total flank budget is drawn once as a log-uniform ratio of the event
    span, then split into independent left/right shares (not halved) so
    the event ends up off-center in most draws.
    """
    span = max(event_end - event_start, 1)

    total_flank = span * float(np.exp(rng.uniform(np.log(flank_lo), np.log(flank_hi))))
    left_share = float(rng.uniform(0.0, 1.0))
    left_flank = total_flank * left_share
    right_flank = total_flank * (1.0 - left_share)

    win_start = max(int(round(event_start - left_flank)), 1)
    win_end = min(int(round(event_end + right_flank)), chrom_len)

    if win_end <= win_start:
        win_start, win_end = max(event_start, 1), min(event_end, chrom_len)
        if win_end <= win_start:
            win_end = win_start + 1

    return win_start, win_end
