"""Extraction windows around an event span.

Third iteration. The two earlier ones are kept in the record below because each
was rejected for a reason that is still true, and this one is not a return to
either.

**v1 -- flank as a log-uniform multiple of the event span** (plan Phase 2 step
5). This made window *width* a function of the event span, and event span
tracks the label (RI introns are short, CSE/NN spans are not). Window geometry
alone reached 46.3% accuracy on the 1260-event set.

**v2 -- span-independent width drawn from the pooled curated benchmark**
(`plans/signal-native-v2-window-geometry-fix.md`). Width was drawn from a
log-normal fitted to `benchmark/{A3,A5,ES,IR}.csv` (n=140, all four CSVs pooled
so the draw carried no label information), and the event was then placed
uniformly at random inside it. The width leak measured at chance afterwards.

**v3 -- this one: the window is the event span plus a small flank.** v2 matched
the benchmark's *marginal* width distribution while destroying the
*relationship* between window and event that the benchmark actually has.
Measured (`python -m dsignal.audit_noise padding`, matching all 140 curated
windows against the same SUPPA catalogue the simulation draws its events from):

    csv   matched   pad_left med [q10,q90]   pad_right med [q10,q90]
    A3     28/28        -1 [  -89,   -1]         -1 [   -1,   -1]
    A5     29/30        -1 [   -1,   -1]         -1 [   -1,   -1]
    ES     55/65       100 [  100,  100]        100 [  100,  100]
    IR     13/17         0 [    0,    0]          0 [    0,    0]

    event span / window width: median 1.000
    both edges within 200 bp: 120/125

The curated window *is* the event span, offset by a per-type constant (ES pads
100 bp on each edge; IR is exact; A3/A5 are off by one, a coordinate
convention). Under v2 the simulated event filled a label-dependent 4-98% of its
window (median 0.567: RI 0.22, A3 0.42, A5 0.49, NN 0.71, CSE 0.94) and windows
were ~2x too wide (log10 mean 3.82 vs the benchmark's 3.44). That is a large
train/test geometry shift, and it is the shift this version removes.

**What this knowingly reintroduces**, and why it is not simply the v1 leak
coming back: window width is again ~ the event span, so width again carries
label information. The difference is that *the deployment distribution has the
same coupling* -- the curator's window is the event span too. A model using it
is using something genuinely available at inference, not a simulation artifact.
That claim is not assumed: the geometry-only control is scored on the **benchmark**,
not only on sim CV, and `sim/width_leak_check.py` is expected to rise. Both
numbers get reported rather than suppressed. See
`plans/signal-native-v2-audit-geometry.md` (Track B).

The flank is drawn **label-independently**, uniform over [0, MAX_FLANK] bp per
edge, rather than reproducing the per-type constants above: a per-class padding
constant would be pure label leakage with no counterpart at inference, where
nobody knows the class. Uniform-over-both-edges reproduces the *magnitude* of a
curator's padding (0-100 bp) and leaves the event boundaries non-deterministic,
so "the window edge is exactly the event edge" is not itself learnable.
"""

from __future__ import annotations

import numpy as np

#: Curated padding is 0 bp (IR), -1 bp (A3/A5, an off-by-one) or 100 bp (ES) per
#: edge. Drawn uniform over this range instead of per type -- see the module
#: docstring on why the per-type constant is not reproduced.
MAX_FLANK = 100


def sample_window_detail(
    event_start: int,
    event_end: int,
    chrom_len: int,
    rng: np.random.Generator,
    max_flank: int = MAX_FLANK,
) -> tuple:
    """Returns (win_start, win_end, clamped), coordinates 1-based inclusive like
    the region_start/region_end convention already used in design.tsv.

    ``clamped`` is retained for the caller's reporting and is True only when the
    chromosome edge forced the window to lose part of its drawn flank. (It used
    to also flag a drawn width narrower than the event span -- the far more
    common case under the superseded v2 sampler. That cannot arise here: the
    window is built outward from the span rather than drawn independently of it.)
    """
    left = int(rng.integers(0, max_flank + 1))
    right = int(rng.integers(0, max_flank + 1))

    win_start = event_start - left
    win_end = event_end + right

    clamped = False
    if win_start < 1:
        win_start, clamped = 1, True
    if win_end > chrom_len:
        win_end, clamped = chrom_len, True

    # a degenerate span (event outside the chromosome, never seen in practice)
    # must still yield a usable interval rather than an empty fetch
    if win_end <= win_start:
        win_start = max(min(event_start, chrom_len - 1), 1)
        win_end = min(max(event_end, win_start + 1), chrom_len)
        clamped = True

    return win_start, win_end, clamped


def sample_window(
    event_start: int,
    event_end: int,
    chrom_len: int,
    rng: np.random.Generator,
    **kw,
) -> tuple:
    """(win_start, win_end); see `sample_window_detail` for the clamp flag."""
    return sample_window_detail(event_start, event_end, chrom_len, rng, **kw)[:2]
