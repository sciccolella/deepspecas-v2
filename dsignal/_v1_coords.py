"""Frozen copy of ``plot2s.get_coords_from_bam`` from the v1 repository.

Source: ``deepSpecas`` (v1) ``plot2s.py``, last modified at commit ``379cbba75``
(old hash; the branch point of ``signal-native-v2``). Vendored so that
``dsignal.verify`` can still check extractor parity against the v1 coverage
code without v1 being installed. DO NOT EDIT: the point of this module is that
it is the reference. v1's ``plot2s.py`` itself is not part of this repository.
"""

import numpy as np
import pysam

_REF_CONSUMING_OPS = frozenset((0, 2, 3, 7, 8))  # M, D, N, =, X
_SKIP_OP = 3  # N


def get_coords_from_bam(bam: pysam.AlignmentFile, region: tuple, mincov: int = 0):
    chrom, start, end = region
    n = end - start

    # Difference arrays: a run [s, e) is applied as +1 at s, -1 at e, then
    # recovered with a cumsum. One extra slot absorbs a decrement at e == n.
    cov_diff = np.zeros(n + 1, dtype=np.int32)
    skip_diff = np.zeros(n + 1, dtype=np.int32)

    for read in bam.fetch(chrom, start, end):
        cigartuples = read.cigartuples
        if cigartuples is None:
            continue

        pos = read.reference_start
        for op, oplen in cigartuples:
            if op not in _REF_CONSUMING_OPS:
                continue
            op_start = pos
            op_end = pos + oplen
            pos = op_end

            s = max(op_start - start, 0)
            e = min(op_end - start, n)
            if e <= s:
                continue

            diff = skip_diff if op == _SKIP_OP else cov_diff
            diff[s] += 1
            diff[e] -= 1

    covs = np.cumsum(cov_diff[:n])
    skips = np.cumsum(skip_diff[:n])

    mask = (covs + skips) <= mincov
    covs[mask] = 0
    skips[mask] = 0

    # Matches the original strict '> region[1]' pileup filter: base 0 of the
    # region is never populated.
    if n > 0:
        covs[0] = 0
        skips[0] = 0

    return covs.astype(np.uint32), skips.astype(np.uint32)
